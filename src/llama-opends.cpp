#include "llama-opends.h"

#include "llama-impl.h"

#include "ggml.h"

#ifdef LLAMA_USE_OPENDS

#include <opends.h>

#include <deque>
#include <mutex>
#include <unordered_map>

struct llama_opends::impl {
    bool opened = false;

    struct file_entry {
        opends_handle_t fh = nullptr;
        bool tried = false;
        bool ok    = false;
    };

    std::unordered_map<int, file_entry> files;
    std::unordered_map<void *, bool> bufs;

    // futures must keep a stable address until awaited, deque never moves them
    struct pending {
        opends_async_future_t fut;
        opends_handle_t fh;
        void * base;
        size_t size;
        off_t file_offset;
        off_t buf_offset;
    };
    std::deque<pending> pendings;
    bool async_supported = true;
};

// the driver session is shared by every model in the process. aisio spends
// ~70 ms in opends_driver_close tearing down its i/o threads, HOMI connection
// and cuda context, so a per-model session would pay that on every load and
// every model swap, after the weights are already in VRAM.
static std::mutex g_driver_mutex;
static int  g_driver_refs = 0;
static bool g_driver_open = false;

static bool driver_acquire() {
    std::lock_guard<std::mutex> lock(g_driver_mutex);
    if (!g_driver_open) {
        opends_error_t err = opends_driver_open();
        if (err.err != OPENDS_SUCCESS) {
            LLAMA_LOG_WARN("%s: opends_driver_open failed: %s\n", __func__, opends_op_status_error(err.err));
            return false;
        }
        g_driver_open = true;
    }
    g_driver_refs++;
    return true;
}

static void driver_release() {
    std::lock_guard<std::mutex> lock(g_driver_mutex);
    if (g_driver_refs > 0) {
        g_driver_refs--;
    }
}

void llama_opends::shutdown() {
    std::lock_guard<std::mutex> lock(g_driver_mutex);
    if (!g_driver_open || g_driver_refs > 0) {
        return;
    }
    opends_driver_close();
    g_driver_open = false;
}

llama_opends::llama_opends() : pimpl(new impl()) {
    pimpl->opened = driver_acquire();
}

llama_opends::~llama_opends() {
    if (!pimpl->opened) {
        return;
    }
    drain();
    const int64_t t0 = ggml_time_us();
    for (auto & it : pimpl->bufs) {
        if (it.second) {
            opends_buf_deregister(it.first);
        }
    }
    for (auto & it : pimpl->files) {
        if (it.second.ok) {
            opends_handle_deregister(it.second.fh);
        }
    }
    driver_release();
    LLAMA_LOG_DEBUG("%s: deregister %.1f ms\n", __func__, (ggml_time_us() - t0) / 1000.0);
}

bool llama_opends::ok() const {
    return pimpl->opened;
}

bool llama_opends::file_ok(int idx, int fd) {
    if (!pimpl->opened) {
        return false;
    }
    auto & e = pimpl->files[idx];
    if (!e.tried) {
        e.tried = true;
        opends_error_t err = opends_handle_register(&e.fh, fd);
        e.ok = err.err == OPENDS_SUCCESS;
        if (!e.ok) {
            LLAMA_LOG_WARN("%s: opends_handle_register failed for file %d: %s\n", __func__, idx, opends_op_status_error(err.err));
        }
    }
    return e.ok;
}

bool llama_opends::buffer_ok(void * base, size_t size) {
    if (!pimpl->opened) {
        return false;
    }
    auto it = pimpl->bufs.find(base);
    if (it == pimpl->bufs.end()) {
        opends_error_t err = opends_buf_register(base, size, 0);
        const bool ok = err.err == OPENDS_SUCCESS;
        if (!ok) {
            LLAMA_LOG_WARN("%s: opends_buf_register(%p, %zu) failed: %s\n",
                __func__, base, size, opends_op_status_error(err.err));
        }
        it = pimpl->bufs.emplace(base, ok).first;
    }
    return it->second;
}

// read a whole span with the blocking API, retrying short reads
static bool sync_read(opends_handle_t fh, void * base, size_t size, off_t file_offset, off_t buf_offset) {
    while (size > 0) {
        ssize_t n = opends_sync_read(fh, base, size, file_offset, buf_offset);
        if (n <= 0) {
            LLAMA_LOG_WARN("%s: opends_sync_read(size=%zu, file_offset=%lld) failed: %s\n",
                __func__, size, (long long) file_offset, n < 0 ? OPENDS_ERRSTR(n) : "short read");
            return false;
        }
        size        -= n;
        file_offset += n;
        buf_offset  += n;
    }
    return true;
}

bool llama_opends::read_async(int idx, void * base, size_t size, off_t file_offset, off_t buf_offset) {
    auto fit = pimpl->files.find(idx);
    if (fit == pimpl->files.end() || !fit->second.ok) {
        return false;
    }
    auto bit = pimpl->bufs.find(base);
    if (bit == pimpl->bufs.end() || !bit->second) {
        return false;
    }
    if (pimpl->async_supported) {
        pimpl->pendings.push_back({{}, fit->second.fh, base, size, file_offset, buf_offset});
        auto & p = pimpl->pendings.back();
        opends_error_t err = opends_async_read(p.fh, base, size, file_offset, buf_offset, &p.fut);
        if (err.err == OPENDS_SUCCESS) {
            return true;
        }
        pimpl->pendings.pop_back();
        if (err.err != OPENDS_ASYNC_NOT_SUPPORTED) {
            LLAMA_LOG_WARN("%s: opends_async_read(size=%zu, file_offset=%lld) failed: %s\n",
                __func__, size, (long long) file_offset, opends_op_status_error(err.err));
            return false;
        }
        pimpl->async_supported = false;
    }
    return sync_read(fit->second.fh, base, size, file_offset, buf_offset);
}

bool llama_opends::drain() {
    bool ok = true;
    while (!pimpl->pendings.empty()) {
        auto & p = pimpl->pendings.front();
        ssize_t n = opends_async_await(&p.fut);
        if (n < 0) {
            LLAMA_LOG_WARN("%s: opends_async_read(size=%zu, file_offset=%lld) failed: %s\n",
                __func__, p.size, (long long) p.file_offset, OPENDS_ERRSTR(n));
            ok = false;
        } else if ((size_t) n < p.size) {
            // finish a short read through the sync path
            ok &= sync_read(p.fh, p.base, p.size - n, p.file_offset + n, p.buf_offset + n);
        }
        pimpl->pendings.pop_front();
    }
    return ok;
}

#else

struct llama_opends::impl {};

llama_opends::llama_opends() {
    LLAMA_LOG_WARN("%s: llama.cpp was built without OpenDS support (LLAMA_OPENDS=OFF)\n", __func__);
}

llama_opends::~llama_opends() = default;

bool llama_opends::ok() const {
    return false;
}

bool llama_opends::file_ok(int, int) {
    return false;
}

bool llama_opends::buffer_ok(void *, size_t) {
    return false;
}

bool llama_opends::read_async(int, void *, size_t, off_t, off_t) {
    return false;
}

bool llama_opends::drain() {
    return true;
}

void llama_opends::shutdown() {
}

#endif
