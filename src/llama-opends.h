#pragma once

#include <cstddef>
#include <memory>
#include <sys/types.h>

// thin wrapper over the OpenDS direct storage API (cuFile-compatible), used by the
// model loader to read tensor data from NVMe directly into registered buffers.
// built as no-op stubs when LLAMA_USE_OPENDS is not defined: ok() returns false
// and the loader falls back to the regular read paths.
struct llama_opends {
    llama_opends();
    ~llama_opends();

    bool ok() const;

    // register the O_DIRECT fd for file idx (cached); false if registration failed
    bool file_ok(int idx, int fd);

    // register a destination buffer (cached); false if the backend rejected it
    bool buffer_ok(void * base, size_t size);

    // submit an async read into a registered buffer;
    // falls back to a blocking read if the backend has no async
    bool read_async(int idx, void * base, size_t size, off_t file_offset, off_t buf_offset);

    // wait for all submitted async reads; false if any failed
    bool drain();

    // close the shared driver session once nothing uses it any more.
    // the session outlives a single model on purpose: opening and closing it
    // costs ~70 ms, which every load would otherwise pay after its last byte
    // already sits in VRAM
    static void shutdown();

    struct impl;
    std::unique_ptr<impl> pimpl;
};
