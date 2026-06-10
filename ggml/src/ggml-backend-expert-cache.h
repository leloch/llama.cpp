#pragma once

#include <stdint.h>
#include <stddef.h>

// Expert cache v3 — dynamic VRAM cache for MoE expert weights on CPU-resident
// MUL_MAT_ID. Integration point is the CPU mul_mat_id kernel itself: thread 0
// dispatches cached expert rows to the GPU while the remaining threads compute
// the uncached rows, then results are collected into dst before the node ends.
//
// This table is the bridge between ggml-cpu (which cannot link CUDA) and the
// CUDA backend (which registers the implementations at backend-reg time).
// begin/plan/dispatch/collect for one node are called from a single thread
// (ith == 0); the implementation may assume no concurrent node processing.

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_expert_cache_v3_api {
    // Decide whether the cache engages for this MUL_MAT_ID node.
    // Returns the device id to use (>= 0) or -1 to stay on the pure-CPU path.
    // Performs lazy per-device initialization on first use and selects the
    // internal slot pool matching (expert_size, wtype).
    //   tensor_name: src0->name (stable cache key source, e.g. "blk.7.ffn_up_exps.weight")
    //   host_base:   src0->data (source for async inserts)
    //   expert_size: src0->nb[2] (bytes per expert)
    //   n_in/n_out:  src0->ne[0] / src0->ne[1]
    //   wtype:       src0->type
    //   n_expert:    src0->ne[2]
    //   n_tokens:    ids->ne[1] (cache engages only when == 1)
    int (*begin)(const char * tensor_name, const void * host_base, size_t expert_size,
                 int64_t n_in, int64_t n_out, int wtype, int64_t n_expert, int64_t n_tokens);

    // For each of the n_ids expert ids: slot_idx[k] = cache slot index (hit) or
    // -1 (miss; CPU computes the row, an async insert may be enqueued).
    // Returns the number of hits.
    int (*plan)(int dev, const int32_t * ids, int n_ids, int32_t * slot_idx);

    // One batched GPU launch computing all n_hits rows:
    //   out_row[i] (n_out floats) = W[slot_idx_compact[i]] . act_rows[i]
    // act_rows are host fp32 pointers (they may all be the same row for
    // gate/up-style nodes; distinct rows for down-style nodes).
    // Asynchronous; results are pulled into dst by collect().
    void (*dispatch)(int dev, int wtype, int64_t n_in, int64_t n_out, int n_hits,
                     const int32_t * slot_idx_compact, const float * const * act_rows);

    // Synchronize the device's compute stream and copy the n_hits result rows
    // (in dispatch order) into dst_rows[0..n_hits-1] (n_out floats each).
    void (*collect)(int dev, int n_hits, float * const * dst_rows, int64_t n_out);

    // Periodic stats logging (rate-limited internally).
    void (*stats)(void);
};

// Zero-initialized in ggml-backend.cpp; populated by the CUDA backend in
// ggml_backend_cuda_reg() when the cache is enabled (LLAMA_EC3=1).
extern struct ggml_expert_cache_v3_api ggml_expert_cache_v3;

#ifdef __cplusplus
}
#endif
