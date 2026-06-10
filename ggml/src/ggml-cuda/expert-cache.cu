// Expert cache v3 — dynamic VRAM cache for CPU-resident MoE expert weights.
//
// Design (from first principles, replacing the v2 scheduler-hook prototype):
//  - Integration lives inside the CPU mul_mat_id kernel (see ggml-cpu.c):
//    thread 0 plans hits/misses and dispatches cached rows to the GPU in ONE
//    batched kernel launch while the remaining threadpool threads compute the
//    miss rows. Results are collected into dst before the node ends, so
//    correctness holds under any split topology and no shared tensors are
//    ever mutated.
//  - The cache fills ONLY from token-generation misses (n_tokens == 1), via
//    dedicated insert worker threads that copy expert weights host->VRAM off
//    the hot path. Prompt processing never touches the cache.
//  - Slots live in per-(expert_size, type) pools whose slot stride equals the
//    source tensor's nb[2] exactly, so the batched mmvq kernel can index the
//    pool like a regular expert tensor (strides are in block units).
//  - Eviction: plain LRU per pool. Capacity on this class of hardware exceeds
//    the decode working set, so eviction policy is not the binding constraint.
//
// Keys are FNV-1a hashes of the weight tensor's name (stable across contexts
// and mmap remaps) mixed with the expert id.

#include "expert-cache.cuh"
#include "common.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "../ggml-backend-expert-cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define EC3_MAX_DEV    8
#define EC3_MAX_POOLS  8
#define EC3_LOG(...)   fprintf(stderr, __VA_ARGS__)

struct ec3_slot {
    uint64_t key;
    int      prev;
    int      next;
    bool     valid;     // contents complete, lookups may hit
    bool     queued;    // insert copy queued or in flight
};

struct ec3_pool {
    size_t expert_size = 0;   // == slot stride == source tensor nb[2]
    int    wtype       = -1;
    char * slab        = nullptr;
    int    n_slots     = 0;
    int    n_used      = 0;

    std::vector<ec3_slot> slots;
    std::unordered_map<uint64_t, int> map;
    int lru_head = -1;
    int lru_tail = -1;
};

struct ec3_device {
    ec3_pool pools[EC3_MAX_POOLS];
    int      n_pools = 0;

    cudaStream_t compute_stream = nullptr;

    // staging for one node's batched dispatch
    int32_t * h_ids = nullptr;  int32_t * d_ids = nullptr;  size_t ids_cap = 0;   // count
    float   * h_act = nullptr;  float   * d_act = nullptr;  size_t act_cap = 0;   // bytes
    void    * d_act_q8 = nullptr;                           size_t act_q8_cap = 0;
    float   * d_out = nullptr;                              size_t d_out_cap = 0;
    float   * h_out = nullptr;                              size_t h_out_cap = 0; // pinned
    int       out_rows = 0;

    // stats
    long long hits = 0, misses = 0, inserts = 0, evictions = 0;
    long long insert_skips = 0, queued_misses = 0;
};

struct ec3_job {
    int        dev;
    int        pool;
    uint64_t   key;
    int        slot_idx;
    const void * src;
    size_t     bytes;
};

struct ec3_global {
    bool   enabled  = false;
    int    n_dev    = 0;
    size_t budget_mb = 0;        // 0 = auto (fraction of free VRAM at init)
    int    inserts_per_plan = 8; // max inserts enqueued per plan() call
    int    queue_max        = 512;
    int    n_workers        = 4;
    int    stats_every      = 0; // log every N collect() calls (0 = off)

    ec3_device dev[EC3_MAX_DEV];

    // insert queue + workers
    std::mutex              mu;  // guards pools/queue of all devices
    std::condition_variable cv;
    std::deque<ec3_job>     queue;
    bool                    workers_started = false;

    // current node context (begin..collect happen on one thread)
    uint64_t     cur_key_base = 0;
    const void * cur_host_base = nullptr;
    size_t       cur_expert_size = 0;
    int64_t      cur_n_expert = 0;
    int          cur_pool = -1;

    long long collect_calls = 0;
};

// intentionally leaked: detached worker threads reference this state through
// process exit; running its destructor would tear a condition variable out
// from under a waiting thread (observed as a hang in atexit).
static ec3_global & g = *new ec3_global();

// LLAMA_EC3_DEBUG=1: trace the first calls of each API to locate stalls
static int g_dbg = -1;
static long long g_dbg_n = 0;
#define EC3_DBG(...) do { \
        if (g_dbg < 0) { const char * _e = getenv("LLAMA_EC3_DEBUG"); g_dbg = _e ? atoi(_e) : 0; } \
        if (g_dbg > 0 && g_dbg_n++ < 400) { EC3_LOG(__VA_ARGS__); fflush(stderr); } \
    } while (0)

static uint64_t ec3_fnv1a(const char * s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 0x100000001b3ULL;
    }
    return h;
}

static inline uint64_t ec3_key(uint64_t name_hash, int eid) {
    return name_hash ^ ((uint64_t)(uint32_t)eid * 0x9E3779B97F4A7C15ULL);
}

// ---- LRU helpers (caller holds g.mu) ---------------------------------------

static void ec3_lru_remove(ec3_pool & p, int idx) {
    ec3_slot & s = p.slots[idx];
    if (s.prev >= 0) p.slots[s.prev].next = s.next; else p.lru_head = s.next;
    if (s.next >= 0) p.slots[s.next].prev = s.prev; else p.lru_tail = s.prev;
    s.prev = s.next = -1;
}

static void ec3_lru_push_back(ec3_pool & p, int idx) {
    ec3_slot & s = p.slots[idx];
    s.prev = p.lru_tail;
    s.next = -1;
    if (p.lru_tail >= 0) p.slots[p.lru_tail].next = idx; else p.lru_head = idx;
    p.lru_tail = idx;
}

// ---- insert workers ----------------------------------------------------------

static void ec3_worker_main() {
    // per-worker pinned staging buffer + per-device copy streams: the host
    // memcpy runs at RAM speed on this thread, the H2D is a true async DMA on
    // a dedicated stream — no pageable-copy driver contention with the
    // dispatch path's kernel launches.
    char * stage = nullptr;
    size_t stage_cap = 0;
    cudaStream_t cstream[EC3_MAX_DEV] = {};

    for (;;) {
        ec3_job job;
        {
            std::unique_lock<std::mutex> lk(g.mu);
            g.cv.wait(lk, []{ return !g.queue.empty(); });
            job = g.queue.front();
            g.queue.pop_front();
        }

        ec3_pool & p = g.dev[job.dev].pools[job.pool];
        cudaSetDevice(job.dev);
        EC3_DBG("[ec3-dbg] worker job dev=%d slot=%d bytes=%zu\n", job.dev, job.slot_idx, job.bytes);
        char * dst = p.slab + (size_t)job.slot_idx * p.expert_size;

        cudaError_t err = cudaSuccess;
        if (stage_cap < job.bytes) {
            if (stage) cudaFreeHost(stage);
            err = cudaMallocHost((void **)&stage, job.bytes * 2);
            stage_cap = (err == cudaSuccess) ? job.bytes * 2 : 0;
            if (err != cudaSuccess) stage = nullptr;
        }
        if (!cstream[job.dev]) {
            cudaStreamCreateWithFlags(&cstream[job.dev], cudaStreamNonBlocking);
        }
        if (err == cudaSuccess && stage && cstream[job.dev]) {
            memcpy(stage, job.src, job.bytes);
            err = cudaMemcpyAsync(dst, stage, job.bytes, cudaMemcpyHostToDevice, cstream[job.dev]);
            if (err == cudaSuccess) {
                err = cudaStreamSynchronize(cstream[job.dev]);
            }
        } else if (err == cudaSuccess) {
            // pinned alloc failed: fall back to a direct pageable copy
            err = cudaMemcpy(dst, job.src, job.bytes, cudaMemcpyHostToDevice);
        }

        {
            std::lock_guard<std::mutex> lk(g.mu);
            ec3_slot & s = p.slots[job.slot_idx];
            if (s.queued && s.key == job.key) {
                s.queued = false;
                if (err == cudaSuccess) {
                    s.valid = true;
                } else {
                    p.map.erase(s.key);
                    s.key = 0;
                }
            }
            if (err != cudaSuccess) {
                cudaGetLastError();
                static int warned = 0;
                if (warned++ < 3) {
                    EC3_LOG("[ec3] insert copy failed: %s\n", cudaGetErrorString(err));
                }
            }
        }
    }
}

static void ec3_start_workers() {
    if (g.workers_started) return;
    g.workers_started = true;
    for (int i = 0; i < g.n_workers; i++) {
        std::thread(ec3_worker_main).detach();
    }
}

// ---- init ---------------------------------------------------------------------

// Records distinct (expert_size, wtype) pool shapes per device until a tensor
// repeats (one full decode token observed), then allocates all pools.
struct ec3_warmup {
    std::unordered_set<uint64_t> seen;
    struct shape { size_t size; int wtype; };
    std::vector<shape> shapes[EC3_MAX_DEV];
    bool done = false;
};
static ec3_warmup g_warm;

static void ec3_dev_alloc(int di) {
    ec3_device & d = g.dev[di];
    if (d.compute_stream) return;

    auto & shapes = g_warm.shapes[di];
    if (shapes.empty()) return;

    ggml_cuda_set_device(di);

    size_t budget;
    if (g.budget_mb > 0) {
        budget = g.budget_mb << 20;
    } else {
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        budget = (size_t)(free_mem * 0.70);
    }

    size_t sum_sizes = 0;
    for (auto & sh : shapes) sum_sizes += sh.size;
    int n_slots = (int)(budget / sum_sizes);   // equal slot count per pool
    if (n_slots < 64) {
        EC3_LOG("[ec3] dev=%d budget too small (%zu MB for %zu pools)\n", di, budget >> 20, shapes.size());
        return;
    }

    for (size_t i = 0; i < shapes.size() && (int)i < EC3_MAX_POOLS; i++) {
        ec3_pool & p = d.pools[d.n_pools];
        char * slab = nullptr;
        int ns = n_slots;
        cudaError_t err = cudaMalloc((void **)&slab, (size_t)ns * shapes[i].size);
        if (err != cudaSuccess) {
            ns /= 2;
            err = cudaMalloc((void **)&slab, (size_t)ns * shapes[i].size);
            if (err != cudaSuccess) {
                EC3_LOG("[ec3] dev=%d pool alloc failed: %s\n", di, cudaGetErrorString(err));
                continue;
            }
        }
        p.expert_size = shapes[i].size;
        p.wtype       = shapes[i].wtype;
        p.slab        = slab;
        p.n_slots     = ns;
        p.slots.assign(ns, ec3_slot{0, -1, -1, false, false});
        d.n_pools++;
        EC3_LOG("[ec3] dev=%d pool[%d]: type=%d slot=%zu KB slots=%d total=%zu MB\n",
                di, d.n_pools - 1, p.wtype, p.expert_size >> 10, ns, ((size_t)ns * p.expert_size) >> 20);
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&d.compute_stream, cudaStreamNonBlocking));
    ec3_start_workers();
}

// ---- API: begin -----------------------------------------------------------------

static int ec3_begin(const char * name, const void * host_base, size_t expert_size,
                     int64_t n_in, int64_t n_out, int wtype, int64_t n_expert, int64_t n_tokens) {
    GGML_UNUSED(n_in); GGML_UNUSED(n_out);

    if (!g.enabled || n_tokens != 1) return -1;

    const char * p = strstr(name, "blk.");
    if (!p || !strstr(name, "_exps")) return -1;
    const int blk = atoi(p + 4);
    const int di  = blk % g.n_dev;

    const uint64_t kb = ec3_fnv1a(name);

    if (!g_warm.done) {
        bool known = false;
        for (auto & sh : g_warm.shapes[di]) {
            if (sh.size == expert_size && sh.wtype == wtype) { known = true; break; }
        }
        if (!known) {
            g_warm.shapes[di].push_back({expert_size, wtype});
            EC3_DBG("[ec3-dbg] warm new shape %s blk=%d dev=%d size=%zu type=%d\n",
                    name, blk, di, expert_size, wtype);
        }

        if (g_warm.seen.count(kb)) {
            g_warm.done = true;
            EC3_LOG("[ec3] warmup done (trigger=%s); shapes/dev:", name);
            for (int i = 0; i < g.n_dev; i++) EC3_LOG(" %zu", g_warm.shapes[i].size());
            EC3_LOG("\n");
            for (int i = 0; i < g.n_dev; i++) ec3_dev_alloc(i);
        } else {
            g_warm.seen.insert(kb);
            return -1;
        }
    }

    ec3_device & d = g.dev[di];
    int pi = -1;
    for (int i = 0; i < d.n_pools; i++) {
        if (d.pools[i].expert_size == expert_size && d.pools[i].wtype == wtype) { pi = i; break; }
    }
    if (pi < 0) return -1;

    EC3_DBG("[ec3-dbg] begin %s dev=%d pool=%d\n", name, di, pi);

    g.cur_key_base    = kb;
    g.cur_host_base   = host_base;
    g.cur_expert_size = expert_size;
    g.cur_n_expert    = n_expert;
    g.cur_pool        = pi;
    return di;
}

// ---- API: plan --------------------------------------------------------------------

static int ec3_plan(int di, const int32_t * ids, int n_ids, int32_t * slot_idx) {
    ec3_device & d = g.dev[di];
    ec3_pool   & p = d.pools[g.cur_pool];

    int n_hits = 0;
    int inserts_left = g.inserts_per_plan;

    std::lock_guard<std::mutex> lk(g.mu);

    for (int k = 0; k < n_ids; k++) {
        slot_idx[k] = -1;
        const int eid = ids[k];
        if (eid < 0 || eid >= g.cur_n_expert) continue;
        const uint64_t key = ec3_key(g.cur_key_base, eid);

        auto it = p.map.find(key);
        if (it != p.map.end()) {
            const int si = it->second;
            ec3_slot & s = p.slots[si];
            if (s.valid) {
                ec3_lru_remove(p, si);
                ec3_lru_push_back(p, si);
                slot_idx[k] = si;
                d.hits++;
                n_hits++;
            } else {
                // insert still queued/in-flight: CPU computes the row this time
                d.queued_misses++;
                d.misses++;
            }
            continue;
        }

        d.misses++;

        // ---- enqueue async insert (budgeted) ----
        if (inserts_left <= 0 || (int)g.queue.size() >= g.queue_max) {
            d.insert_skips++;
            continue;
        }
        // admission throttle at capacity: when the pool is full, churn (evict +
        // re-copy on every miss) steals host RAM bandwidth from the CPU matmuls.
        // Admit only a fraction of misses so the content still adapts but the
        // copy traffic stays bounded.
        if (p.n_used >= p.n_slots && (d.misses & 7) != 0) {
            d.insert_skips++;
            continue;
        }

        int si = -1;
        if (p.n_used < p.n_slots) {
            si = p.n_used++;
        } else {
            int cand = p.lru_head;
            int guard = 0;
            while (cand >= 0 && p.slots[cand].queued && guard++ < 64) cand = p.slots[cand].next;
            if (cand < 0 || p.slots[cand].queued) { d.insert_skips++; continue; }
            si = cand;
            ec3_slot & old = p.slots[si];
            if (old.valid || old.queued) {
                p.map.erase(old.key);
                d.evictions++;
            }
            ec3_lru_remove(p, si);
        }

        const char * src = (const char *)g.cur_host_base + (size_t)eid * g.cur_expert_size;

        p.slots[si] = ec3_slot{key, -1, -1, false, true};
        ec3_lru_push_back(p, si);
        p.map[key] = si;
        d.inserts++;
        inserts_left--;

        g.queue.push_back(ec3_job{di, g.cur_pool, key, si, src, g.cur_expert_size});
        g.cv.notify_one();
    }

    EC3_DBG("[ec3-dbg] plan dev=%d hits=%d q=%zu\n", di, n_hits, g.queue.size());
    return n_hits;
}

// ---- API: dispatch ------------------------------------------------------------------

static void ec3_dispatch(int di, int wtype_int, int64_t n_in, int64_t n_out, int n_hits,
                         const int32_t * slot_idx_compact, const float * const * act_rows) {
    if (n_hits <= 0) return;
    ec3_device & d = g.dev[di];
    ec3_pool   & p = d.pools[g.cur_pool];
    ggml_cuda_set_device(di);
    cudaStream_t st = d.compute_stream;

    const ggml_type wtype = (ggml_type)wtype_int;
    const int64_t n_in_padded = ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;

    // distinct activation rows: all-same (gate/up) -> 1 row; else one per hit
    bool shared_act = true;
    for (int i = 1; i < n_hits; i++) {
        if (act_rows[i] != act_rows[0]) { shared_act = false; break; }
    }
    const int act_n = shared_act ? 1 : n_hits;

    // grow staging
    if (d.ids_cap < (size_t)n_hits) {
        const size_t cap = n_hits * 2 + 8;
        if (d.h_ids) cudaFreeHost(d.h_ids);
        if (d.d_ids) cudaFree(d.d_ids);
        CUDA_CHECK(cudaMallocHost((void **)&d.h_ids, cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc((void **)&d.d_ids, cap * sizeof(int32_t)));
        d.ids_cap = cap;
    }
    const size_t need_act = (size_t)act_n * n_in * sizeof(float);
    if (d.act_cap < need_act) {
        const size_t cap = need_act * 2;
        if (d.h_act) cudaFreeHost(d.h_act);
        if (d.d_act) cudaFree(d.d_act);
        CUDA_CHECK(cudaMallocHost((void **)&d.h_act, cap));
        CUDA_CHECK(cudaMalloc((void **)&d.d_act, cap));
        d.act_cap = cap;
    }
    const size_t need_q8 = (size_t)act_n * (n_in_padded / QK8_1) * sizeof(block_q8_1);
    if (d.act_q8_cap < need_q8) {
        const size_t cap = need_q8 * 2;
        if (d.d_act_q8) cudaFree(d.d_act_q8);
        CUDA_CHECK(cudaMalloc(&d.d_act_q8, cap));
        d.act_q8_cap = cap;
    }
    const size_t need_out = (size_t)n_hits * n_out * sizeof(float);
    if (d.d_out_cap < need_out) {
        const size_t cap = need_out * 2 + 65536;
        if (d.d_out) cudaFree(d.d_out);
        CUDA_CHECK(cudaMalloc((void **)&d.d_out, cap));
        d.d_out_cap = cap;
    }
    if (d.h_out_cap < need_out) {
        const size_t cap = need_out * 2 + 65536;
        if (d.h_out) cudaFreeHost(d.h_out);
        CUDA_CHECK(cudaMallocHost((void **)&d.h_out, cap));
        d.h_out_cap = cap;
    }

    // gather to pinned staging, then async H2D
    for (int i = 0; i < n_hits; i++) d.h_ids[i] = slot_idx_compact[i];
    CUDA_CHECK(cudaMemcpyAsync(d.d_ids, d.h_ids, n_hits * sizeof(int32_t), cudaMemcpyHostToDevice, st));
    for (int i = 0; i < act_n; i++) {
        memcpy(d.h_act + (size_t)i * n_in, act_rows[i], n_in * sizeof(float));
    }
    CUDA_CHECK(cudaMemcpyAsync(d.d_act, d.h_act, need_act, cudaMemcpyHostToDevice, st));

    quantize_row_q8_1_cuda(d.d_act, /*ids=*/nullptr, d.d_act_q8, wtype,
                           n_in, /*s01=*/n_in, /*s02=*/(int64_t)act_n * n_in, /*s03=*/(int64_t)act_n * n_in,
                           n_in_padded, /*ne1=*/act_n, /*ne2=*/1, /*ne3=*/1, st);

    ggml_cuda_ec3_mmv(p.slab, wtype, (const char *)d.d_act_q8, d.d_ids, d.d_out,
                      n_in, n_out, p.n_slots, (int64_t)p.expert_size,
                      n_hits, /*act_rows=*/act_n, st);

    d.out_rows = n_hits;
    EC3_DBG("[ec3-dbg] dispatch dev=%d hits=%d act_n=%d n_in=%lld n_out=%lld\n",
            di, n_hits, act_n, (long long)n_in, (long long)n_out);
}

// ---- API: collect --------------------------------------------------------------------

static void ec3_stats(void);

static void ec3_collect(int di, int n_hits, float * const * dst_rows, int64_t n_out) {
    ec3_device & d = g.dev[di];
    if (n_hits != d.out_rows) {
        EC3_LOG("[ec3] BUG: collect rows %d != dispatched %d\n", n_hits, d.out_rows);
    }
    ggml_cuda_set_device(di);
    EC3_DBG("[ec3-dbg] collect dev=%d rows=%d pre-sync\n", di, d.out_rows);
    const size_t bytes = (size_t)d.out_rows * n_out * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(d.h_out, d.d_out, bytes, cudaMemcpyDeviceToHost, d.compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(d.compute_stream));
    EC3_DBG("[ec3-dbg] collect dev=%d post-sync\n", di);
    for (int i = 0; i < d.out_rows && i < n_hits; i++) {
        memcpy(dst_rows[i], d.h_out + (size_t)i * n_out, n_out * sizeof(float));
    }
    d.out_rows = 0;

    if (g.stats_every > 0 && ++g.collect_calls % g.stats_every == 0) {
        ec3_stats();
    }
}

// ---- API: stats ----------------------------------------------------------------------

static void ec3_stats(void) {
    for (int i = 0; i < g.n_dev; i++) {
        ec3_device & d = g.dev[i];
        if (!d.compute_stream) continue;
        const long long tot = d.hits + d.misses;
        int used = 0, slots = 0;
        for (int pi = 0; pi < d.n_pools; pi++) { used += d.pools[pi].n_used; slots += d.pools[pi].n_slots; }
        EC3_LOG("[ec3] dev=%d hits=%lld/%lld (%.1f%%) inserts=%lld evict=%lld skip=%lld queued-miss=%lld used=%d/%d q=%zu\n",
                i, d.hits, tot, tot ? 100.0 * d.hits / tot : 0.0,
                d.inserts, d.evictions, d.insert_skips, d.queued_misses,
                used, slots, g.queue.size());
    }
}

// ---- self-test ---------------------------------------------------------------------------
//
// LLAMA_EC3_SELFTEST=1 runs at registration: builds a synthetic quantized pool,
// runs the full plan-free dispatch+collect path, and compares against a host
// reference matvec on dequantized weights. No model required — this validates
// the batched mmvq stride mapping and the staging logic in seconds.

static bool ec3_selftest_one(int di, ggml_type wtype, int64_t n_in, int64_t n_out,
                             int n_hits, bool shared_act) {
    const int n_slots = 16;
    const size_t row_bytes  = ggml_row_size(wtype, n_in);
    const size_t slot_bytes = (size_t)n_out * row_bytes;

    // build random fp32 weights, quantize per slot
    std::vector<float> wf((size_t)n_slots * n_out * n_in);
    for (size_t i = 0; i < wf.size(); i++) wf[i] = 0.02f * (float)((int)(i * 2654435761u % 1000) - 500) / 500.0f;
    std::vector<char> wq((size_t)n_slots * slot_bytes);
    for (int s = 0; s < n_slots; s++) {
        ggml_quantize_chunk(wtype, wf.data() + (size_t)s * n_out * n_in,
                            wq.data() + (size_t)s * slot_bytes, 0, n_out, n_in, nullptr);
    }

    // upload pool
    ggml_cuda_set_device(di);
    char * d_pool = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_pool, wq.size()));
    CUDA_CHECK(cudaMemcpy(d_pool, wq.data(), wq.size(), cudaMemcpyHostToDevice));

    // fabricate device + pool state
    ec3_device & d = g.dev[di];
    if (!d.compute_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&d.compute_stream, cudaStreamNonBlocking));
    }
    ec3_pool & p = d.pools[0];
    p.expert_size = slot_bytes;
    p.wtype = wtype;
    p.slab = d_pool;
    p.n_slots = n_slots;
    g.cur_pool = 0;

    // activations
    std::vector<float> act((size_t)n_hits * n_in);
    for (size_t i = 0; i < act.size(); i++) act[i] = 0.05f * (float)((int)(i * 40503u % 997) - 498) / 498.0f;

    int32_t slot_ids[64];
    const float * act_rows[64];
    for (int i = 0; i < n_hits; i++) {
        slot_ids[i] = (i * 5 + 3) % n_slots;
        act_rows[i] = shared_act ? act.data() : act.data() + (size_t)i * n_in;
    }

    ec3_dispatch(di, (int)wtype, n_in, n_out, n_hits, slot_ids, act_rows);

    std::vector<float> out((size_t)n_hits * n_out, -12345.0f);
    float * out_rows[64];
    for (int i = 0; i < n_hits; i++) out_rows[i] = out.data() + (size_t)i * n_out;
    ec3_collect(di, n_hits, out_rows, n_out);

    // host reference: dequantize slot rows, dot with fp32 act (mmvq quantizes the
    // activation to q8_1, so allow a tolerance)
    const ggml_type_traits * tr = ggml_get_type_traits(wtype);
    std::vector<float> wrow(n_in);
    double max_rel = 0.0;
    for (int i = 0; i < n_hits; i++) {
        const char * slot = wq.data() + (size_t)slot_ids[i] * slot_bytes;
        const float * a = act_rows[i];
        for (int r = 0; r < n_out; r += 37) {   // sample rows
            tr->to_float(slot + (size_t)r * row_bytes, wrow.data(), n_in);
            double ref = 0.0;
            for (int64_t c = 0; c < n_in; c++) ref += (double)wrow[c] * a[c];
            const double got = out[(size_t)i * n_out + r];
            const double rel = fabs(got - ref) / (fabs(ref) + 1e-3);
            if (rel > max_rel) max_rel = rel;
        }
    }

    // latency micro-benchmark: 200 dispatch+collect cycles
    cudaStreamSynchronize(d.compute_stream);
    const int reps = 200;
    int64_t t0 = ggml_time_us();
    for (int r = 0; r < reps; r++) {
        ec3_dispatch(di, (int)wtype, n_in, n_out, n_hits, slot_ids, act_rows);
        ec3_collect(di, n_hits, out_rows, n_out);
    }
    const double us_per_node = (double)(ggml_time_us() - t0) / reps;

    // tolerance: mmvq quantizes the activation to q8_1 while the reference uses
    // fp32, so a few percent of relative error on near-zero outputs is expected
    const bool ok = max_rel < 0.10;
    EC3_LOG("[ec3-selftest] dev=%d type=%s n_in=%lld n_out=%lld hits=%d %s: max_rel=%.4f %s | %.1f us/node\n",
            di, ggml_type_name(wtype), (long long)n_in, (long long)n_out, n_hits,
            shared_act ? "shared-act" : "multi-act", max_rel, ok ? "OK" : "FAIL", us_per_node);

    cudaFree(d_pool);
    p = ec3_pool{};
    return ok;
}

static void ec3_selftest(void) {
    bool all = true;
    all &= ec3_selftest_one(0, GGML_TYPE_Q4_K, 2048, 768,  8, true);
    all &= ec3_selftest_one(0, GGML_TYPE_Q4_K, 768,  2048, 8, false);
    all &= ec3_selftest_one(0, GGML_TYPE_Q4_K, 2048, 768,  1, true);
    all &= ec3_selftest_one(0, GGML_TYPE_Q6_K, 2048, 768,  5, true);
    all &= ec3_selftest_one(0, GGML_TYPE_Q6_K, 512,  2048, 8, false);
    EC3_LOG("[ec3-selftest] %s\n", all ? "ALL PASS" : "FAILURES PRESENT");
}

// ---- registration ----------------------------------------------------------------------

void ggml_expert_cache_v3_register(void) {
    const char * v = getenv("LLAMA_EC3");
    g.enabled = v && atoi(v) > 0;
    if (!g.enabled) return;

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count <= 0) { g.enabled = false; return; }

    g.n_dev = dev_count > EC3_MAX_DEV ? EC3_MAX_DEV : dev_count;
    if (const char * e = getenv("LLAMA_EC3_NDEV"))      { int n = atoi(e); if (n > 0 && n < g.n_dev) g.n_dev = n; }
    if (const char * e = getenv("LLAMA_EC3_BUDGET_MB")) g.budget_mb = (size_t)atoll(e);
    if (const char * e = getenv("LLAMA_EC3_INSERTS"))   g.inserts_per_plan = atoi(e);
    if (const char * e = getenv("LLAMA_EC3_WORKERS"))   { int n = atoi(e); if (n > 0 && n <= 16) g.n_workers = n; }
    if (const char * e = getenv("LLAMA_EC3_STATS"))     g.stats_every = atoi(e);

    ggml_expert_cache_v3.begin    = ec3_begin;
    ggml_expert_cache_v3.plan     = ec3_plan;
    ggml_expert_cache_v3.dispatch = ec3_dispatch;
    ggml_expert_cache_v3.collect  = ec3_collect;
    ggml_expert_cache_v3.stats    = ec3_stats;

    EC3_LOG("[ec3] enabled: n_dev=%d budget=%s inserts/plan=%d workers=%d stats_every=%d\n",
            g.n_dev, g.budget_mb ? "env" : "auto-70%-free", g.inserts_per_plan,
            g.n_workers, g.stats_every);

    if (const char * e = getenv("LLAMA_EC3_SELFTEST"); e && atoi(e) > 0) {
        ec3_selftest();
    }
}
