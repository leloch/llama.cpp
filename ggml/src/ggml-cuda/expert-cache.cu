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
    size_t    act_q8_half = 0;   // byte offset of the second staging half
    float   * d_out = nullptr;                              size_t d_out_cap = 0;
    float   * h_out = nullptr;                              size_t h_out_cap = 0; // pinned
    int       out_rows = 0;

    // gate-defer: a gate node's collect is postponed into the same layer's up
    // node (nothing reads gate's dst between the two MMIDs in build_moe_ffn),
    // halving stream syncs and overlapping gate GPU work with up CPU work.
    bool      pending_active = false;
    int       pending_blk = -1;
    int       pending_rows = 0;
    float *   pending_dst[64];
    int64_t   pending_n_out = 0;

    // activation reuse: gate and up of one layer share the same input row
    const float * q8_act_ptr = nullptr;   // host act already quantized on device
    int           q8_act_blk = -1;


    // stats
    long long hits = 0, misses = 0, inserts = 0, evictions = 0;
    long long insert_skips = 0, queued_misses = 0;
    // per-phase wall time (thread-0 serial cost), microseconds
    long long t_plan_us = 0, t_disp_us = 0, t_coll_us = 0, n_nodes = 0;
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
    size_t budget_mb = 0;        // 0 = auto (free VRAM at init minus reserve)
    size_t reserve_mb = 3072;    // VRAM left untouched per device: the CUDA pool
                                 // grows lazily AFTER our init; stealing it
                                 // crashes the model mid-decode (measured)
    int    inserts_per_plan = 8; // max inserts enqueued per plan() call
    int    queue_max        = 512;
    int    n_workers        = 4;
    size_t min_expert_bytes = 1u << 20; // skip models whose experts are too small
                                        // to amortize per-node dispatch (measured:
                                        // 0.45MB experts lose, 3MB+ win big)
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
    int          cur_blk  = -1;
    int          cur_role = -1;   // 0=gate 1=up 2=down -1=other
    bool         defer    = true; // LLAMA_EC3_DEFER=0 to disable gate-defer
    bool         reuse    = true; // LLAMA_EC3_REUSE=0 to disable act-quant reuse

    // gate-defer safety: defer only on layers where the up node was OBSERVED
    // to directly follow the gate node (learned during the first decode token,
    // so exotic graphs / partial GPU placement can never corrupt gate's dst).
    int  learn_gate_blk = -1;
    bool safe_defer_blk[1024] = {};

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
        if (g_dbg > 0 && g_dbg_n++ < (g_dbg >= 10 ? (long long)g_dbg : 400)) { EC3_LOG(__VA_ARGS__); fflush(stderr); } \
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

// Shape discovery. Pools are created per device ON DEMAND once any tensor name
// has repeated globally (i.e. the steady decode loop has begun — allocating on
// the very first sighting would mis-budget before the model placement and KV
// allocations settle). There is no "warmup complete" latch: a shape first seen
// late (odd per-layer quants, partial GPU placement, bench context churn)
// simply gets its pool late. No visit order can lock a device out.
struct ec3_discovery {
    std::unordered_set<uint64_t> seen;
    struct shape { size_t size; int wtype; };
    std::vector<shape> pending[EC3_MAX_DEV];   // shapes seen, pool not yet built
    bool any_repeat = false;
};
static ec3_discovery g_disc;

// build one pool for (size, wtype) on device di; caller ensures no duplicate
static bool ec3_pool_alloc(int di, size_t expert_size, int wtype, int n_shapes_pending) {
    ec3_device & d = g.dev[di];
    if (d.n_pools >= EC3_MAX_POOLS) return false;

    ggml_cuda_set_device(di);

    const size_t reserve = g.reserve_mb << 20;
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    size_t avail = free_mem > reserve ? free_mem - reserve : 0;
    if (g.budget_mb > 0 && (g.budget_mb << 20) < avail) {
        avail = g.budget_mb << 20;
    }
    // leave room for the device's other not-yet-built pools. MoE models have
    // (at least) two expert shapes — gate/up and down — and discovery order is
    // not guaranteed, so never let a single pool claim more than half.
    const int pool_div = n_shapes_pending > 2 ? n_shapes_pending : 2;
    const size_t budget = avail / pool_div;

    int ns = (int)(budget / expert_size);
    if (ns < 64) {
        EC3_LOG("[ec3] dev=%d pool for %zu KB slots skipped (budget %zu MB too small)\n",
                di, expert_size >> 10, budget >> 20);
        return false;
    }

    char * slab = nullptr;
    cudaError_t err = cudaMalloc((void **)&slab, (size_t)ns * expert_size);
    if (err != cudaSuccess) {
        cudaGetLastError();
        EC3_LOG("[ec3] dev=%d pool alloc failed: %s\n", di, cudaGetErrorString(err));
        return false;
    }

    ec3_pool & p = d.pools[d.n_pools];
    p.expert_size = expert_size;
    p.wtype       = wtype;
    p.slab        = slab;
    p.n_slots     = ns;
    p.n_used      = 0;
    p.map.clear();
    p.lru_head = p.lru_tail = -1;
    p.slots.assign(ns, ec3_slot{0, -1, -1, false, false});
    d.n_pools++;
    EC3_LOG("[ec3] dev=%d pool[%d]: type=%d slot=%zu KB slots=%d total=%zu MB\n",
            di, d.n_pools - 1, wtype, expert_size >> 10, ns, ((size_t)ns * expert_size) >> 20);

    if (!d.compute_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&d.compute_stream, cudaStreamNonBlocking));
    }
    ec3_start_workers();
    return true;
}

// ---- API: begin -----------------------------------------------------------------

static int ec3_begin(const char * name, const void * host_base, size_t expert_size,
                     int64_t n_in, int64_t n_out, int wtype, int64_t n_expert, int64_t n_tokens) {
    GGML_UNUSED(n_in); GGML_UNUSED(n_out);

    if (!g.enabled || n_tokens != 1) return -1;
    if (expert_size < g.min_expert_bytes) return -1;

    const char * p = strstr(name, "blk.");
    if (!p || !strstr(name, "_exps")) return -1;
    const int blk = atoi(p + 4);
    const int di  = blk % g.n_dev;

    const uint64_t kb = ec3_fnv1a(name);
    ec3_device & d = g.dev[di];

    // shape discovery + on-demand pool construction (see ec3_discovery)
    int pi = -1;
    for (int i = 0; i < d.n_pools; i++) {
        if (d.pools[i].expert_size == expert_size && d.pools[i].wtype == wtype) { pi = i; break; }
    }
    if (pi < 0) {
        bool pending = false;
        for (auto & sh : g_disc.pending[di]) {
            if (sh.size == expert_size && sh.wtype == wtype) { pending = true; break; }
        }
        if (!pending) {
            g_disc.pending[di].push_back({expert_size, wtype});
            EC3_DBG("[ec3-dbg] new shape %s blk=%d dev=%d size=%zu type=%d\n",
                    name, blk, di, expert_size, wtype);
        }
        if (!g_disc.any_repeat) {
            if (g_disc.seen.count(kb)) {
                g_disc.any_repeat = true;
                EC3_LOG("[ec3] decode loop detected (trigger=%s) — building pools on demand\n", name);
            } else {
                g_disc.seen.insert(kb);
                return -1;
            }
        }
        // steady state reached: build this device's pending pools now
        auto & pend = g_disc.pending[di];
        for (size_t i = 0; i < pend.size(); i++) {
            ec3_pool_alloc(di, pend[i].size, pend[i].wtype, (int)(pend.size() - i));
        }
        pend.clear();
        for (int i = 0; i < d.n_pools; i++) {
            if (d.pools[i].expert_size == expert_size && d.pools[i].wtype == wtype) { pi = i; break; }
        }
        if (pi < 0) return -1;
    }
    if (!g_disc.any_repeat) {
        if (g_disc.seen.count(kb)) {
            g_disc.any_repeat = true;
        } else {
            g_disc.seen.insert(kb);
            return -1;
        }
    }

    int role = -1;
    if      (strstr(name, "_gate_exps")) role = 0;
    else if (strstr(name, "_up_exps"))   role = 1;
    else if (strstr(name, "_down_exps")) role = 2;

    // gate-defer safety learning: mark a layer safe when its up node is the
    // very next cache visit after its gate node
    if (role == 1 && blk == g.learn_gate_blk && blk >= 0 && blk < 1024) {
        if (!g.safe_defer_blk[blk]) EC3_DBG("[ec3-dbg] defer-safe blk=%d\n", blk);
        g.safe_defer_blk[blk] = true;
    }
    g.learn_gate_blk = (role == 0) ? blk : -1;

    g.cur_blk  = blk;
    g.cur_role = role;

    EC3_DBG("[ec3-dbg] begin %s dev=%d pool=%d role=%d\n", name, di, pi, role);

    g.cur_key_base    = kb;
    g.cur_host_base   = host_base;
    g.cur_expert_size = expert_size;
    g.cur_n_expert    = n_expert;
    g.cur_pool        = pi;
    return di;
}

// flush a deferred gate collect: sync the stream, scatter the pending rows
static void ec3_flush(int di) {
    ec3_device & d = g.dev[di];
    if (!d.pending_active) return;
    ggml_cuda_set_device(di);
    const size_t bytes = (size_t)d.out_rows * d.pending_n_out * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(d.h_out, d.d_out, bytes, cudaMemcpyDeviceToHost, d.compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(d.compute_stream));
    for (int i = 0; i < d.pending_rows; i++) {
        memcpy(d.pending_dst[i], d.h_out + (size_t)i * d.pending_n_out, d.pending_n_out * sizeof(float));
    }
    d.pending_active = false;
    d.pending_rows   = 0;
    d.out_rows       = 0;
    d.q8_act_ptr     = nullptr;
}

// ---- API: plan --------------------------------------------------------------------

static int ec3_plan(int di, const int32_t * ids, int n_ids, int32_t * slot_idx) {
    ec3_device & d = g.dev[di];
    ec3_pool   & p = d.pools[g.cur_pool];

    const int64_t t0 = ggml_time_us();
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

    // resolve any deferred gate collect that this node will not absorb
    // (absorbed only by the same layer's up node when it has hits of its own)
    if (d.pending_active && !(g.cur_role == 1 && g.cur_blk == d.pending_blk && n_hits > 0)) {
        ec3_flush(di);
    }

    d.t_plan_us += ggml_time_us() - t0;
    d.n_nodes++;
    EC3_DBG("[ec3-dbg] plan dev=%d hits=%d q=%zu\n", di, n_hits, g.queue.size());
    return n_hits;
}

// ---- API: dispatch ------------------------------------------------------------------

static void ec3_dispatch(int di, int wtype_int, int64_t n_in, int64_t n_out, int n_hits,
                         const int32_t * slot_idx_compact, const float * const * act_rows) {
    if (n_hits <= 0) return;
    const int64_t t0 = ggml_time_us();
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

    // grow staging. Host staging (h_ids/h_act) is allocated 2x and used in
    // halves: with gate-defer, a second dispatch is enqueued while the first
    // one's async H2D copies may not have executed yet — the DMA engine reads
    // pinned host memory at stream-execution time, so the staging being
    // written now must not be the staging still in flight. Device-side
    // buffers are stream-ordered and safe to reuse.
    if (d.ids_cap < (size_t)n_hits) {
        const size_t cap = n_hits * 2 + 8;
        if (d.h_ids) cudaFreeHost(d.h_ids);
        if (d.d_ids) cudaFree(d.d_ids);
        CUDA_CHECK(cudaMallocHost((void **)&d.h_ids, 2 * cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc((void **)&d.d_ids, 2 * cap * sizeof(int32_t)));
        d.ids_cap = cap;
    }
    const size_t need_act = (size_t)act_n * n_in * sizeof(float);
    if (d.act_cap < need_act) {
        const size_t cap = need_act * 2;
        if (d.h_act) cudaFreeHost(d.h_act);
        if (d.d_act) cudaFree(d.d_act);
        CUDA_CHECK(cudaMallocHost((void **)&d.h_act, 2 * cap));
        CUDA_CHECK(cudaMalloc((void **)&d.d_act, 2 * cap));
        d.act_cap = cap;
    }
    const int       half    = d.pending_active ? 1 : 0;
    int32_t * const h_ids_h = d.h_ids + (size_t)half * d.ids_cap;
    int32_t * const d_ids_h = d.d_ids + (size_t)half * d.ids_cap;
    float   * const h_act_h = (float *)((char *)d.h_act + (size_t)half * d.act_cap);
    float   * const d_act_h = (float *)((char *)d.d_act + (size_t)half * d.act_cap);
    const size_t need_q8 = (size_t)act_n * (n_in_padded / QK8_1) * sizeof(block_q8_1);
    if (d.act_q8_cap < need_q8) {
        const size_t cap = need_q8 * 2;
        if (d.d_act_q8) cudaFree(d.d_act_q8);
        CUDA_CHECK(cudaMalloc(&d.d_act_q8, 2 * cap));
        d.act_q8_cap  = cap;
        d.act_q8_half = cap;
    }
    const size_t need_out = (size_t)(d.out_rows + n_hits) * n_out * sizeof(float);
    if (d.d_out_cap < need_out) {
        // reallocating d_out would drop deferred rows still resident there
        if (d.pending_active) ec3_flush(di);
        const size_t cap = ((size_t)(d.out_rows + n_hits) * n_out * sizeof(float)) * 2 + 65536;
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

    // fill pinned staging on the host (the async copies read it at execution)
    for (int i = 0; i < n_hits; i++) h_ids_h[i] = slot_idx_compact[i];

    // gate and up of one layer read the same activation row: reuse the
    // quantized copy already on the device when possible
    const bool reuse_q8 = g.reuse && act_n == 1 && g.cur_role == 1 &&
                          d.q8_act_ptr == act_rows[0] && d.q8_act_blk == g.cur_blk;
    const char * act_q8 = (const char *)d.d_act_q8 + (reuse_q8 ? 0 : (size_t)half * d.act_q8_half);
    if (!reuse_q8) {
        for (int i = 0; i < act_n; i++) {
            memcpy(h_act_h + (size_t)i * n_in, act_rows[i], n_in * sizeof(float));
        }
        d.q8_act_ptr = (act_n == 1 && half == 0) ? act_rows[0] : nullptr;
        d.q8_act_blk = g.cur_blk;
    }

    // the GPU chain: ids H2D [+ act H2D + quantize] + batched mmv. All buffer
    // addresses and sizes are fixed for a given shape key, so the chain can be
    // captured once into a CUDA graph and replayed as a single launch — the
    // chain's per-op launch latency is the dominant per-node cost.
    auto emit_chain = [&](cudaStream_t s) {
        CUDA_CHECK(cudaMemcpyAsync(d_ids_h, h_ids_h, n_hits * sizeof(int32_t), cudaMemcpyHostToDevice, s));
        if (!reuse_q8) {
            CUDA_CHECK(cudaMemcpyAsync(d_act_h, h_act_h, need_act, cudaMemcpyHostToDevice, s));
            quantize_row_q8_1_cuda(d_act_h, /*ids=*/nullptr, (void *)act_q8, wtype,
                                   n_in, /*s01=*/n_in, /*s02=*/(int64_t)act_n * n_in, /*s03=*/(int64_t)act_n * n_in,
                                   n_in_padded, /*ne1=*/act_n, /*ne2=*/1, /*ne3=*/1, s);
        }
        ggml_cuda_ec3_mmv(p.slab, wtype, act_q8, d_ids_h, d.d_out + (size_t)d.out_rows * n_out,
                          n_in, n_out, p.n_slots, (int64_t)p.expert_size,
                          n_hits, /*act_rows=*/act_n, s);
    };

    // note: CUDA-graph capture of this chain was tried and measured to be a
    // net loss — the chain is GPU-exec-bound, not launch-bound, and pools can
    // hold mixed (n_in, n_out) shapes which makes graph keying hazardous.
    emit_chain(st);

    d.out_rows += n_hits;
    d.t_disp_us += ggml_time_us() - t0;
    EC3_DBG("[ec3-dbg] dispatch dev=%d hits=%d act_n=%d n_in=%lld n_out=%lld\n",
            di, n_hits, act_n, (long long)n_in, (long long)n_out);
}

// ---- API: collect --------------------------------------------------------------------

static void ec3_stats(void);

static void ec3_collect(int di, int n_hits, float * const * dst_rows, int64_t n_out) {
    const int64_t t0 = ggml_time_us();
    ec3_device & d = g.dev[di];
    const int new_rows = d.out_rows - d.pending_rows;
    if (n_hits != new_rows) {
        EC3_LOG("[ec3] BUG: collect rows %d != dispatched %d\n", n_hits, new_rows);
    }

    // gate-defer: postpone this sync into the same layer's up node (only on
    // layers where up was observed to directly follow gate — see begin())
    if (g.defer && g.cur_role == 0 && !d.pending_active && n_hits <= 64 &&
        g.cur_blk >= 0 && g.cur_blk < 1024 && g.safe_defer_blk[g.cur_blk]) {
        d.pending_active = true;
        d.pending_blk    = g.cur_blk;
        d.pending_rows   = n_hits;
        d.pending_n_out  = n_out;
        memcpy(d.pending_dst, dst_rows, n_hits * sizeof(float *));
        if (g_dbg > 0) {
            // poison: detect any reader/writer touching the rows mid-defer
            for (int i = 0; i < n_hits; i++) dst_rows[i][0] = 1e30f;
        }
        // bisection aid: LLAMA_EC3_DEFER_SYNC=1 keeps the bookkeeping but syncs
        // here anyway — separates bookkeeping bugs from async-interaction bugs
        static const bool defer_sync = []{ const char * e = getenv("LLAMA_EC3_DEFER_SYNC"); return e && atoi(e) > 0; }();
        if (defer_sync) {
            ggml_cuda_set_device(di);
            CUDA_CHECK(cudaStreamSynchronize(d.compute_stream));
        }
        // bisection aid: LLAMA_EC3_DEFER_IMM=1 resolves the stash immediately via
        // the flush path — identical timing to non-defer, but through defer code
        static const bool defer_imm = []{ const char * e = getenv("LLAMA_EC3_DEFER_IMM"); return e && atoi(e) > 0; }();
        if (defer_imm) {
            ec3_flush(di);
        }
        d.t_coll_us += ggml_time_us() - t0;
        EC3_DBG("[ec3-dbg] defer-stash dev=%d blk=%d rows=%d\n", di, g.cur_blk, n_hits);
        return;
    }

    ggml_cuda_set_device(di);
    EC3_DBG("[ec3-dbg] collect dev=%d rows=%d pre-sync\n", di, d.out_rows);
    const size_t bytes = (size_t)d.out_rows * n_out * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(d.h_out, d.d_out, bytes, cudaMemcpyDeviceToHost, d.compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(d.compute_stream));
    EC3_DBG("[ec3-dbg] collect dev=%d post-sync\n", di);
    if (g_dbg > 0 && d.pending_rows > 0) {
        static int violations = 0, checks = 0;
        for (int i = 0; i < d.pending_rows; i++) {
            checks++;
            if (d.pending_dst[i][0] != 1e30f && violations++ < 8) {
                EC3_LOG("[ec3-dbg] DEFER VIOLATION blk=%d row=%d poison gone (%.3g) after %d ok\n",
                        d.pending_blk, i, d.pending_dst[i][0], checks);
            }
        }
    }
    for (int i = 0; i < d.pending_rows; i++) {
        memcpy(d.pending_dst[i], d.h_out + (size_t)i * n_out, n_out * sizeof(float));
    }
    for (int i = 0; i < n_hits; i++) {
        memcpy(dst_rows[i], d.h_out + (size_t)(d.pending_rows + i) * n_out, n_out * sizeof(float));
    }
    d.out_rows       = 0;
    d.pending_active = false;
    d.pending_rows   = 0;
    d.q8_act_ptr     = nullptr;
    d.t_coll_us += ggml_time_us() - t0;

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
        if (d.n_nodes > 0) {
            EC3_LOG("[ec3] dev=%d timing: nodes=%lld plan=%.1fus disp=%.1fus coll=%.1fus per-node total=%.1fus\n",
                    i, d.n_nodes,
                    (double)d.t_plan_us / d.n_nodes, (double)d.t_disp_us / d.n_nodes,
                    (double)d.t_coll_us / d.n_nodes,
                    (double)(d.t_plan_us + d.t_disp_us + d.t_coll_us) / d.n_nodes);
        }
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
    if (const char * e = getenv("LLAMA_EC3_MIN_EXPERT_KB")) g.min_expert_bytes = (size_t)atoll(e) << 10;
    if (const char * e = getenv("LLAMA_EC3_RESERVE_MB"))    g.reserve_mb = (size_t)atoll(e);
    if (const char * e = getenv("LLAMA_EC3_DEFER"))         g.defer = atoi(e) > 0;
    if (const char * e = getenv("LLAMA_EC3_REUSE"))         g.reuse = atoi(e) > 0;

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
