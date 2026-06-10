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
#include "ggml-backend-impl.h"
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
    char * slab        = nullptr;   // up weights (mmv x operand)
    char * slab2       = nullptr;   // gate weights (fusion operand), paired pools only
    bool   paired      = false;     // one entry covers the (gate, up) pair of an expert
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

    // GPU-resident dst handoff machinery. The pinned image and its event are
    // double-buffered by parity: node N+1's D2H must not overwrite the image
    // node N's consumer-stream H2D still reads.
    char    * h_redir = nullptr; size_t h_redir_half = 0;    // pinned image x2
    cudaEvent_t redir_evt[2] = {};
    int       redir_par = 0;

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

    // fused gate+up+GLU: one layer in flight per device. Filled at the gate
    // node, matched+scattered by the CPU GLU kernel via glu_hits().
    struct {
        bool         active = false;
        const void * gate_dst = nullptr;  // gate MMID dst base (== glu src0)
        const void * up_dst   = nullptr;  // up MMID dst base   (== glu src1)
        unsigned long long mask = 0;      // dst row bits computed on GPU
        int          rows[64];            // dst row index per d_out row
        int          n = 0;
        int64_t      n_out = 0;
        bool         scattered = false;
        long long    serial = 0;          // gallocr reuses dst pointers every
                                          // layer: the GLU hook must match the
                                          // NEWEST entry, not any stale one
    } fused;
    long long fused_layers = 0;


    // stats
    long long hits = 0, misses = 0, inserts = 0, evictions = 0;
    long long insert_skips = 0, queued_misses = 0;
    // miss decomposition (counters only, no behavior change)
    long long pool_hits[EC3_MAX_POOLS] = {}, pool_miss[EC3_MAX_POOLS] = {};
    long long miss_compulsory = 0, miss_capacity = 0, miss_admission = 0;
    long long skip_throttle = 0, skip_budget = 0, skip_qfull = 0, skip_lrubusy = 0;
    std::unordered_set<uint64_t> ever_seen, ever_inserted;
    // per-phase wall time (thread-0 serial cost), microseconds
    long long t_plan_us = 0, t_disp_us = 0, t_coll_us = 0, n_nodes = 0;
    long long t_coll_role_us[3] = {}, n_coll_role[3] = {};   // gate/up/down split
    long long redirect_claims = 0, redirect_misses_up = 0;
};

struct ec3_job {
    int        dev;
    int        pool;
    uint64_t   key;
    int        slot_idx;
    const void * src;        // up weights (paired) or sole tensor
    const void * src_gate;   // gate weights (paired pools), else NULL
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
    int    throttle_mod     = 8; // at capacity admit 1-in-N misses (LLAMA_EC3_THROTTLE)
    bool   greedy_last      = false; // last pending pool takes all remaining avail
    int    queue_max        = 512;
    int    n_workers        = 4;
    size_t min_expert_bytes = 1u << 20; // skip models whose experts are too small
                                        // to amortize per-node dispatch (measured:
                                        // 0.45MB experts lose, 3MB+ win big)
    int    max_batch        = 1; // decode batches up to this size use the cache
                                 // (LLAMA_EC3_MAX_BATCH; >1 for spec-verify/parallel)
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
    int64_t      cur_n_tokens = 1;
    int32_t      cur_slot_idx[64] = {};
    int          cur_n_ids = 0;
    std::unordered_map<const void *, int> glu_learn;  // gate MMID dst base -> blk
    bool         defer    = false; // LLAMA_EC3_DEFER=1 to enable gate-defer.
                                   // OFF by default: measured perf-neutral, and the
                                   // adjacency-learned safety proof does not cover
                                   // models with readers between the MMIDs
                                   // (gpt-oss bias add_id, per-expert scales) —
                                   // see EC3_READINESS.md B2.
    bool         reuse    = true; // LLAMA_EC3_REUSE=0 to disable act-quant reuse
    bool         stripe   = false; // LLAMA_EC3_STRIPE=1: role-stripe devices (forces defer off)

    // gate-defer safety: defer only on layers where the up node was OBSERVED
    // to directly follow the gate node (learned during the first decode token,
    // so exotic graphs / partial GPU placement can never corrupt gate's dst).
    int  learn_gate_blk = -1;
    bool safe_defer_blk[1024] = {};

    // fused gate+up+GLU path. Pair-fused dispatch engages per layer only after
    // the CPU GLU hook was OBSERVED matching that layer's gate/up dst pair
    // (learned on the first decode tokens) — a graph without the hook firing
    // would otherwise compute silu(garbage)*garbage for the skipped rows.
    bool fuse = true;                    // LLAMA_EC3_FUSE=0 to disable. Stale-entry
                                         // hazard (EC3_READINESS.md B1) closed by
                                         // gate-begin epoch invalidation + up-node
                                         // mask reuse.
    long long fuse_serial = 0;
    bool safe_fuse_blk[1024] = {};
    const void * role_base[2][1024] = {}; // [role][blk] -> tensor host base
    // last gate/up MMID dst bases per blk (for GLU-hook learning)
    const void * learn_gate_dst[1024] = {};
    const void * learn_up_dst[1024] = {};

    // GPU-resident dst handoff: host dst base -> offered GPU copy. Entries are
    // one-shot: offered before the CPU split, optionally populated by collect,
    // resolved (claimed or dropped) at the consumer's input-copy site.
    struct redirect_entry {
        size_t  nb1 = 0;
        int64_t n_rows = 0;
        void *  gpu_ptr = nullptr;
        int     dev = -1;
        uint64_t hit_mask = 0;     // rows relayed by collect
        bool    populated = false; // collect engaged
        int     par = 0;           // pinned-image parity used by collect
    };
    std::unordered_map<const void *, redirect_entry> redirect;
    bool redirect_on = true;       // LLAMA_EC3_REDIRECT=0 to disable

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
        char * dst_gate = job.src_gate ? p.slab2 + (size_t)job.slot_idx * p.expert_size : nullptr;

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
            if (err == cudaSuccess && dst_gate) {
                memcpy(stage, job.src_gate, job.bytes);
                err = cudaMemcpyAsync(dst_gate, stage, job.bytes, cudaMemcpyHostToDevice, cstream[job.dev]);
                if (err == cudaSuccess) {
                    err = cudaStreamSynchronize(cstream[job.dev]);
                }
            }
        } else if (err == cudaSuccess) {
            // pinned alloc failed: fall back to a direct pageable copy
            err = cudaMemcpy(dst, job.src, job.bytes, cudaMemcpyHostToDevice);
            if (err == cudaSuccess && dst_gate) {
                err = cudaMemcpy(dst_gate, job.src_gate, job.bytes, cudaMemcpyHostToDevice);
            }
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
    struct shape { size_t size; int wtype; int n_tensors; int roles; int64_t n_expert; };
    std::vector<shape> pending[EC3_MAX_DEV];   // shapes seen, pool not yet built
    bool any_repeat = false;
    // stable-census guard: pools are built only after the shape census has not
    // changed for a window of eligible visits AND a tensor has repeated. A
    // partially-discovered census mis-sizes pools permanently (measured: the
    // 754B server lost its down pools to visit-order luck).
    int  stable_count = 0;
};
static ec3_discovery g_disc;

// build one pool for (size, wtype) on device di; caller ensures no duplicate
static bool ec3_pool_alloc(int di, size_t expert_size, int wtype, size_t budget, bool paired,
                           int64_t max_entries) {
    ec3_device & d = g.dev[di];
    if (d.n_pools >= EC3_MAX_POOLS) return false;

    ggml_cuda_set_device(di);

    // a paired slot stores the (gate, up) tensors of one expert in two
    // parallel slabs: same byte budget, half the slot count, same number of
    // cached EXPERTS per byte as two independent entries — but joint by
    // construction, which the fused kernel requires
    int ns = (int)(budget / (paired ? 2 * expert_size : expert_size));
    // cap slots at the number of distinct cacheable entries (safe now: pools
    // are only built after the stable-census window, so n_tensors is real)
    if (max_entries > 0 && ns > max_entries) ns = (int)max_entries;
    if (ns < 64) {
        static int warned = 0;
        if (warned++ < 2) {
            EC3_LOG("[ec3] dev=%d pool for %zu KB slots skipped (budget %zu MB too small) — cache stays off for this shape\n",
                    di, expert_size >> 10, budget >> 20);
        }
        // dead marker: prevents endless re-discovery + re-trigger + log spam
        ec3_pool & p = d.pools[d.n_pools];
        p.expert_size = expert_size;
        p.wtype       = wtype;
        p.slab        = nullptr;
        p.n_slots     = 0;
        d.n_pools++;
        return false;
    }

    char * slab = nullptr;
    cudaError_t err = cudaMalloc((void **)&slab, (size_t)ns * expert_size);
    if (err != cudaSuccess) {
        cudaGetLastError();
        EC3_LOG("[ec3] dev=%d pool alloc failed: %s\n", di, cudaGetErrorString(err));
        return false;
    }
    char * slab2 = nullptr;
    if (paired) {
        err = cudaMalloc((void **)&slab2, (size_t)ns * expert_size);
        if (err != cudaSuccess) {
            cudaGetLastError();
            cudaFree(slab);
            EC3_LOG("[ec3] dev=%d paired pool alloc failed: %s\n", di, cudaGetErrorString(err));
            return false;
        }
    }

    ec3_pool & p = d.pools[d.n_pools];
    p.expert_size = expert_size;
    p.wtype       = wtype;
    p.slab        = slab;
    p.slab2       = slab2;
    p.paired      = paired;
    p.n_slots     = ns;
    p.n_used      = 0;
    p.map.clear();
    p.lru_head = p.lru_tail = -1;
    p.slots.assign(ns, ec3_slot{0, -1, -1, false, false});
    d.n_pools++;
    EC3_LOG("[ec3] dev=%d pool[%d]: type=%d slot=%zu KB slots=%d total=%zu MB%s\n",
            di, d.n_pools - 1, wtype, expert_size >> 10, ns,
            ((size_t)(paired ? 2 : 1) * ns * expert_size) >> 20, paired ? " (paired)" : "");

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

    if (!g.enabled || n_tokens < 1 || n_tokens > g.max_batch) return -1;
    if (expert_size < g.min_expert_bytes) return -1;

    // only types with a kernel case in mul_mat_vec_q_switch_type — anything else
    // would GGML_ABORT on the first cached row (e.g. F16/BF16/TQ expert tensors)
    switch ((ggml_type)wtype) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0: case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:  case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:   case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
            break;
        default:
            return -1;
    }

    const char * p = strstr(name, "blk.");
    if (!p || !strstr(name, "_exps")) return -1;
    const int blk = atoi(p + 4);

    int role = -1;
    if      (strstr(name, "_gate_exps")) role = 0;
    else if (strstr(name, "_up_exps"))   role = 1;
    else if (strstr(name, "_down_exps")) role = 2;

    // LLAMA_EC3_STRIPE=1: spread the three roles of a layer across devices so
    // gate/up traffic does not serialize on one device's stream (probe)
    const int di = g.stripe ? (blk * 3 + (role < 0 ? 0 : role)) % g.n_dev
                            : blk % g.n_dev;

    const uint64_t kb = ec3_fnv1a(name);
    ec3_device & d = g.dev[di];
    const bool first_sight = g_disc.seen.count(kb) == 0;

    // shape discovery + on-demand pool construction (see ec3_discovery)
    int pi = -1;
    for (int i = 0; i < d.n_pools; i++) {
        if (d.pools[i].expert_size == expert_size && d.pools[i].wtype == wtype) { pi = i; break; }
    }
    if (pi < 0) {
        ec3_discovery::shape * shp = nullptr;
        for (auto & sh : g_disc.pending[di]) {
            if (sh.size == expert_size && sh.wtype == wtype) { shp = &sh; break; }
        }
        if (!shp) {
            g_disc.pending[di].push_back({expert_size, wtype, 0, 0, n_expert});
            shp = &g_disc.pending[di].back();
            g_disc.stable_count = 0;   // census changed: restart the stability window
            EC3_DBG("[ec3-dbg] new shape %s blk=%d dev=%d size=%zu type=%d\n",
                    name, blk, di, expert_size, wtype);
        } else {
            g_disc.stable_count++;
        }
        if (first_sight) shp->n_tensors++;   // distinct tensors using this shape
        if (role >= 0) shp->roles |= 1 << role;
        if (!g_disc.any_repeat) {
            if (g_disc.seen.count(kb)) {
                g_disc.any_repeat = true;
            } else {
                g_disc.seen.insert(kb);
                return -1;
            }
        }
        // stable-census window: 64 eligible visits without a new shape
        if (g_disc.stable_count < 64) {
            return -1;
        }
        static bool announced = false;
        if (!announced) {
            announced = true;
            EC3_LOG("[ec3] decode loop detected, shape census stable — building pools\n");
        }
        // steady state reached: build this device's pending pools in ONE
        // proportional pass. Budgets are weighted by referenced bytes per
        // token (shape size x number of tensors using the shape — gate+up
        // share a shape, so theirs weighs ~2x per layer vs down's 1x); the
        // previous sequential-halving scheme left ~25% of the budget
        // unallocated and starved the down pool (measured).
        auto & pend = g_disc.pending[di];
        if (!pend.empty()) {
            const size_t reserve = g.reserve_mb << 20;
            size_t free_mem = 0, total_mem = 0;
            ggml_cuda_set_device(di);
            CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            size_t avail = free_mem > reserve ? free_mem - reserve : 0;
            if (g.budget_mb > 0 && (g.budget_mb << 20) < avail) {
                avail = g.budget_mb << 20;
            }
            double total_w = 0.0;
            for (auto & sh : pend) {
                total_w += (double)sh.size * (sh.n_tensors > 0 ? sh.n_tensors : 1);
            }
            for (auto & sh : pend) {
                const double w = (double)sh.size * (sh.n_tensors > 0 ? sh.n_tensors : 1);
                const bool paired = g.fuse && (sh.roles & 0b11) == 0b11;
                // census may be incomplete on this device (visit-order dependent):
                // cap any single pool so a later-discovered shape always fits
                size_t budget = (size_t)(avail * (w / total_w));
                const size_t cap = (size_t)(avail * 0.60);
                if (budget > cap) budget = cap;
                const int64_t max_entries = (paired ? sh.n_tensors / 2 : sh.n_tensors) * sh.n_expert;
                ec3_pool_alloc(di, sh.size, sh.wtype, budget, paired, max_entries);
            }
        }
        pend.clear();
        for (int i = 0; i < d.n_pools; i++) {
            if (d.pools[i].expert_size == expert_size && d.pools[i].wtype == wtype) { pi = i; break; }
        }
        if (pi < 0) return -1;
    }
    if (d.pools[pi].slab == nullptr) return -1;   // dead marker (alloc failed)
    if (!g_disc.any_repeat) {
        if (g_disc.seen.count(kb)) {
            g_disc.any_repeat = true;
        } else {
            g_disc.seen.insert(kb);
            return -1;
        }
    }

    // fused-state epoch: every gate node on a device invalidates any leftover
    // fused entry (zero-hit gate nodes never reach collect, and gallocr reuses
    // dst pointers across layers — a stale entry must never survive into the
    // next layer's GLU; see EC3_READINESS.md B1)
    if (role == 0) {
        g.dev[di].fused.active = false;
    }

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

    // paired pools share ONE entry per (blk, expert): key by blk, not by name
    if (d.pools[pi].paired && (role == 0 || role == 1)) {
        g.cur_key_base = 0xEC3000000000000ULL ^ ((uint64_t)blk << 32);
        if (blk >= 0 && blk < 1024) g.role_base[role][blk] = host_base;
    } else {
        g.cur_key_base = kb;
    }
    g.cur_host_base   = host_base;
    g.cur_expert_size = expert_size;
    g.cur_n_expert    = n_expert;
    g.cur_n_tokens    = n_tokens;
    g.cur_pool        = pi;
    return di;
}

// scatter kernel for the GPU-resident dst handoff: copy row r of src
// (contiguous n_out floats per row) into dst at row_idx[r]*nb1 bytes
static __global__ void ec3_scatter_rows(const float * __restrict__ src,
                                        char * __restrict__ dst,
                                        const int32_t * __restrict__ row_idx,
                                        int64_t n_out, size_t nb1, int n_rows) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n_rows * n_out) return;
    const int     r = (int)(i / n_out);
    const int64_t c = i % n_out;
    ((float *)(dst + (size_t)row_idx[r] * nb1))[c] = src[(size_t)r * n_out + c];
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

    // fused up node: reuse the gate node's recorded hit mask verbatim. The
    // rows are computed already (fused dispatch at the gate node); a fresh
    // lookup could diverge (eviction between the plans) and skip a row that
    // nobody computed.
    if (g.cur_n_tokens == 1 && p.paired && g.cur_role == 1 &&
        g.cur_blk >= 0 && g.cur_blk < 1024 && g.safe_fuse_blk[g.cur_blk] &&
        d.fused.active && d.fused.gate_dst != nullptr) {
        int nh = 0;
        for (int k = 0; k < n_ids && k < 64; k++) {
            const bool hit = (d.fused.mask >> k) & 1ull;
            slot_idx[k] = hit ? 0 : -1;   // value unused (no dispatch); sign is the skip signal
            if (hit) nh++;
        }
        g.cur_n_ids = n_ids;
        for (int k = 0; k < n_ids && k < 64; k++) g.cur_slot_idx[k] = slot_idx[k];
        d.t_plan_us += ggml_time_us() - t0;
        d.n_nodes++;
        return nh;
    }

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
                d.pool_hits[g.cur_pool]++;
                n_hits++;
            } else {
                // insert still queued/in-flight: CPU computes the row this time
                d.queued_misses++;
                d.misses++;
                d.pool_miss[g.cur_pool]++;
            }
            continue;
        }

        d.misses++;
        d.pool_miss[g.cur_pool]++;
        if (d.ever_seen.insert(key).second) {
            d.miss_compulsory++;
        } else if (d.ever_inserted.count(key)) {
            d.miss_capacity++;   // was in cache once, evicted, needed again
        } else {
            d.miss_admission++;  // seen before but never admitted
        }

        // ---- enqueue async insert (budgeted) ----
        if (inserts_left <= 0) {
            d.insert_skips++;
            d.skip_budget++;
            continue;
        }
        if ((int)g.queue.size() >= g.queue_max) {
            d.insert_skips++;
            d.skip_qfull++;
            continue;
        }
        // admission throttle at capacity: when the pool is full, churn (evict +
        // re-copy on every miss) steals host RAM bandwidth from the CPU matmuls.
        // Admit only a fraction of misses so the content still adapts but the
        // copy traffic stays bounded.
        if (p.n_used >= p.n_slots && (d.misses % g.throttle_mod) != 0) {
            d.insert_skips++;
            d.skip_throttle++;
            continue;
        }
        // paired-entry inserts (gate/up roles only — pools can be SHARED with
        // other roles whose tensors merely have the same shape; those use the
        // plain name-keyed path below and never collide in key space)
        const bool pair_entry = p.paired && (g.cur_role == 0 || g.cur_role == 1);
        if (pair_entry && (g.cur_blk < 0 || g.cur_blk >= 1024 ||
                           !g.role_base[0][g.cur_blk] || !g.role_base[1][g.cur_blk])) {
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
            if (cand < 0 || p.slots[cand].queued) { d.insert_skips++; d.skip_lrubusy++; continue; }
            si = cand;
            ec3_slot & old = p.slots[si];
            if (old.valid || old.queued) {
                p.map.erase(old.key);
                d.evictions++;
            }
            ec3_lru_remove(p, si);
        }

        const void * src_up   = nullptr;
        const void * src_gate = nullptr;
        if (pair_entry) {
            src_up   = (const char *)g.role_base[1][g.cur_blk] + (size_t)eid * g.cur_expert_size;
            src_gate = (const char *)g.role_base[0][g.cur_blk] + (size_t)eid * g.cur_expert_size;
        } else {
            src_up = (const char *)g.cur_host_base + (size_t)eid * g.cur_expert_size;
        }

        p.slots[si] = ec3_slot{key, -1, -1, false, true};
        ec3_lru_push_back(p, si);
        p.map[key] = si;
        d.inserts++;
        d.ever_inserted.insert(key);
        inserts_left--;

        g.queue.push_back(ec3_job{di, g.cur_pool, key, si, src_up, src_gate, g.cur_expert_size});
        g.cv.notify_one();
    }

    // stash the per-position result so collect can reconstruct dst row indices
    g.cur_n_ids = n_ids;
    for (int k = 0; k < n_ids && k < 64; k++) g.cur_slot_idx[k] = slot_idx[k];

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

    const bool blk_ok     = g.cur_blk >= 0 && g.cur_blk < 1024;
    const bool fuse_layer = g.cur_n_tokens == 1 && p.paired && blk_ok && g.safe_fuse_blk[g.cur_blk];
    if (fuse_layer && g.cur_role == 1) {
        // fused rows were computed at the gate node; nothing to launch here
        d.t_disp_us += ggml_time_us() - t0;
        return;
    }
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
        const void * mmv_x   = p.slab;
        const void * mmv_gate = nullptr;
        int          mmv_glu  = -1;
        if (p.paired) {
            if (fuse_layer && g.cur_role == 0) {
                // fused: x = up slab (result operand), gate slab silu'd on-chip
                mmv_gate = p.slab2;
                mmv_glu  = (int)GGML_GLU_OP_SWIGLU;
            } else if (g.cur_role == 0) {
                mmv_x = p.slab2;   // separate gate matvec reads the gate slab
            }
        }
        ggml_cuda_ec3_mmv(mmv_x, wtype, act_q8, d_ids_h, d.d_out + (size_t)d.out_rows * n_out,
                          n_in, n_out, p.n_slots, (int64_t)p.expert_size,
                          n_hits, /*act_rows=*/act_n, s, mmv_gate, mmv_glu);
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

    // fused-GLU learning must happen before any early return: record this
    // node's dst base so the GLU hook can prove the pair wiring per layer
    if (g.cur_n_tokens == 1 && g.cur_pool >= 0 && d.pools[g.cur_pool].paired && n_hits > 0 &&
        g.cur_blk >= 0 && g.cur_blk < 1024 && (g.cur_role == 0 || g.cur_role == 1)) {
        int k0l = -1;
        for (int k = 0; k < g.cur_n_ids; k++) {
            if (g.cur_slot_idx[k] >= 0) { k0l = k; break; }
        }
        if (k0l >= 0) {
            const char * dbase = (const char *)dst_rows[0] - (size_t)k0l * n_out * sizeof(float);
            if (g.cur_role == 0) {
                g.learn_gate_dst[g.cur_blk] = dbase;
                g.glu_learn[dbase] = g.cur_blk;
                EC3_DBG("[ec3-dbg] gate-dst blk=%d %p\n", g.cur_blk, (const void *)dbase);
            } else {
                g.learn_up_dst[g.cur_blk] = dbase;
            }
        }
    }

    // gate-defer: postpone this sync into the same layer's up node (only on
    // layers where up was observed to directly follow gate — see begin())
    if (g.defer && g.cur_n_tokens == 1 && g.cur_role == 0 && !d.pending_active && n_hits <= 64 &&
        g.cur_blk >= 0 && g.cur_blk < 1024 && g.safe_defer_blk[g.cur_blk] &&
        !(g.fuse && g.cur_pool >= 0 && d.pools[g.cur_pool].paired && g.safe_fuse_blk[g.cur_blk])) {
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

    // ---- fused gate+up+GLU path (paired pools) ----
    if (g.cur_n_tokens == 1 && g.cur_pool >= 0 && d.pools[g.cur_pool].paired && n_hits > 0 &&
        g.cur_blk >= 0 && g.cur_blk < 1024 && (g.cur_role == 0 || g.cur_role == 1)) {
        // reconstruct this node's dst base from the first hit's ids position
        int k0 = -1;
        for (int k = 0; k < g.cur_n_ids; k++) {
            if (g.cur_slot_idx[k] >= 0) { k0 = k; break; }
        }
        const char * dst_base = k0 >= 0
            ? (const char *)dst_rows[0] - (size_t)k0 * n_out * sizeof(float) : nullptr;

        if (g.safe_fuse_blk[g.cur_blk]) {
            if (g.cur_role == 0) {
                // gate node of a fused layer: d_out holds the fused swiglu rows.
                // D2H them (async) and hand off to the GLU hook; nothing is
                // written to the gate/up MMID dsts (the GLU kernel skips these
                // rows and nothing else reads them).
                auto & f = d.fused;
                f.active   = true;
                f.scattered = false;
                f.serial   = ++g.fuse_serial;
                f.gate_dst = dst_base;
                f.up_dst   = nullptr;   // filled at the up node
                f.mask     = 0;
                f.n        = 0;
                f.n_out    = n_out;
                for (int k = 0; k < g.cur_n_ids && f.n < 64; k++) {
                    if (g.cur_slot_idx[k] < 0) continue;
                    f.rows[f.n++] = k;
                    f.mask |= 1ull << k;
                }
                const size_t bytes = (size_t)d.out_rows * n_out * sizeof(float);
                CUDA_CHECK(cudaMemcpyAsync(d.h_out, d.d_out, bytes, cudaMemcpyDeviceToHost, d.compute_stream));
                d.out_rows = 0;
                d.q8_act_ptr = nullptr;
                d.fused_layers++;
                d.t_coll_us += ggml_time_us() - t0;
                if (g.stats_every > 0 && ++g.collect_calls % g.stats_every == 0) ec3_stats();
                return;
            }
            if (g.cur_role == 1) {
                // up node of a fused layer: nothing was dispatched; just record
                // the up dst base so the GLU hook can match the pair
                if (d.fused.active && dst_base) d.fused.up_dst = dst_base;
                d.out_rows = 0;
                d.t_coll_us += ggml_time_us() - t0;
                return;
            }
        }
    }

    // ---- GPU-resident dst handoff (down nodes) ----
    // If the scheduler offered the consumer's GPU copy of this dst, scatter the
    // hit rows straight into it (peer write, async) and skip the D2H + host
    // scatter + thread-0 sync entirely. CPU miss rows are uploaded later in
    // redirect_finalize (after the node barrier, when they are complete).
    if (g.redirect_on && g.cur_n_tokens == 1 && g.cur_role == 2 && n_hits > 0 && d.pending_rows == 0 && n_hits <= 64) {
        ec3_global::redirect_entry * re = nullptr;
        const void * base = nullptr;
        for (auto & kv : g.redirect) {
            const char * b = (const char *)kv.first;
            if ((const char *)dst_rows[0] >= b &&
                (const char *)dst_rows[0] <  b + (size_t)kv.second.n_rows * kv.second.nb1) {
                re = &kv.second;
                base = kv.first;
                break;
            }
        }
        if (re && re->n_rows <= 64) {
            // P2P-free relay: async-D2H each hit row into a pinned full-tensor
            // image at its TARGET row offset, record an event. No host sync.
            // redirect_finalize fills the miss rows into the same image and
            // issues one H2D on the consumer's stream, ordered by the event.
            const size_t img_cap = 64 * re->nb1;
            if (d.h_redir_half < img_cap) {
                if (d.h_redir) cudaFreeHost(d.h_redir);
                CUDA_CHECK(cudaMallocHost((void **)&d.h_redir, 2 * img_cap));
                d.h_redir_half = img_cap;
            }
            d.redir_par ^= 1;
            const int par = d.redir_par;
            char * img = d.h_redir + (size_t)par * d.h_redir_half;
            if (!d.redir_evt[par]) {
                CUDA_CHECK(cudaEventCreateWithFlags(&d.redir_evt[par], cudaEventDisableTiming));
            }
            uint64_t mask = 0;
            for (int i = 0; i < n_hits; i++) {
                const int ridx = (int)(((const char *)dst_rows[i] - (const char *)base) / re->nb1);
                mask |= 1ull << ridx;
                CUDA_CHECK(cudaMemcpyAsync(img + (size_t)ridx * re->nb1,
                                           d.d_out + (size_t)i * n_out,
                                           n_out * sizeof(float),
                                           cudaMemcpyDeviceToHost, d.compute_stream));
            }
            CUDA_CHECK(cudaEventRecord(d.redir_evt[par], d.compute_stream));
            re->hit_mask  = mask;
            re->populated = true;
            re->dev       = di;
            re->par       = par;
            d.out_rows = 0;
            d.t_coll_us += ggml_time_us() - t0;
            if (g.cur_role >= 0 && g.cur_role < 3) {
                d.t_coll_role_us[g.cur_role] += ggml_time_us() - t0;
                d.n_coll_role[g.cur_role]++;
            }
            if (g.stats_every > 0 && ++g.collect_calls % g.stats_every == 0) ec3_stats();
            return;
        }
    }

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
    const int64_t dt = ggml_time_us() - t0;
    d.t_coll_us += dt;
    if (g.cur_role >= 0 && g.cur_role < 3) {
        d.t_coll_role_us[g.cur_role] += dt;
        d.n_coll_role[g.cur_role]++;
    }

    if (g.stats_every > 0 && ++g.collect_calls % g.stats_every == 0) {
        ec3_stats();
    }
}

// ---- API: GPU-resident dst handoff ----------------------------------------------------

static void ec3_redirect_offer(const void * host_dst_data, size_t nb1, int64_t n_rows,
                               void * gpu_copy_data, void * consumer_backend) {
    GGML_UNUSED(consumer_backend);
    if (!g.redirect_on || n_rows > 64) return;
    ec3_global::redirect_entry re;
    re.nb1     = nb1;
    re.n_rows  = n_rows;
    re.gpu_ptr = gpu_copy_data;
    g.redirect[host_dst_data] = re;
}

static int ec3_redirect_finalize(const void * host_dst_data, void * consumer_backend) {
    auto it = g.redirect.find(host_dst_data);
    if (it == g.redirect.end()) return 0;
    ec3_global::redirect_entry re = it->second;
    g.redirect.erase(it);
    if (!re.populated || re.dev < 0) return 0;

    ec3_device & d = g.dev[re.dev];
    char * img = d.h_redir + (size_t)re.par * d.h_redir_half;

    // fill the miss rows into the pinned image (the node barrier has passed:
    // the CPU-computed rows are complete in host dst memory)
    for (int r = 0; r < (int)re.n_rows; r++) {
        if (re.hit_mask & (1ull << r)) continue;
        memcpy(img + (size_t)r * re.nb1,
               (const char *)host_dst_data + (size_t)r * re.nb1, re.nb1);
        d.redirect_misses_up++;
    }

    // one H2D of the full image on the CONSUMER's own stream, ordered behind
    // the hit-row D2H copies via the event. No host sync anywhere, no P2P.
    ggml_backend_t be = (ggml_backend_t)consumer_backend;
    ggml_backend_cuda_context * cc = (ggml_backend_cuda_context *)be->context;
    ggml_cuda_set_device(cc->device);
    CUDA_CHECK(cudaStreamWaitEvent(cc->stream(), d.redir_evt[re.par], 0));
    CUDA_CHECK(cudaMemcpyAsync(re.gpu_ptr, img, (size_t)re.n_rows * re.nb1,
                               cudaMemcpyHostToDevice, cc->stream()));

    d.redirect_claims++;
    return 1;
}

// ---- API: fused GLU hook ---------------------------------------------------------------

static unsigned long long ec3_glu_hits(const void * src0_data, const void * src1_data,
                                       void * dst_data, size_t dst_nb1, int ith) {
    // learning: observing the GLU node whose inputs are a layer's gate/up MMID
    // dsts proves the fused dispatch is safe for that layer
    if (!g.fuse) return 0;
    if (ith == 0) {
        EC3_DBG("[ec3-dbg] glu call src0=%p src1=%p\n", src0_data, src1_data);
    }
    auto lit = g.glu_learn.find(src0_data);
    if (lit == g.glu_learn.end()) return 0;
    const int blk = lit->second;
    if (blk >= 0 && blk < 1024 && g.learn_up_dst[blk] == src1_data && !g.safe_fuse_blk[blk]) {
        g.safe_fuse_blk[blk] = true;
        EC3_DBG("[ec3-dbg] fuse-safe blk=%d\n", blk);
    }

    // active fused rows for this pair? dst buffers are reused across layers,
    // so several devices can hold matching (stale) entries — take the newest
    int best = -1;
    long long best_serial = -1;
    for (int di = 0; di < g.n_dev; di++) {
        auto & f = g.dev[di].fused;
        if (!f.active || f.gate_dst != src0_data || f.up_dst != src1_data) continue;
        if (f.serial > best_serial) { best_serial = f.serial; best = di; }
    }
    {
        const int di = best;
        if (di < 0) return 0;
        auto & f = g.dev[di].fused;
        if (ith == 0 && !f.scattered) {
            ec3_device & d = g.dev[di];
            ggml_cuda_set_device(di);
            CUDA_CHECK(cudaStreamSynchronize(d.compute_stream));   // D2H of fused rows
            for (int i = 0; i < f.n; i++) {
                memcpy((char *)dst_data + (size_t)f.rows[i] * dst_nb1,
                       d.h_out + (size_t)i * f.n_out,
                       f.n_out * sizeof(float));
            }
            f.scattered = true;
        }
        return f.mask;
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
        EC3_LOG("[ec3] dev=%d decomp: compulsory=%lld capacity=%lld admission=%lld inflight=%lld uniq-seen=%zu uniq-inserted=%zu | skips: throttle=%lld budget=%lld qfull=%lld lru=%lld\n",
                i, d.miss_compulsory, d.miss_capacity, d.miss_admission, d.queued_misses,
                d.ever_seen.size(), d.ever_inserted.size(),
                d.skip_throttle, d.skip_budget, d.skip_qfull, d.skip_lrubusy);
        for (int pi = 0; pi < d.n_pools; pi++) {
            const long long ptot = d.pool_hits[pi] + d.pool_miss[pi];
            EC3_LOG("[ec3] dev=%d pool[%d]: hits=%lld/%lld (%.1f%%) slots=%d slot=%zuKB\n",
                    i, pi, d.pool_hits[pi], ptot, ptot ? 100.0 * d.pool_hits[pi] / ptot : 0.0,
                    d.pools[pi].n_slots, d.pools[pi].expert_size >> 10);
        }
        if (d.n_nodes > 0) {
            EC3_LOG("[ec3] dev=%d timing: nodes=%lld plan=%.1fus disp=%.1fus coll=%.1fus per-node total=%.1fus\n",
                    i, d.n_nodes,
                    (double)d.t_plan_us / d.n_nodes, (double)d.t_disp_us / d.n_nodes,
                    (double)d.t_coll_us / d.n_nodes,
                    (double)(d.t_plan_us + d.t_disp_us + d.t_coll_us) / d.n_nodes);
            EC3_LOG("[ec3] dev=%d redirect: claims=%lld miss-rows-up=%lld fused-layers=%lld\n",
                    i, d.redirect_claims, d.redirect_misses_up, d.fused_layers);
            EC3_LOG("[ec3] dev=%d coll-by-role: gate=%.1fus(n=%lld) up=%.1fus(n=%lld) down=%.1fus(n=%lld)\n", i,
                    d.n_coll_role[0] ? (double)d.t_coll_role_us[0]/d.n_coll_role[0] : 0.0, d.n_coll_role[0],
                    d.n_coll_role[1] ? (double)d.t_coll_role_us[1]/d.n_coll_role[1] : 0.0, d.n_coll_role[1],
                    d.n_coll_role[2] ? (double)d.t_coll_role_us[2]/d.n_coll_role[2] : 0.0, d.n_coll_role[2]);
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
    if (const char * e = getenv("LLAMA_EC3_THROTTLE"))  { g.throttle_mod = atoi(e); if (g.throttle_mod < 1) g.throttle_mod = 1; }
    if (const char * e = getenv("LLAMA_EC3_GREEDY_LAST")) g.greedy_last = atoi(e) != 0;
    if (const char * e = getenv("LLAMA_EC3_WORKERS"))   { int n = atoi(e); if (n > 0 && n <= 16) g.n_workers = n; }
    if (const char * e = getenv("LLAMA_EC3_STATS"))     g.stats_every = atoi(e);
    if (const char * e = getenv("LLAMA_EC3_MIN_EXPERT_KB")) g.min_expert_bytes = (size_t)atoll(e) << 10;
    if (const char * e = getenv("LLAMA_EC3_RESERVE_MB"))    g.reserve_mb = (size_t)atoll(e);
    if (const char * e = getenv("LLAMA_EC3_DEFER"))         g.defer = atoi(e) > 0;
    if (const char * e = getenv("LLAMA_EC3_REUSE"))         g.reuse = atoi(e) > 0;
    if (const char * e = getenv("LLAMA_EC3_STRIPE"))        g.stripe = atoi(e) > 0;
    if (const char * e = getenv("LLAMA_EC3_FUSE"))          g.fuse = atoi(e) > 0;
    if (const char * e = getenv("LLAMA_EC3_MAX_BATCH"))     { int n = atoi(e); if (n >= 1 && n <= 8) g.max_batch = n; }
    if (g.stripe && g.fuse) {
        g.fuse = false;  // pair state is per-device; striping splits roles across devices
        EC3_LOG("[ec3] stripe mode: fuse disabled\n");
    }
    if (g.stripe) {
        // gate and up land on different devices: the defer absorb would dangle
        // past the swiglu read (corruption), and the act-quant reuse state is
        // per-device — both must be off in striped mode
        g.defer = false;
        g.reuse = false;
        EC3_LOG("[ec3] stripe mode: defer/reuse disabled\n");
    }

    ggml_expert_cache_v3.begin    = ec3_begin;
    ggml_expert_cache_v3.plan     = ec3_plan;
    ggml_expert_cache_v3.dispatch = ec3_dispatch;
    ggml_expert_cache_v3.collect  = ec3_collect;
    ggml_expert_cache_v3.stats    = ec3_stats;
    ggml_expert_cache_v3.redirect_offer    = ec3_redirect_offer;
    ggml_expert_cache_v3.redirect_finalize = ec3_redirect_finalize;
    ggml_expert_cache_v3.glu_hits          = ec3_glu_hits;
    if (const char * e = getenv("LLAMA_EC3_REDIRECT")) g.redirect_on = atoi(e) > 0;

    EC3_LOG("[ec3] enabled: n_dev=%d budget=%s inserts/plan=%d workers=%d stats_every=%d\n",
            g.n_dev, g.budget_mb ? "env" : "auto-70%-free", g.inserts_per_plan,
            g.n_workers, g.stats_every);

    if (const char * e = getenv("LLAMA_EC3_SELFTEST"); e && atoi(e) > 0) {
        ec3_selftest();
    }
}
