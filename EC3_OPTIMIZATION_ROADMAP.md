# EC3 Further-Optimization Roadmap — merged multi-agent analysis (2026-06-11)

Produced by two adversarially-verified workflows (62 agents): general SOTA/source analysis + cache-axis with live 754B experiments.

## PART I — General optimization roadmap

# EC3 Optimization Roadmap — GLM-5.1 754B Decode (post-adversarial-review synthesis, 2026-06-11)

## 1. Executive Summary

From the current 17.9 t/s, the adversarially-adjusted, realistically-stackable headroom is **+2.0 to +3.5 t/s near-term (→ ~20–21.5 t/s)**, with the cache-aware expert-drop lever raising the theoretical ceiling from ~24 to ~25.5–26.5 t/s once the overhead gap is closed. Every surviving proposal's headline gain was deflated 1.5–2x on review, almost always for the same reason: **the GPU mmv chain is exec-bound (memory-bandwidth-bound), not launch-bound** — at 75% hit, ~5.7ms/token of the 8–9ms "overhead" is irreducible VRAM reads, and CPU miss work (~10µs/node across 47T) is hidden under the 37–44µs chain wait. The three dominant themes that survive: **(A) delete the down-dst D2H→scatter→re-H2D round trip and its blocking syncs** (the single strongest, code-confirmed waste — three independent sources converged on it, one verdict "strong"); **(B) shrink and parallelize the GPU chain itself** (gate+up fused mmvq using existing fusion-args, host-side q8_1 quantize, multi-GPU striping of hit rows — the only lever that cuts the exec floor); **(C) near-free config/policy fixes uncovered by audit** (pool VRAM split is actually 2:1 with ~25% of budget unallocated and down starved 33%; per-pool stats already exist behind `LLAMA_EC3_STATS`). Gains across themes A and B partially overlap (several attack the same 37–44µs collect wait) — do not sum naively; the table below already accounts for the dominant overlaps.

## 2. Ranked Roadmap

Ranked by expected-value per effort, with dependency ordering. Gains are reviewer-adjusted, standalone unless noted.

| # | Lever | Adjusted gain (t/s) | Effort | Risk | Source |
|---|-------|--------------------|--------|------|--------|
| 1 | **Pool VRAM rebalance** (weighted 57.4/42.6 split; first run the free `LLAMA_EC3_STATS` probe — counters already exist) | +0.2–0.5 @500-tok horizon; +0.5–1 on ceiling; ~0 @tg200 | 0.5 day (~15 LOC) | Low | code-audit |
| 2 | **Down-dst GPU-resident handoff** (kill D2H→scatter→re-H2D round trip + blocking syncs; merged from 3 sources, one verdict *strong*) | +0.5–0.9 | 2–4 days (~250–300 LOC) | Medium | code-audit ×2 + engine |
| 3 | **Role-stripe probe** (`di = (blk*3+role) % n_dev`, breaks gate/up same-stream BW serialization) → gate full **per-eid striping** on result | probe +0.3–0.5; full +0.6–0.9 (needs #4/#5 first) | probe: hours (~10 LOC); full: 4–7 days | probe Low / full Med-high | code-audit |
| 4 | **Gate+up+SwiGLU fused mmvq** via existing `fusion_args` path (paired slots, 2 chains/layer → 1) | +0.3–0.5 | 2–3 days (~300 LOC) | Low | hardware-engine |
| 5 | **Host-side q8_1 quantize + single merged H2D** (4 stream ops → 2; act bytes ÷3.5) | +0.2–0.4 | 1–2 days (~150 LOC) | Low-med | code-audit |
| 6 | **Flag-poll sync replacement** (tail kernel + mapped volatile flag + CPU spin; include mapped-host mmvq dst) | +0.15–0.4 | 1 day (~100 LOC) | Low | hardware-engine |
| 7 | **Device-direct activation quantize** (quantize_q8_1 reads the GPU-resident input tensor; standalone subset of early-ids hook) | +0.1–0.2 | ~50 LOC | Low | code-audit |
| 8 | **Cache-aware miss-row expert dropping** (ik_llama `-ser` PR #239, position-proxy form on miss rows only) | +0.0–0.3 now; ceiling 24 → 25.5–26.5 | 1 day (~20–40 LOC) + PPL A/B | Low-med (quality) | engine (ik_llama) |
| 9 | **Early-ids pre-issue hook** (full version, beyond #7) | +0.25–0.6 | 2–3 days | Low-med | code-audit |
| 10 | **Full persistent doorbell megakernel** — only if #6's probe shows >10µs/node of sync-detection cost | +0.7–1.0 (corrected large-grid design) | 7–12 days | High | hardware-engine |

Stacked realistic outcome of #1–#7: **~20–21 t/s**; #8 then converts remaining ceiling headroom as the overhead gap closes.

## 3. Top-5 Implementation Sketches

### 3.1 Pool VRAM rebalance (rank 1)

**Probe first (5 min, zero LOC):** the proposal's "add per-pool stats" premise is false — `pool_hits`/`pool_miss` exist at `expert-cache.cu:97`, incremented at `:484/:490/:496`, printed by `ec3_stats` at `:774-778`. Run production launcher with `LLAMA_EC3_STATS=N` set (it currently isn't — confirmed in `/tmp/ec3-server2.log`).

**Audit finding (direction inverted vs original proposal):** `ec3_pool_alloc` (`expert-cache.cu:295-347`) re-queries `cudaMemGetInfo` per allocation with `pool_div` clamped ≥2 (`:311`), and pools allocate sequentially from `g_disc.pending` (`:392-393`). Actual split: gate/up pool gets avail/2, down gets avail/4, **~25% of budget never allocated**. Down is starved 33% in entries-per-set, not gate/up.

**Fix:** weight the split by `set_count × expert_size` → gate/up 57.4%, down 42.6% (Lagrange-optimal: equal logical-expert capacity per pool). Both pools *grow* since the unallocated 25% gets used. ~15 LOC in `ec3_pool_alloc`.

**Validation:** 500-token tg run (production regime — evictions begin >14K entries at 200 tok) with `LLAMA_EC3_STATS`, compare per-pool hit rates and t/s vs the 17.6 @500-tok baseline.
**Kill criteria:** stats show down hit-rate within 10pp of gate/up (no asymmetry to fix), or rebalanced run shows no blended hit-rate gain at 500 tokens.

### 3.2 Down-dst GPU-resident handoff (rank 2)

**Verified waste:** `ec3_collect` does blocking D2H + streamSync + CPU scatter (`expert-cache.cu:727-748`); `cpy_tensor_async` rejects CPU-src copies (`ggml-cuda.cu:3237-3243`), forcing the scheduler's host-blocking sync fallback at `ggml-backend.cpp:1687-1694` to re-upload the ~200KB down dst the EC3 just downloaded. Cache device == consumer device (both `blk%4`).

**Pre-probe (30 min):** split `t_coll_us` by role; instrument the 1687-1694 path for down-dst inputs to measure true sched-copy cost. If it measures >40µs/node, upside approaches +1 t/s.

**Implementation:** (a) extend `ggml_expert_cache_v3_api` (`ggml/src/ggml-backend-expert-cache.h`) with `set_dst_redirect(host_dst, dev_ptr, dev, strides)` + `finalize_redirect(host_dst, miss_rows) → event`; (b) in `ggml_backend_sched_compute_splits` (`ggml-backend.cpp:1581`), before a CPU split whose last node is MUL_MAT_ID, pointer-match the following split's `input_cpy` (`tensor_copy` macro :838, `hv_tensor_copies` :792) and hand its device pointer to EC3; (c) in `expert-cache.cu`, batched mmvq writes hit rows, a small same-stream scatter kernel writes them into `input_cpy`; down collect becomes a no-op for hits; (d) `finalize_redirect` H2Ds only miss rows (~50KB pinned-staged) and records an event; sched does `cudaStreamWaitEvent` on the consuming backend stream instead of synchronize. EC3 stream must wait on a GPU-backend event before writing (gallocr reuse rule, `:1593-1597`). Silent fallback to current path on mapping miss, multi-consumer dst, or device mismatch.

**Validation:** existing selftest; PPL@ub=1 gate (bug-6 defer-corruption history mandates it); poison-probe; A/B debug mode falling back to blocking path with output comparison; tg200 + 500-tok with production launcher.
**Kill criteria:** pre-probe shows down collect+sched-copy <20µs/node combined (nothing to reclaim), or post-build tg200 gain <0.3 t/s, or any PPL/output divergence vs blocking path.

### 3.3 Role-stripe probe → per-eid striping (rank 3)

**Probe (hours):** change `di = blk % g.n_dev` (`expert-cache.cu:361`) to `di = (blk*3 + role) % g.n_dev` behind an env flag. Gate/up currently share one device AND one stream (gate-defer overlaps issue, not bandwidth): 38MB ≈ 44µs for the pair. Role-striping splits that to ~22µs/layer → predicted ~1.5ms/token, +0.3–0.5 t/s. Note: cache refill transient after keying change — warm up before measuring.

**Full build (only after #4/#5 land — hard dependency, 4 chains at today's 13µs dispatch = ~52µs serial issue erases the gain):** key cache by expert id (`eid % 4`), plan partitions the 8 ids into per-device hit lists (per-device pools/streams/staging/insert-workers already symmetric), dispatch issues up to 4 slim chains, collect spin-polls 4 flags. Gate/up of a layer route identical eids, so per-device partitions match and gate-defer + q8-reuse survive per device. E[max] of ~6 hits over 4 devs ≈ 2.7 rows → max-device exec ~45% of today. Optional `+150 LOC` per-device dispatcher threads if thread-0 issue serialization shows up.

**Validation:** tg200 A/B via `LLAMA_EC3` env flag with production launcher; per-device hit/exec timing counters.
**Kill criteria for full build:** role-stripe probe delivers <1ms/token (per-chain latency floor higher than modeled — 1–3-row mmvq kernels are latency-dominated at a 5–10µs floor) → shelve.

### 3.4 Gate+up+SwiGLU fused mmvq (rank 4)

**Verified:** EC3's decode path already dispatches through `mul_mat_vec_q_switch_fusion` (`mmvq.cu:889-911`); the `has_fusion` branches consuming `ggml_cuda_mm_fusion_args_device {gate, glu_op}` (`mmvq.cu:475-672`, struct `common.cuh:1505-1514`) apply SWIGLU on-chip with post-GLU output (`:651-668`). `ggml_cuda_ec3_mmv` (`mmvq.cu:1269`) currently passes empty `fusion_local`.

**Implementation:** admit/evict gate+up as a pair per expert. **Key gotcha (from review):** the fused kernel reuses the same `kbx_offset` for `vgate` (`mmvq.cu:584-588`) — gate/up must sit at identical slot indices with identical strides, so the shared (3.17MB, type-16) pool needs a layout change: interleaved pair-slots with 2× stride or twin sub-pools. Pass `fusion.gate` + `GGML_GLU_OP_SWIGLU` in `ggml_cuda_ec3_mmv`; scatter destination moves from up-dst to swiglu-dst; CPU computes swiglu for miss rows only (it already computes their matvecs); hit/miss masking must be consistent across the pair. Pairing costs ~no hit rate (co-routed experts have identical recency).

**Validation:** selftest (fp32-reference compare) extended with a fused-pair case; PPL@ub=1; tg200.
**Kill criteria:** fused path saves <10µs/layer in chain timing counters (would mean exec time dominates even harder than modeled), or paired admission drops blended hit rate >2pp.

### 3.5 Host-side q8_1 quantize + merged H2D (rank 5)

**Verified:** `emit_chain` (`expert-cache.cu:657-672`) issues 4 serial stream ops (ids-H2D, act-H2D, quantize_q8_1, mmv); `quantize_q8_1` output is plain contiguous `block_q8_1` (`quantize.cu:29-35`); `ggml_cuda_ec3_mmv` takes a raw q8 device pointer with explicit strides (`mmvq.cu:1254-1278`). CPU/GPU q8_1 differ in the `s` term, but IQ2_XXS/IQ3_XXS vec_dot uses only `d` — inert, selftest covers it.

**Implementation:** one pinned staging block `[ids | dst_ptrs | q8 act rows]` filled on host (`quantize_row_q8_1` + MATRIX_ROW_PADDING zero-pad), ONE `cudaMemcpyAsync` + mmv. For down nodes, pre-quantize the 8 rows in the from_float pre-barrier section of `ggml-cpu.c` (`:1587-1681`) guarded by a per-node atomic done-counter (no barrier exists between from_float and thread-0 dispatch). Keep gate→up q8 reuse pointing at the staged region. **Scope note:** up nodes already run a 2-op chain via `reuse_q8` — benefit lands on gate/down nodes only (~⅔ of the 150 cache-active nodes).

**Validation:** existing selftest (already compares vs fp32 with q8 tolerance — catches block-format/padding mismatches immediately); tg200.
**Kill criteria:** wall-clock saving <4µs on gate/down nodes (would confirm issue cost fully hidden under chain exec) — but keep the staging-block layout anyway, since it's the prerequisite for the striping build's slim chains (#3) and the memcpy+mmv+scatter ~35µs/node floor.

## 4. Not Worth It — verified dead ends (do not retry)

- **Cross-layer ids prefetch / lead-time scheduling: structurally impossible** (verdict *strong*). `deepseek2.cpp:372-422`: ids(L+1) depends on moe_out(L) through attention(L+1); router/topk are the last GPU ops before the split boundary, so no intra-split window exists either. Measured 74–77% hit already equals the destructive-mask ceiling hit rate — recency-based prefetch buys nothing. Only a learned predictor beating temporal locality helps, and that whole class (HOBBIT/ProMoE/Fate/FineMoE/MoE-SpeQ/pre-attention prediction) is absorbed literature / refuted proposals.
- **CPU kernel ports (ik_llama iqk, KTransformers AVX2):** microbenchmark on this EPYC 7R13 @47T showed IQ3_XXS vec_dot at ~95% of the 146 GB/s DRAM ceiling (zero headroom on down) and IQ2_XXS at 73–84% (max ~+0.6 t/s best case, ~0 while the miss term stays hidden under the GPU chain). ik's 1.6× table (github.com/ikawrakow/ik_llama.cpp/discussions/164) doesn't transfer — 47 threads saturate DDR4. KT 0.5.3 has no IQ2-class kernels; KT expert scheduling = static placement, already measured losing (16.05 vs 17.9). *Standing caveat:* re-evaluate a 1–2 day IQ2_XXS vec_dot tune only after GPU-chain latency drops enough to expose the miss term.
- **Launch-overhead elimination as a theme** (CUDA-graph capture: measured net loss; megakernel headline claims): the chain is exec-bound. Mirage/Hazy-style citations are all-GPU workloads; here ~5.7ms/token is irreducible VRAM bandwidth. Only sync-*detection* cost (~3–8µs) is recoverable (rank 6).
- **Speculative-decode family** (plain MTP, cache-only self-spec, draft-pass lookahead): verify cost scales with batch on CPU; all three refuted on arithmetic.
- **Eviction/admission policy zoo** (LFU, TinyLFU doorkeeper, buckets, profiles): all ±3% in v2; working set fits at tg200; the real capacity lever was the pool-split bug (rank 1).
- **Misc refuted:** TEAL-style block sparsity (wrong quant geometry for IQ2_XXS/IQ3_XXS), REAP offline pruning (gap misattribution), 2MB huge pages (already aligned-alloc'd, 5–10× deflated), pinned compute buffer (already pinned via `llama-context.cpp:336-343`), CPU-side weighted sum (dominated by rank-2 direct write), bigger cache budget (saturates).

**Recurring failure mode to check in future proposals:** any gain model that counts the 8–9ms/token "overhead" or the 37–44µs collect wait as recoverable is double-counting bandwidth-bound mmv exec; and anything "moved off the thread-0 critical path" must account for the scheduler thread *being* CPU compute thread 0 (`ggml-cpu.c:3418`).

---

## PART II — Cache-axis roadmap (with live experiments)

# EC3 Cache-Axis Optimization Roadmap (synthesis of verified findings, 2026-06-10)

Baseline: EC3 17.89±1.15 t/s tg200 (stock 14.0), 74-80% steady hit, pools full by ~200 tokens. Session-measured run noise is large (same-config runs 17.52 vs 15.84), so all validation below requires ≥3 reps or tg500+ per arm.

## Strategic frame (verified "strong")
Hit-rate work above ~80% has near-zero or negative marginal t/s under the current serial-collect design. Three independent measurements triangulate: mask ceiling ~24 t/s at the SAME hit rate; Run C (-1.7 t/s at constant hit from churn); Run D (+3.7-5pts hit, -2.4 t/s from VRAM pressure). Collect (GPU chain wait) is 35-40us of ~50us/node. **The cache's content is within ~1-2pts of its constant-VRAM optimum; the remaining >1 t/s levers make hits cheaper, not more numerous.**

## Ranked roadmap (adjusted-gain / effort)

| # | Item | Adjusted gain | Effort | Ratio |
|---|------|--------------|--------|-------|
| 1 | THROTTLE=16/32 probe (stricter direction, untested) | +0.2-0.5 t/s | ~0 (knob exists) + 2 bench runs | highest |
| 2 | Pool divisor fix + constant-total value-weighted rebalance | +0.2-0.6 t/s | ~10 lines + free stats check + 3-rep bench | high |
| 3 | Keep-rows-on-device collect surgery (cash in existing 80% hit) | +1.5-2.5 t/s | multi-day | medium, but only >1 t/s lever left |
| 4 | nsys root-cause of Run D's ~10ms residual VRAM-pressure cost | 0 direct; unlocks +2-3 t/s latent (Run D's 84.6% hit penalty-free) | ~0.5 day | medium |
| 5 | Per-role + joint gate/up hit stats (gates paired-admission, ceiling +0.2-0.5) | 0 direct | ~12-15 lines (per-pool stats already exist) | low; bundle with #2 |
| 6 | Queue/pool mutex split + move ec3_flush sync outside g.mu | +0.0-0.05 t/s | small | only if editing plan() anyway |

## Top 3: sketches, validation, kill-criteria

### 1. THROTTLE=16/32 probe
- **Sketch**: no code. `LLAMA_EC3_THROTTLE` already shipped (expert-cache.cu lines 122/520/917, default 8). Run 754B llama-bench tg500 with `LLAMA_EC3_THROTTLE=16` (then 32 if 16 wins), `LLAMA_EC3_STATS=100`.
- **Why it should work**: Run C proved admission-rejected misses convert ~1:1 into capacity misses when admitted (LRU churn) — the same conversion run in reverse predicts tightening costs ~0 hit rate while cutting Run B's residual ~600 inserts/dev/interval (~2.6 GB/s host BW stolen from CPU matmuls).
- **Validation**: steady inserts/dev/interval drop ~582→~290; steady hit within 1pt of 79.6%; t/s vs ≥3-rep baseline.
- **Kill**: steady hit drops >2pts (slow hot-set adaptation) or t/s ≤ baseline mean → declare 1-in-8 locally optimal in BOTH directions, axis fully closed.

### 2. Pool divisor fix + constant-total rebalance
- **Pre-step (free, do first)**: read `pool_hits[]/pool_miss[]` per-pool split from existing /tmp/ec3-exp STATS logs. If pool[1] (down) within ~3pts of pool[0], expected gain collapses — skip to kill.
- **Sketch**: `ec3_pool_alloc` (expert-cache.cu:302-316): the `pool_div = max(n_shapes_pending, 2)` floor makes the LAST pending shape allocate at avail/2, stranding ~4.3GB/dev. Replace the sequential halving loop (ec3_begin ~386-393) with one proportional allocation over all pending shapes weighted by referenced bytes per role — gate/up:down = 57.4:42.6 — at **constant total ~12.9GB/dev** (pool0 ~7.6GB/2400 slots, pool1 ~5.3GB/1150 slots). Keep the half-cap order-proofing for non-final shapes (bug-ledger #8 says it's deliberate). Bump RESERVE_MB +1GB: the stranded VRAM currently acts as de-facto headroom against the lazy-CUDA-pool OOM (bug-ledger #7).
- **Critical constraint**: do NOT grow total budget — Run D proved +4GB/dev total at +5pts hit LOSES 2.4 t/s via a ~10ms/token silent VRAM-pressure cost. Constant-total is the only safe variant.
- **Validation**: pool sizes in [ec3] pool log lines; ≥3 reps tg500 (effect size +0.1-0.6 vs ±1.7 same-config scatter demands it); llama-server long-context smoke test for the OOM cliff; per-pool steady hit (expect pool1 75→~81%, pool0 84→~81%, byte-weighted net positive since down = 42.6% of bytes and 1.48x CPU savings per hit).
- **Kill**: 3-rep mean ≤ baseline → rebalance axis closed; residual-10ms mechanism (#4) becomes the sole capacity-side target.

### 3. Keep-rows-on-device collect surgery
- **Sketch**: today every cached-row GPU output round-trips to host and collect blocks ~40us/node on the GPU chain. For rows where gate, up, AND down all hit (~60-70% of hit rows at 75%/role — probeable at gate plan time since the ids tensor is shared across the layer's three MUL_MAT_ID nodes), keep gate/up outputs on device, run silu*mul on GPU, feed down's cached matvec directly, and only sync the final down output for the host weighted-sum. CPU-side rows unchanged. Touches expert-cache.cu collect (~689) plus the per-cnode split execution in ggml-backend.cpp; the defer-stash + poison-debug machinery (expert-cache.cu:699-726) is the template. The per-layer down-collect sync survives (host reads dst immediately) — that bounds the gain.
- **Why this and not more hit rate**: it simultaneously cashes in the existing 80% hits AND removes the penalty that makes higher hit rates unprofitable. Deflated arithmetic: ~40-60us saved/layer × 75 layers on a 57ms token = +1.4-2.1 t/s.
- **Validation**: LLAMA_EC3_SELFTEST=1 ALL PASS → 35B PPL at -b1 -ub1 with LLAMA_EC3_MIN_EXPERT_KB=128 (bit-exactness gate) → 754B tg200 ≥3 reps. Decisive stat: collect us/node on hit-dominated nodes drops 40 → <15.
- **Kill**: prototype gate→up pair only first (~1 day); if PPL diverges, or collect time doesn't drop, or extrapolated full gain <+0.5 t/s, stop before the down-chain work.

## Hit-rate headroom decomposition (steady-state, Run B dev1, with verifier caveats)
- **In-flight / fill-latency: 0%.** Queue-full and queued-miss deltas are exactly 0 after the first ~50 tokens across all devs/runs. WORKERS/INSERTS/queue tuning is dead — answered without a run.
- **Compulsory: ~10% of misses (~2% of accesses).** Infinite-capacity ceiling ~98%; declining over the run.
- **Admission-rejected: ~54%** — but this bucket is *manufactured* by the deliberate anti-churn throttle (removing it costs 2.4 t/s; relaxing to 1-in-2 cost 1.7 t/s with NO hit gain). Not recoverable by admitting more; only by admitting *better at fixed bandwidth*, which the v2 zoo bounds at ±3%.
- **Capacity: ~36%** — caveat: the capacity-vs-admission split is a run-length artifact (cumulative ever-inserted set migrates keys between buckets, 1→41% over the run), so it is telemetry, not a causal guide. The one real capacity defect is the pool1 starvation addressed by item #2.
- **Net**: practical policy-addressable headroom is ~2-3.5 byte-weighted points (item #2), worth +0.3-0.7 t/s — everything beyond that is the cheaper-hits axis (item #3).

## Not worth it (closed, with reasons)
- **S3-FIFO / SIEVE port**: quick-demotion requires admitting every miss (3.2-4.7MB each = the exact churn the throttle exists to bound); SIEVE's lock saving bounded <0.05 t/s vs 2.9us plan cost; no scans exist in this workload (decode-only fill killed the only one). Do not build.
- **Fill-rate levers** (WORKERS, INSERTS, queue cap, warmup): in-flight misses are zero in steady state. Dead by measurement.
- **Loosening the throttle** (<1-in-8) or any admission-selectivity filter at current capacity: THROTTLE=2 = -1.7 t/s, no hit gain; miss stream already samples admission frequency-proportionally, so doorkeeper/TinyLFU/ghost filters gain ~nothing (consistent with v2 LFU/score nulls).
- **Shadow IQ1 cold-tail pool**: requant hard-asserts without imatrix (ggml-quants.c:4452), gate/up saving only 1.32x not 1.9x, mixed-type nodes +13us dispatch, PPL risk on down. Strictly dominated by item #2.
- **Lock-free LRU / standalone mutex work**: hard ceiling +0.21 t/s if plan were free; realistic +0.02. Below noise.
- **Dropped as weak**: TinyLFU count-min duel, ghost lists (telemetry variant subsumed by existing miss-decomposition counters), segmented LRU, ghost-feedback adaptive admission rate, per-device mutex shard, paired gate/up single-entry coupling (revisit ONLY if item #5's joint-hit stat shows pairs badly desynced AND the defer-flush cost at line 554 proves material — ceiling +0.2-0.5 t/s).
- **Previously closed, still closed**: bigger total budget (Run D re-confirmed: VRAM pressure makes it net negative), eviction-policy zoo, PP-driven fill, hybrid static placement, plain MTP spec-decode.

Key paths: /home/user/llama.cpp-v3/ggml/src/ggml-cuda/expert-cache.cu (all code changes), /tmp/ec3-exp/run{A,B,C,D}.err + analyze.py (evidence), branch v3-expert-cache, launcher /home/user/llama.cpp/start-glm51-ec3.sh.