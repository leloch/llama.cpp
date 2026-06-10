# EC3 PRODUCTION-READINESS REPORT

Audit scope: branch `v3-expert-cache` (HEAD 04d1a3c66), worktree `/home/user/llama.cpp-v3`. All findings below were independently verified against code and logs; severities reflect post-verification adjustments.

---

## 1. Executive verdict

**EC3 is NOT safe to ship always-on today, and is not safe even in its current opt-in form on anything other than the validated GLM/Qwen plain-SwiGLU targets.**

The core mechanism is sound and the wins are real and reproduced: **+34% on GLM-5.1 754B (19.2 vs 14.0 t/s bench)**, **+10.4% steady-state on Qwen3.5-397B (31.16 vs 28.22 t/s)**, PPL-validated, with verified-zero residue when dormant. But the audit found **five technical blockers**, two of which are *silent numerical corruption* bugs reachable in EC3's own default knob configuration (fused-GLU stale-entry match during warmup; gate-defer corruption on gpt-oss-class models), plus a crash-on-first-hit for F16/BF16 experts, a process-global singleton that serves wrong weights in any multi-model process, and CUDA_CHECK aborts on the very OOM pressure EC3 itself creates.

**Distance to "always-on, autofit-like, transparent":** the empirical matrix proves it is structurally impossible today — `--fit` packs VRAM to a 1 GiB margin, below EC3's 3 GiB reserve, so `LLAMA_EC3=1` with fit is a silent no-op (754B: 14.09 t/s vs 14.0 stock, plus 14,504 lines of stderr spam); the winning config requires manual `-ncmoe 99`, which is **-49% on models that fit in VRAM** (122B: 37.80 vs 74.68 t/s). Always-on requires: the five blockers fixed, the pool-ordering race fixed (the flagship 754B *server* currently regresses to 6.1–12.0 t/s, below 14.0 stock), warmup-window mitigation, slot-count capping, and an EC3-aware fit planner with a calibrated break-even gate and runtime bail-out. Estimated: **~3–4 weeks of engineering plus a multi-model validation sweep.**

**Distance to an upstream PR:** all of the above, plus a hard non-technical gate — **CONTRIBUTING.md prohibits predominantly AI-generated PRs** (edited-after counts as AI-generated; ban risk). This branch (~2,010 added lines, AI-authored commit messages) is categorically non-submittable as-is. The viable path is a human-written design-first feature proposal referencing issue #20757, followed by human re-authorship of a minimal core split into 3 PRs, with the hygiene/CI/config work (sections 3, M6–M11) done first.

---

## 2. Empirical model matrix

| Model / config | Engaged? | EC3 t/s vs stock t/s | Verdict |
|---|---|---|---|
| GLM-5.1 754B, llama-bench, `-ngl 99 -ncmoe 99` | Yes, full | **19.2 vs 14.0 (+34%)** | **Win** |
| GLM-5.1 754B, llama-server, same flags | Broken: 3/4 devices lost the down-pool (ordering race) | **6.08 cold / 12.04 warm vs 14.0 (-57% / -14%)** | **LOSS** |
| GLM-5.1 754B, `-fitt 1024` + `LLAMA_EC3=1` | Starved (free 1 GiB < 3 GiB reserve) | 14.09 vs 14.0 | Neutral no-op + 14,504 log lines |
| Qwen3.5-397B-A17B, `-ncmoe`, tg100 | Yes (filling) | **26.20 vs 28.22 (-7.2%)** | **LOSS (warmup window)** |
| Qwen3.5-397B-A17B, `-ncmoe`, tg500 | Yes, 89–92% hit | **31.16 vs 28.22 (+10.4%)** | **Win (steady state)** |
| Qwen3.5-122B-A10B, `-fitt` + EC3 | Dormant | 74.77 vs 74.68 | Correctly dormant |
| Qwen3.5-122B-A10B, EC3 + `-ncmoe 99` (documented usage) | Partial (down-only; gate/up 888 KB < 1 MB gate) | **37.80 vs 74.68 fitt (-49%)** | **LOSS (config-induced)** |
| Qwen3.5-35B-A3B, `-ncmoe`, EC3=1 vs unset | Auto-excluded (experts 576–840 KB) | 75.20/76.41 vs 77.17/74.95 (within noise) | Correctly dormant |
| Qwen3.5-35B-A3B, `-fitt` + EC3=1 | Dormant | 145.86, banner only | Correctly dormant |
| Qwen3.5-35B-A3B, forced on (gate bypassed) | Yes | 54 vs 75 (-28%) | Loss (why the gate exists) |

Cross-cutting safe results: 754B server survived an 11,105-token prompt with pools at 21/24.5 GB per device (VRAM grew only ~25 MB; pp 59.2 t/s) — the 3 GB reserve holds for the default buffer ordering. Off/dormant residue is verified zero (null-pointer short-circuits; `safe_fuse_blk` writable only from the swiglu_f32 hook).

---

## 3. Ranked fix list

**Legend:** [A] = required for always-on; [P] = required for upstream PR; [N] = required NOW, even for current opt-in use.

### Blockers

| # | Title | Fix | Effort |
|---|---|---|---|
| B1 [N][A][P] | **Fused-GLU stale-entry corruption in default config.** `d.fused.active` is never cleared (only write: expert-cache.cu:931); collect is skipped on zero-hit nodes (ggml-cpu.c:1757); gallocr reuses dst pointers — a later layer's GLU claims a previous layer's hit mask and skips rows nobody computed. At 40% warmup hit rate ×8 experts ×~90 layers ≈ corruption every few tokens, silently, with FUSE=on default. PPL runs don't exonerate it (plausible-magnitude garbage; ub=4 bypasses fuse). Adjacent bug: asymmetric gate/up hit sets + line 956 repointing a stale entry's `up_dst`. | One-shot layer-scoped fused state: store blk; clear `f.active` at the next `ec3_begin` on that device (NOT inside glu_hits — racy across CPU threads); invalidate on zero-hit gate nodes and on `ec3_flush`; require gate/up hit-set equality before engaging the skip. Cold-start stress test (`LLAMA_EC3_THROTTLE=64`, small pool). | ~1 day |
| B2 [N][A][P] | **Gate-defer "observed adjacency" proof is wrong for models with readers between the MMIDs.** Defer learning sees only MMID cache visits (expert-cache.cu:527-531); gpt-oss bias `ggml_add_id`, per-expert scales (qwen3moe/gemma4/mistral3/...), Step35 clamp, and MoE LoRA all read the deferred (unwritten) gate dst between gate and up — adjacency still learns, defer engages, silu(garbage)×up, silently. gpt-oss passes every gate (MXFP4 ~4.4 MB). DEFER defaults on. | Require fuse-level proof: defer only where the GLU hook observed gate-dst==src0 AND up-dst==src1 (extend glu_learn to non-paired pools, guard pointer staleness), or graph-level check at the offer site that the gate dst's only pre-up consumer is the GLU. gpt-oss regression test. | 1–2 days |
| B3 [A][P] | **Unsupported expert types (F16/BF16/TQ) hit `GGML_ABORT` on first cache hit.** ec3_begin has no wtype filter; `mul_mat_vec_q_switch_type` default-aborts (mmvq.cu:1118). Any F16/BF16 GGUF MoE conversion passes the 1 MB size gate and kills the process mid-decode. | Type allowlist in ec3_begin before shape tracking; host-side switch helper from mmvq.cu (note: `get_vec_dot_q_cuda` is `constexpr __device__`; needs re-annotation or explicit host switch). Selftest case asserting graceful refusal. | 2–3 h |
| B4 [A][P] | **Process-global singleton: wrong weights across models, no invalidation, worker UAF, lock-free races.** Keys = FNV-1a(tensor name)⊕eid — identical across models; paired key = blk only. llama-bench multi-model (`-m a,b`), server model swap, same-arch draft → silent wrong logits; role_base can pair model A's gate with model B's up. No invalidate on `llama_free`: queued `ec3_job.src` pointers into unmapped weights → SIGSEGV. `g.cur_*`, `g.redirect`, `g.glu_learn` are written lock-free; two concurrent contexts (legal libllama, test-thread-safety) corrupt planning state. | Mix `src0->data` into plain AND paired keys; add `invalidate(host_base_range)` hook from host-buffer/mmap teardown (drop slots, drain in-range jobs, reset blk-indexed learning arrays) — key change alone is insufficient (address reuse after munmap). Per-sched session handle (or engagement mutex) for `cur_*`; synchronize glu_learn. | 3–5 days |
| B5 [A][P] | **CUDA_CHECK aborts in hot paths + grow-only cache = EC3 manufactures OOMs that kill runs stock survives.** ~13 abort sites in dispatch/flush/collect/redirect (expert-cache.cu:753-791, 945, 989-1007, 1025-1026, 1097-1098); pools take all free-minus-3GB at first decode and are never freed; the CUDA backend's own OOM retry (ggml-cuda.cu:438-445) cannot reclaim EC3 slabs. Anything needing VRAM after pool build (graph pools, second context, batch growth) aborts the process. The insert worker (:292-342) already demonstrates the correct checked-error pattern. | `ec3_disable_device(di)`: mark dead, flush state, begin()→-1 (CPU fallback is already clean); replace hot-path CUDA_CHECKs with checked calls routing there. Shrink: synchronous trim hook called from the backend pool's `cudaErrorMemoryAllocation` retry before the final CUDA_CHECK; whole-pool free as v1 (must drain/version the worker queue first), chunked slabs later. | 1–2 d disable; 2–3 d shrink |
| B6 [P only] | **Upstream AI policy is a hard gate.** CONTRIBUTING.md:11-18: no predominantly AI-generated PRs; edited-after still counts; ban risk. AGENTS.md additionally prohibits AI-written PR descriptions/commit messages/feature requests and autonomous-agent contributions. This branch's provenance and commit history are non-submittable. Precedent: #20757 (two-tier expert cache, PoC 0.5-1→12-14 t/s) closed with no maintainer engagement; niche open as of June 2026. | Human-written feature-request/discussion first, referencing #20757 with the +34%/PPL-parity numbers; ask maintainers (ggerganov: ggml-cpu/ggml-backend; @ggml-org/ggml-cuda) which integration shape they'd accept. Any eventual code must be human-authored/explainable with AI-assist disclosure; commit history cannot be reused. | 1–2 d proposal; weeks re-authorship |

### Majors

| # | Title | Fix | Effort |
|---|---|---|---|
| M1 [N][A][P] | **Pool construction ordering race permanently starves shapes** — the cause of the 754B server loss. One-shot proportional sizing over the partially-discovered pending list; devs 0/2/3 gave 100% to gate/up, then 3,808× "pool for 4704 KB slots skipped (budget 1 MB too small)". The in-code claim "No visit order can lock a device out" is false. | Enumerate all MoE expert shapes up front (or rebalance/rebuild on late shape discovery, synchronized with workers); interim: defer pool build until the shape census is stable for N full decode tokens. Dedupe the skip message. Re-validate 754B server. | 1–2 days |
| M2 [A] | **fit/EC3 composition.** fit's 1 GiB default margin < EC3's 3 GiB reserve → structural starvation; manual `-ncmoe` is -49% on fitting models and the MIN_EXPERT_KB gate cannot undo the placement. | EC3-aware fit planner: never move experts off GPU when they fit; when they don't, fit chooses placement AND explicitly reserves the EC3 budget (pass down via llama_model_params → proc-address setter). Intermediate (days): fit inflates its margin when EC3 requested. Short-term: one-line starvation warning + stop the retry/log loop. | warning: hours; planner: 1–2 wks |
| M3 [A] | **Cold-cache warmup window is net-negative** (397B tg100 -7.2%; 754B first server reply 6.08 t/s). Fill is purely demand-driven; pools are cudaMalloc'd synchronously inside the decode path. Note: the throttle is NOT the cause (capacity-gated, verified throttle=0 during fill); the essential fixes are prefetch + moving pool alloc off the decode path. | Prefetch top-k experts per layer during model load / prompt phase (pinned-staging workers exist); allocate pools at context init or asynchronously. Target near-parity at 100 tokens (full parity may be unreachable by design). | 2–4 days |
| M4 [A][P] | **Auto-budget over-allocates ~5× and the banner lies.** 122B: 19 GB/device of slots for ≤3.6 GB of per-device cacheable data (>81% dead slots given `blk % n_dev` routing); "auto-70%-free" banner vs code taking 100% of free-minus-reserve. Also dev3's undersized pool on 397B caused persistent 87–89% vs 91–92% hit rate. | Cap slots at distinct (tensor, expert) entries mappable to the pool (record n_expert in discovery; paired = (n_tensors/2)×n_expert); fix banner; rebalance across pools. | hours |
| M5 [P] | **No stream ordering between worker slab writes and in-flight reads of an evicted slot** (defer/redirect/fused-gate collect paths return without host sync). Structurally protected at default knobs (64-slot floor + LRU distance), reachable with documented knobs (INSERTS≈64, THROTTLE=1). | Slot serial tagging: refuse to evict slots with serial ≥ last-synced (bump in all sync points incl. ec3_flush, GLU fetch, redirect via cudaEventQuery). Stress test drives INSERTS/THROTTLE, not budget (64-slot floor makes tiny-budget repro impossible). | 1 day |
| M6 [P] | **Scope creep / stock-path divergence:** negative-id memset replaces `assert(i02>=0)` unconditionally (dead code even for EC3 — nothing generates negative ids), THP commit 87f6085a1 (ggml.c, all users/platforms), 54-line package-lock.json churn, redirect-leak on aborted graphs needs the per-graph epoch hook (folded into B1's epoch work). | Revert the sentinel outright; cherry-pick EC3 commits onto clean master; THP as its own PR (re-measure 19.2 t/s baseline without it — it's included in that number). | hours |
| M7 [P] | **Lifecycle: leaked global, detached unjoinable workers, no teardown** of slabs/pinned/streams/events; cudaDeviceReset by an embedder mid-copy is UB; LSan/TSan red. | Stop flag in the CV wait predicate + notify_all + join from an explicit `..._free()` on the reg/device destruction path (not atexit — Windows loader-lock deadlock); tolerate cudaErrorCudartUnloading. Fixes the original hang (destroy-CV-under-waiter UB). | 1 day |
| M8 [P] | **redirect_finalize blind-casts the consumer backend to ggml_backend_cuda_context** — UB/crash with RPC/Vulkan/SYCL consumers (offer gate only excludes CPU; `consumer_backend` is currently GGML_UNUSED). EC3 also silently activates on HIP/MUSA builds (unvalidated). | `ggml_backend_is_cuda()` guard inside ec3_redirect_offer; gate registration to validated platforms. | 2–4 h |
| M9 [P] | **Config/logging/test surface:** 20 env vars, fprintf(stderr) everywhere, SPLITDUMP/SCHEDPROBE probes baked into the scheduler, env-gated selftest (Q4_K/Q6_K, dev0 only — the production IQ2 family is untested anywhere), `v3` in public symbols. | `--expert-cache auto\|on\|off\|<MiB>` + `--expert-cache-reserve` mirroring --fit (common_params, set_env aliases); demote the rest to GGML-internal debug; GGML_LOG_* migration; delete probes; rename `ggml_backend_expert_cache_*`. | 2–3 days |
| M10 [P] | **Integration shape:** mutable extern global in ggml-backend.cpp populated by getenv is a hidden cross-backend channel in the most owner-sensitive files. | `ggml_backend_reg_get_proc_address(reg, "ggml_backend_expert_cache")` discovery + `set_params()`; split into 3 PRs: core cache (≈25% of the win alone), scheduler dst-handoff, fuse/defer/reuse. Note the mul_mat_id/swiglu hook lines remain — extra-buffer-type can't express intra-node row splits. | ~1 wk + review cycles |
| M11 [P] | **CI/tests/docs:** a 220 GB-only-validatable feature is unreviewable. | `tests/test-expert-cache.cpp` (ctest, skip-if-no-CUDA): all supported quants incl. IQ2 family, eviction/throttle/collision/multi-device; tiny-MoE GGUF determinism test in ci/run.sh (must set `LLAMA_EC3_MIN_EXPERT_KB=0` and assert nonzero hits, else it tests nothing); docs page; 1 INFO line telemetry. | ~6 days |
| M12 [A] | **Auto-enable gate + bail-out** — see section 4. | — | ~1 wk |

### Minors

- **Per-tensor MIN_EXPERT_KB gate → arbitrary partial engagement** (122B caches down-only; actually a per-layer/role/device patchwork on dev2). Keep per-tensor gating (a median gate degenerates both ways); add a one-line engagement summary; one measurement that half-engaged ≥ stock. Half day. [A]
- **Merged `ffn_gate_up_exps` misparsed as role=up** (deepseek2/3.2, qwen3.5moe, qwen3next, gemma4). Correct today by four implicit invariants; match `_gate_up_exps` first, distinct role 3. ~1 h. [P]
- **Hardcoded SWIGLU in fused dispatch** — verified safe-by-construction today, but unguarded: record GLU op + swapped flag in learn map/fused state, assert == SWIGLU, comment the cross-file contract; tie glu_learn entries to node identity, not dst address alone. 1–2 h. [P]

---

## 4. Auto-enablement design (mode `auto`)

**Decision tree** (decision + budget at context init in common, where fit already computes placement and per-device margins; pool *construction* stays lazy at first decode — correct, since free VRAM is only known post-warmup):

1. ≥1 genuinely-CUDA device registered (not HIP/MUSA until validated; not RPC/Vulkan consumers — M8).
2. After fit placement, ≥1 `*_exps` tensor is host-resident. **fit owns this choice: experts that fit on GPU stay on GPU and EC3 never engages** (kills the 122B/35B -49% trap). When experts don't fit, fit picks the dense-on-GPU placement and **explicitly carves out the EC3 budget** (reserve = fit margin + measured pool-growth headroom, not constant 3072), printable via `--fit-print`.
3. Predicted win > 0 via **calibrated break-even** replacing the two-datapoint 1 MB constant: host_bw probe (64 MB memcpy, ~10 ms) + 3-shape dispatch microbench (the selftest already measures µs/node, expert-cache.cu:1279; ~100 ms, model-free); require `expert_bytes/host_bw > k × dispatch_us/n_expert_used`, k≈1.5; keep the KB threshold as a floor only. Per-tensor granularity is correct — don't switch to per-model median.
4. Free VRAM at pool build > reserve + min-useful-pool (~512 MB), else stay off with one INFO line — no retry loop, no log spam.

**CLI surface:** `--expert-cache auto|on|off|<MiB>` (default `auto` only after B1–B5 + M1 + bail-out land), `--expert-cache-reserve <MiB>`; `LLAMA_ARG_` env aliases; everything else internal.

**Adaptive bail-out (final safety net, feasible — instrumentation exists):** counterfactual sampling, not t/s comparison (confounded by hit-rate ramp): every Nth hit-bearing node, also run the hit rows on CPU; EWMA of (cpu_time − cache_time); if negative for 256 consecutive hit-bearing nodes after hit rate plateaus (<1%/64-token delta), disable, **free pools** (requires B5's teardown), log once. Cost ~3% on sampled nodes, transient.

**Prerequisites before `auto` can default on:** B1–B5, M1–M4, and the bail-out. Validation sweep across ≥4 MoE models (35B / 122B / 397B / 754B regimes + one gpt-oss-class) demonstrating: never engages where it loses, ≥ stock−noise at tg100, wins where v3 wins today.

---

## 5. Residual risks after all fixes (honest list)

1. **Warmup may never reach full stock parity.** EC3 starts with experts CPU-resident where stock places them statically on GPU; prefetch narrows but cannot guarantee closing the tg100 gap on every model.
2. **The break-even cost model is calibrated, not proven.** It ignores hit rate, PCIe insert traffic, and CPU thread parallelism; the k≈1.5 margin + floor + bail-out are mitigations, but a mispredict on unmeasured hardware (different bw ratios, ROCm if later enabled) means a transient loss until the bail-out fires (~hundreds of tokens).
3. **fit-planner throughput estimates** add a maintenance burden: every new quant/arch shifts the placement-cost model; wrong estimates pick the slower placement silently.
4. **Partial engagement on straddling quants** (122B-style) remains possible by design; each engaged tensor individually clears break-even, but "half-engaged ≥ stock" has one supporting measurement, not a sweep.
5. **Learned-safety still rests on gallocr pointer-identity behavior.** Epochs/serials/op-tags harden it, but a future allocator change could silently disable fuse/defer (fail-safe direction: degrades to the slower correct path).
6. **n_expert_used × batch > 64 bypass:** parallel/batched serving beyond B≈6–8 on high-top-k models gets zero benefit; the +22% B=2 aggregate result does not generalize upward. Document, static_assert.
7. **Shrink-under-pressure interacts with async workers:** the trim hook can fire from an arbitrary allocating thread; the drain/version protocol is the most race-prone new code in the fix set and needs TSan plus the eviction stress test.
8. **HIP/MUSA stay unvalidated** (deliberately gated off); enabling them later is a new measurement campaign, not a flag flip.
9. **Upstream appetite is unproven.** #20757 — nearly this exact feature with a working PoC — was closed without maintainer engagement; even a perfect proposal may stall, and the AI-provenance constraint means the submission must be substantially re-authored by a human who can defend every line.
10. **Single-process scope:** even with session handles, the cache is per-process VRAM; multi-process serving on one GPU gets no coordination and pools from one server starve another (standard for llama.cpp, but worth stating).