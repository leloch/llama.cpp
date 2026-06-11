# EC3 optimization-idea validation (mini-prototypes & experiments, 2026-06-11)

Every idea from the design review measured with a dedicated experiment before any
build decision. "Contam." = run with background downloads (ratios valid, absolutes low).

## Verdict table

| Idea | Experiment | Measured | Verdict |
|---|---|---|---|
| **Cache-aware routing bias** | ORACLE knob (every miss served by a resident expert = 100% effective hit, outputs intentionally wrong, speed-only) | 754B 16.48 -> 19.13 same-conditions = **+16.1% ceiling**; budget curve RISING on 754B (6G=14.4, 12G=16.2, 19G=17.3) unlike flat 122B; quality bound: dropping weakest 1/2 experts = +2.1/+3.7% PPL (bias swaps are gentler) | **BUILD-WORTHY #1 (non-MTP).** Conservative near-tie swaps ~+8% expected on big-spill models. Needs router-logit bias before top-k + score-gap threshold + PPL gate |
| **Tiered residency (static-hot)** | per-slot popularity counters | **top10% of experts carry 80-81% of hits, top30% = 95-96%** (122B; 754B pending clean run) | Concentration is extreme, but the prize (chain overhead on hot hits) overlaps striping+bias upside (~3ms pool). Loader surgery (per-expert tensor split) heavy. **DEFER behind bias; revisit if bias lands well.** Cheap subset worth doing: persist hot-set for instant warm start |
| **Expert striping (multi-GPU per-layer parallelism)** | selftest n-sweep at 754B dims | chain = **33us fixed + 10us/row** (n=1:40, n=2:62, n=4:74, n=8:110); 8-row chain striped to 4 devs ~53us vs 110 — could fully hide the fused chain under CPU work (stick-out 3.3ms/token) | Revised **+3-5%** (launch serialization eats part). Borderline; behind bias and kernel work |
| **Reserve shrink 3GB->1.5GB** | A/B on 754B | 17.31 vs ~17.6 default = parity (curve's flat tail) | **CLOSED — keep 3GB** |
| **Per-row down-chaining** | stick-out probe | down stick-out 1.0ms/token, ~50% of chains already hidden | **Stays parked** (neutral-class); revisit under MTP batching |
| **CPU kernel (DRAM saturation)** | triad/read microbench + in-vivo BW from bail-EWMA | Ceiling: **141 GB/s** read (48T). In-vivo: Q2_K-family **97 GB/s (69%)**, IQ2-family **77 GB/s (55%)** — IQ decode is ALU-bound, NOT DRAM-saturated (user's hunch confirmed; old 95% claim was wrong for IQ quants) | **Real headroom on IQ-quant models (the 754B!): +5-10% via IQ2_XXS/IQ3_XXS vec_dot optimization.** ik_llama end-to-end NOT the shortcut (8.3/7.9 vs our 12.2 same config — mainline outpaced it); port its dequant *techniques* instead. 2-4 days, kernel-level work |
| **MTP / spec decode** | seq-batch ub-sweep | 122B V_seq(4)=2.4-3.1x; 754B showed over-linear anomaly under contamination (vanilla control V(4)=3.0 vs our 5.2 — different contamination, inconclusive) | **Still candidate #1 overall.** Open action: clean 754B ub-sweep A/B (our batch path may have a 754B-specific perf bug) |

## Recommended build order
1. **Routing bias** (conservative, score-gap-gated, PPL-validated) — biggest measured ceiling, contained surgery.
2. **IQ2-family CPU kernel optimization** — attacks the miss path that bias doesn't remove.
3. **MTP enablement** — multiplies everything; resolve the 754B batch anomaly first.
4. Striping / tiering — only if 1-3 land and the residual stick-out still matters.

## Instrumentation added (permanent)
Per-slot popularity counters + concentration stats; chain (n_hits, wall) buckets;
bail-EWMA dump (in-vivo CPU BW per model); stick-out counters; LLAMA_EC3_ORACLE
research knob (wrong outputs by design); LLAMA_EC3_NSWEEP selftest sweep.
