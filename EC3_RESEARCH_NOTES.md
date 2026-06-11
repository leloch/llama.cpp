# EC3 optimization-idea validation (mini-prototypes & experiments, 2026-06-11)

Every idea from the design review measured with a dedicated experiment before any
build decision. "Contam." = run with background downloads (ratios valid, absolutes low).

## Verdict table

| Idea | Experiment | Measured | Verdict |
|---|---|---|---|
| **Cache-aware routing bias** | ORACLE ceiling +16.1%; then FULL v1 PROTOTYPE BUILT (additive selection bias, DSv3 pattern: EC3 residency bitmaps -> per-layer graph input -> add before top-k, LLAMA_EC3_RBIAS) | Strength sweep on 122B@2.5G budget: PPL 4.10 (off) -> 4.27 (0.02) -> 7.46 (0.05) -> 22.7 (0.10); speed 50.2 -> 45.5-45.8 (host-side bias input adds sched copies/splits) | **v1 REFUTED both axes.** Ceiling real but uniform additive bias can't harvest it: prob domain too tight (quality) + input plumbing costs more than it returns (speed). Salvageable only via gap-conditioned top-k custom op + GPU-resident bias — unproven, parked. Knob ships default-off |
| **Tiered residency (static-hot)** | per-slot popularity counters | **top10% of experts carry 80-81% of hits, top30% = 95-96%** (122B; 754B pending clean run) | Concentration is extreme, but the prize (chain overhead on hot hits) overlaps striping+bias upside (~3ms pool). Loader surgery (per-expert tensor split) heavy. **DEFER behind bias; revisit if bias lands well.** Cheap subset worth doing: persist hot-set for instant warm start |
| **Expert striping (multi-GPU per-layer parallelism)** | selftest n-sweep at 754B dims | chain = **33us fixed + 10us/row** (n=1:40, n=2:62, n=4:74, n=8:110); 8-row chain striped to 4 devs ~53us vs 110 — could fully hide the fused chain under CPU work (stick-out 3.3ms/token) | Revised **+3-5%** (launch serialization eats part). Borderline; behind bias and kernel work |
| **Reserve shrink 3GB->1.5GB** | A/B on 754B | 17.31 vs ~17.6 default = parity (curve's flat tail) | **CLOSED — keep 3GB** |
| **Per-row down-chaining** | stick-out probe | down stick-out 1.0ms/token, ~50% of chains already hidden | **Stays parked** (neutral-class); revisit under MTP batching |
| **CPU kernel (DRAM saturation)** | triad/read microbench + in-vivo BW from bail-EWMA | Ceiling: **141 GB/s** read (48T). In-vivo: Q2_K-family **97 GB/s (69%)**, IQ2-family **77 GB/s (55%)** — IQ decode is ALU-bound, NOT DRAM-saturated (user's hunch confirmed; old 95% claim was wrong for IQ quants) | **Real headroom on IQ-quant models (the 754B!): +5-10% via IQ2_XXS/IQ3_XXS vec_dot optimization.** ik_llama end-to-end NOT the shortcut (8.3/7.9 vs our 12.2 same config — mainline outpaced it); port its dequant *techniques* instead. 2-4 days, kernel-level work |
| **MTP / spec decode** | clean(er) 754B ub A/B at t=80 | EC3: ub1 20.38, ub4 15.81 (V=5.2x); vanilla ub4 15.69 — **at batch 4 the cache contributes ~0** and verify cost 5.2x vs the cache-accelerated single-token base. alpha=.85/N=3 yields 3.2 tok/step < 5.2 break-even | **NOT currently positive on 754B — parked.** Root cause identified: the batch path uploads+quantizes the activation per HIT ROW (<=32x) instead of per distinct TOKEN (4x) — 8x waste; plus CPU naturally amortizes expert reads across batch rows, eroding the cache's edge. Prerequisite for MTP: batch act dedup + per-hit y-channel mapping in the mmv wrapper |

## Build outcomes (the implementation round)
1. Routing bias v1: BUILT and REFUTED (see row above). Knob ships default-off.
2. IQ2 kernel headroom: harvested for FREE via SMT — **-t 80 = 19.85 t/s under
   download contamination (beats the clean 19.4 at -t 48)**; t=96 collapses
   (6.0), peak plateau 80-88. Launcher updated. Quant-dependent: BW-bound
   (K-quant) models should stay at physical cores.
3. MTP: parked with measured break-even analysis (above); batch act-dedup is
   the named prerequisite.
4. Hot-set persistence: built (see below).
5. Striping/tiering: parked; residual stick-out shrinks further at t=80
   (more CPU threads hide more chain time).

## Instrumentation added (permanent)
Per-slot popularity counters + concentration stats; chain (n_hits, wall) buckets;
bail-EWMA dump (in-vivo CPU BW per model); stick-out counters; LLAMA_EC3_ORACLE
research knob (wrong outputs by design); LLAMA_EC3_NSWEEP selftest sweep.
