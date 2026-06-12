# MoE expert cache

The MoE expert cache adaptively caches the hottest CPU-resident Mixture-of-Experts
expert weights in spare VRAM. When a MoE model is too large for VRAM and its expert
tensors are kept in system RAM (`--cpu-moe`, `--n-cpu-moe`, or the automatic fit),
the CPU computes every expert row at RAM bandwidth — usually the dominant cost of
token generation. The cache moves the frequently-routed experts into otherwise-idle
VRAM and computes those rows on the GPU instead, while the CPU threads compute the
misses in parallel.

It is CUDA-only, on by default, zero-config, and engages only when it can win:
prompt processing, dense models, fully-offloaded models, and models with very small
experts are unaffected.

## Usage

Nothing is required. With a spilling MoE model the cache announces itself during
generation:

```
[moe-cache] decode loop detected, shape census stable — building pools
[moe-cache] dev=0 pool[0]: type=12 slot=28160 KB slots=183 total=5034 MB (paired)
```

Control it with the flag:

```sh
llama-server -m model.gguf --moe-cache 0       # disable
llama-server -m model.gguf --moe-cache 8192    # cap the budget at 8 GiB per device
```

| Setting | Effect |
|---|---|
| (default) | auto: budget = free VRAM minus a safety reserve, decided per device |
| `--moe-cache 0` | off (also restores the default static placement in the fit) |
| `--moe-cache N` | VRAM budget capped at N MiB per device |

Environment variables (equivalent to the flag, plus tuning):

| Variable | Default | Effect |
|---|---|---|
| `GGML_CUDA_MOE_CACHE` | 1 | `0` disables the cache entirely |
| `GGML_CUDA_MOE_CACHE_BUDGET_MB` | auto | per-device VRAM budget cap in MiB |
| `GGML_CUDA_MOE_CACHE_RESERVE_MB` | 3072 | VRAM left untouched per device (the CUDA graph pool grows after init) |
| `GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB` | 256 | skip models whose experts are too small to amortize dispatch |
| `GGML_CUDA_MOE_CACHE_HOTSET` | 1 | persist the hot expert set across runs (`~/.cache/llama.cpp/`) |
| `GGML_CUDA_MOE_CACHE_STATS` | 0 | log hit-rate/timing stats every N MUL_MAT_ID nodes |

## How it engages (and disengages)

The cache is conservative by design; every gate below must pass before a single
byte of VRAM is spent:

1. **Decode only.** Filling happens exclusively from single-token generation
   misses. Prompt processing never touches the cache (caching prompt-phase
   routing measurably thrashes it).
2. **Expert size and type.** Experts must be at least 256 KiB each and use a
   quantization type with a batched matvec kernel (Q2_K..Q6_K, Q4_0..Q8_0,
   MXFP4, IQ1_S..IQ4_XS).
3. **Stable shape census.** Pools are sized only after the set of expert shapes
   has been stable for a window of visits, so mixed-quant models get correctly
   proportioned budgets (per role group: gate+up pairs vs down projections).
4. **Measured bail-out.** The first ~2 750 eligible nodes run on the pure CPU
   path to sample a baseline. If the cache-engaged node time ever sustains above
   that baseline, the cache trips, frees all of its VRAM, and stays off for the
   run — a wrong placement bet costs a few seconds, not the session.
5. **VRAM pressure.** If the CUDA allocator hits OOM, the cache surrenders its
   memory (`trim`) before the allocation is retried. CUDA errors inside the
   cache disable it on that device; the CPU path always remains correct.

With `--fit` (automatic layer placement), an additional placement rule applies:
when the model spills heavily (model bytes ≥ 1.8× usable VRAM), the fit prefers
keeping **all** experts in RAM — leaving maximum VRAM to this cache — over
statically placing a few layers. On heavily-spilling models this combination
beats static placement; on models near the fit boundary the rule keeps the
stock behavior.

## Design

Integration happens inside the CPU `mul_mat_id` kernel, not the graph scheduler:

- **Plan/dispatch/collect.** Thread 0 looks up each routed expert id in the
  per-device slot pools, launches ONE batched matvec over all hit rows, and the
  remaining threads compute the miss rows concurrently. Results merge into the
  node's dst before the node completes, so correctness holds under any split
  topology and outputs are bit-identical to the pure CPU path.
- **Slot pools.** Per (expert-size, type) pools whose slot stride equals the
  source tensor's `nb[2]`, so the batched kernel indexes the pool like a regular
  expert tensor. Plain LRU with an admission throttle at capacity.
- **Paired pools + fused dispatch.** Gate and up weights of one expert live in
  twin slabs under a single cache entry; for layers whose gate→up→GLU wiring has
  been observed, one fused kernel computes `silu(gate)·up` on-chip, halving
  launches and skipping those rows in the CPU GLU kernel.
- **Async fills.** Misses are enqueued to pinned-staging worker threads; idle
  workers prefetch the rest of the expert space in the background, warm-started
  from the previous session's persisted hot set.
- **GPU-resident handoff.** For down-projection nodes whose consumer is a GPU
  split, hit rows relay through a pinned image directly into the consumer's
  input copy (stream-ordered), skipping the host round trip.
- **Multi-model safety.** Cache keys mix the weight tensor's data pointer, and
  host-buffer teardown invalidates in-flight fills, so several models in one
  process never alias.

## Measured results

4× RTX 3090 (96 GB VRAM), EPYC 7R13 48c, 8-channel DDR4. `llama-bench`, tg300,
same thread count both arms. *auto* = default zero-config run vs vanilla
auto-fit; *forced* = all experts on CPU (`-ngl 99 -ncmoe 99`), isolating the
cache's contribution; smaller models forced to emulate VRAM-starved hosts.

| Model | Regime | t/s cache | t/s vanilla | Gain |
|---|---|---|---|---|
| GLM-5.1 754B IQ2_M | auto | 17.49 | 13.96 | +25% |
| GLM-5.1 754B IQ2_M | forced | 18.32 | 11.71 | +56% |
| Qwen3.5 397B Q3_K_XL | auto | 30.25 | 28.18 | +7% |
| Qwen3.5 397B Q3_K_XL | forced | 30.21 | 21.49 | +41% |
| Qwen3.5 122B Q2_K_XL | auto | 74.98 | 74.98 | parity (dormant) |
| Qwen3.5 122B Q2_K_XL | forced | 49.82 | 44.49 | +12% |
| MiniMax-M2.7 IQ2_XXS | forced | 44.77 | 34.60 | +29% |
| gpt-oss-120b F16 | forced | 58.59 | 45.53 | +29% |
| Llama-4-Scout 109B Q4_K | forced | 36.35 | 25.85 | +41% |
| granite-4.0-h-small Q4_K | forced | 57.73 | 36.87 | +57% |
| Qwen3 30B-A3B Q4_K_XL | forced | 93.44 | 69.22 | +35% |
| ERNIE-4.5 21B Q4_K | forced | 112.00 | 79.45 | +41% |
| DeepSeek-V2-Lite Q4_K | forced | 76.67 | 61.85 | +24% |
| gpt-oss-20b F16 | forced | 87.02 | 67.36 | +29% |
| OLMoE 7B Q3_K | forced | 211.91 | 209.26 | +1% |

Temperature-0 outputs are identical with the cache on and off across all nine
architectures tested (glm-dsa, qwen3moe/qwen35moe, ernie4.5-moe, gpt-oss, olmoe,
llama4, granitehybrid, deepseek2).

## Diagnostics

`GGML_CUDA_MOE_CACHE_STATS=2000` logs every 2000 nodes:

```
[moe-cache] dev=0 hits=41873/52480 (79.8%) inserts=9023 evict=412 ... used=2812/2816
[moe-cache] dev=0 timing: nodes=13120 plan=8.2us disp=21.4us coll=33.0us per-node total=62.6us
```

`GGML_CUDA_MOE_CACHE_SELFTEST=1` runs a model-free numerical self-test of the
batched dispatch path at backend registration. `GGML_CUDA_MOE_CACHE_DEBUG=1`
traces the first API calls for stall diagnosis.
