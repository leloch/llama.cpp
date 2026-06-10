# Expert Cache v3 (EC3) — dynamic VRAM cache for CPU-resident MoE experts

For MoE models whose routed experts do not fit in VRAM (e.g. GLM-5.1 754B
UD-IQ2_M at 220 GB on 96 GB of GPUs), llama.cpp keeps the experts in host RAM
and the CPU computes them — the dominant cost of token generation. EC3 keeps
the *hot* experts in spare VRAM and computes them on the GPU concurrently with
the CPU computing the rest.

## Measured results (GLM-5.1 754B UD-IQ2_M, 4x RTX 3090, EPYC 7R13, 8ch DDR4)

| Config | tg t/s |
|---|---|
| stock `--fit` (best vanilla placement)        | 14.0 |
| all experts on CPU, no cache (`-ncmoe 99`)    | 12.5 |
| **EC3 final stack (tg300, r=3)**              | **19.2 ± 1.0 (+34% vs stock)** |
| EC3, 500-token horizon                        | stable (hit rate still rising at 77%) |
| destructive-mask ceiling (GPU rows were free) | ~24 |

`llama-server` end-to-end: **15.5–16.2 t/s sustained** across domain-switching
prompts (was 14.8–15.0 before the optimization phase).

Real-world `llama-server` (16K ctx, sampling): **~14.8–15.0 t/s sustained**,
measured across consecutive prompts switching domains (code → cooking →
physics → code) with **no slowdown at topic switches** — the cache refills in
the background (first-ever response ~13 t/s during the initial cold fill).

Correctness: perplexity with forced single-token decode matches cache-off to
0.4% (inside the measurement error bar), with the full optimization stack on.
Cached rows run the same mmvq kernels used for GPU-resident experts.

## Usage

```bash
LLAMA_EC3=1 ./llama-server -m <model> -ngl 99 -ncmoe 99 -fa on -t <n_cores> ...
```

VRAM is sized automatically (free memory minus a 3 GB/device reserve, split
across per-shape pools, allocated on demand at the first decoded tokens).
Everything else is optional tuning:

| Env | Default | Meaning |
|---|---|---|
| `LLAMA_EC3` | off | `1` enables the cache |
| `LLAMA_EC3_BUDGET_MB` | auto | per-device cache budget cap |
| `LLAMA_EC3_RESERVE_MB` | 3072 | VRAM never touched (the CUDA pool grows lazily after init; stealing it crashes decode) |
| `LLAMA_EC3_NDEV` | all | number of CUDA devices to stripe over |
| `LLAMA_EC3_INSERTS` | 8 | max insert enqueues per node visit |
| `LLAMA_EC3_WORKERS` | 4 | background copy threads (pinned staging + own streams) |
| `LLAMA_EC3_MIN_EXPERT_KB` | 1024 | skip models with experts smaller than this (too little CPU work to amortize dispatch — e.g. 35B-A3B class) |
| `LLAMA_EC3_DEFER` | on | defer a gate node's sync into the same layer's up node |
| `LLAMA_EC3_REUSE` | on | reuse the quantized activation between gate and up |
| `LLAMA_EC3_FUSE` | on | fused gate+up+SwiGLU GPU dispatch (paired pools; engages per layer after the GLU wiring is observed — GLM yes, Qwen3.6 no) |
| `LLAMA_EC3_REDIRECT` | on | down-projection dst handoff: rows relayed via pinned image to the consumer's stream, no host syncs |
| `LLAMA_EC3_THROTTLE` | 8 | at-capacity admission: admit 1-in-N misses |
| `LLAMA_EC3_STRIPE` | off | role-striping probe (disables defer/reuse/fuse) |
| `LLAMA_EC3_STATS` | off | print hit/timing stats every N hit-bearing nodes |
| `LLAMA_EC3_DEBUG` | off | trace cache events (value = max events) |
| `LLAMA_EC3_SELFTEST` | off | model-free numeric self-test + latency micro-bench at startup |

## Design

- **In-kernel integration.** The CPU `mul_mat_id` kernel itself plans hits and
  misses: thread 0 issues ONE batched mmvq launch for the cached rows while the
  remaining threadpool threads compute the uncached rows; results are written
  into dst before the node completes. Correct under any scheduler split
  topology; no shared tensor is ever mutated.
- **Decode-only, demand-driven fill.** Only single-token (generation) visits
  touch the cache. Prompt processing keeps its existing op-offload path —
  prompt-driven fills measurably pollute the cache (v2's core mistake).
- **Asynchronous inserts.** Misses enqueue a copy job; worker threads stage
  through pinned buffers onto dedicated copy streams. The decode loop never
  waits for a fill. At full capacity an admission throttle (1-in-8 misses)
  keeps eviction churn from stealing host RAM bandwidth from the CPU matmuls.
- **Exact-stride slot pools, built on demand.** Slots live in per-(expert
  size, quant type) pools whose stride equals the source tensor's nb[2], so
  the batched MMID kernel indexes a pool exactly like a normal expert tensor.
  Pools allocate lazily per device as shapes appear (no visit order can lock a
  device out), each capped at half the available budget so the second shape
  always fits.
- **Plain LRU + admission throttle.** Measured: eviction sophistication is not
  the binding constraint; capacity and admission are.
- **Micro-optimizations** (measured on, validated by perplexity): gate→up
  shared activation quantization; gate-sync deferral into the up node
  (per-layer safety learned by observing the node sequence on the first
  decoded token). CUDA-graph capture of the dispatch chain was tried and
  measured to be a net loss (the chain is GPU-exec-bound, not launch-bound).

## When it helps

The win scales with per-expert CPU work. Experts of ~3 MB+ (huge models at
low bpw) gain ~25-30%; models with sub-MB experts (e.g. 35B-A3B at Q4) lose
more to dispatch overhead than the GPU saves, which is why `MIN_EXPERT_KB`
gates them off by default.

## Future work (identified, not implemented)

- Keep GPU-computed rows on-device and hand them to the next split's input
  copy directly (saves the D2H + re-H2D round trip for the down projection;
  requires scheduler cooperation).
- The remaining per-node cost (~50-60 us serial on thread 0) is dominated by
  the GPU chain's execution latency; shrinking it further needs kernel fusion
  of quantize+mmv or persistent-kernel approaches.
