# CUDA MoE expert cache

The CUDA MoE expert cache accelerates decode when routed expert weights remain in host memory. A cache hit runs the selected expert matvec on CUDA while the CPU computes the miss rows through the normal `MUL_MAT_ID` kernel. The cache belongs to one backend scheduler and persists until that scheduler is destroyed.

This is an opportunistic path. Unsupported nodes, unavailable cache capacity, contention, and cache failures fall back to CPU execution.

## Configuration

Use `--moe-cache MODE` with programs that use the common argument parser:

| Mode | Cache budget | Weight repacking | Device requirements |
| --- | --- | --- | --- |
| `auto` | Free VRAM minus the reserve | Preserved | At least two eligible selected CUDA devices, compute capability 7.5 or newer |
| `on` | Free VRAM minus the reserve | Disabled | At least one eligible selected CUDA device, compute capability 7.0 or newer |
| `N` | At most `N` MiB per device, after the reserve | Disabled | Same as `on` |
| `off` or `0` | No cache session | Preserved unless changed separately | None |

`auto` is the default. Preserving repacking means that `auto` may remain dormant when a repacked CPU kernel consumes the expert operation before the regular CPU `MUL_MAT_ID` path sees it. Use `on` or a positive budget to request the canonical CPU weights required by the cache.

The cache only sees experts already assigned to CPU memory. Use `--cpu-moe`, `--n-cpu-moe`, or a tensor override when that placement is required. The normal fit logic is not cache-aware and may place some experts statically in VRAM. A fit target below the cache reserve can also leave no runtime cache budget; for example, the usual 1024 MiB fit margin is smaller than the default 3072 MiB cache reserve.

The cache considers only CUDA backends selected for the scheduler. It does not discover or use an unselected device.

## Eligibility and allocation

With default settings, a node must satisfy all of these conditions:

- The operation is the regular CPU `MUL_MAT_ID` path with F32 activations.
- The weight tensor name contains `_exps`.
- The weight type is supported by the CUDA quantized matvec kernel.
- One expert is at least 1 MiB.
- The graph node contains one token and no more than 64 routed rows.
- The selected device can hold a pool of at least 64 experts of that shape.

Each device budget is measured once, at first eligible use. The automatic budget is:

```
min(configured budget, free VRAM - 3072 MiB reserve)
```

The configured-budget term is omitted for `auto` and `on`. A fixed `N` is a cap for cache slabs and device-side dispatch scratch together, not a guaranteed slab allocation. Pool allocation may be smaller after reserving scratch or retrying an allocation, and no pool is created when fewer than 64 slots fit.

Tensor shapes are collected before pools are allocated. Allocation waits for a repeated shape census and 64 stable visits so early graph discovery does not give all capacity to the first tensor shape. Capacity is divided among discovered shapes. A layer is assigned once to a selected device and keeps that assignment while the device remains usable. If that device cannot host a different tensor shape from the layer, the tensor receives its own stable assignment. Initial assignments are deterministic and weighted by usable slab capacity after existing pools and dispatch scratch are accounted for.

The `[moe-cache] enabled` message is printed only after the first pool is allocated. If it is absent, the cache did not become active.

## Demand fill and eviction

The cache never transfers a missing expert synchronously for the current matvec. The miss stays on the CPU, and a bounded background request may populate the expert for a later use.

By default:

- An expert is admitted after its second observed miss.
- At most 8 fills are enqueued by one node.
- A device queue is limited to 128 jobs and 512 MiB.
- Each device has one low-priority CUDA fill stream.
- Host-to-device fills are serialized across devices in one session.
- Full pools use LRU eviction. After a successful fill, that expert needs eight fresh misses before it can replace another entry.

The default maximum node batch is one token, so nodes with more than one token, including normal prompt-processing nodes, do not populate or use the cache.

## Concurrency, fallback, and lifetime

The scheduler binds its cache session only while it computes a graph. Within one session, per-device scratch buffers and dispatch are serialized. If another cached node in that session already owns a device, the contending node takes the regular CPU path. Separate schedulers own separate sessions, pools, streams, and dispatch locks even when they select the same physical device.

Valid slots are pinned for the lifetime of a node. Miss rows remain in the normal CPU work set. Hit rows are removed from that set only after the complete GPU dispatch has been accepted.

If dispatch fails, every planned hit row is restored to the CPU work set before worker threads begin. If collection fails, every skipped row is recomputed with the stock CPU helper. A fatal CUDA cache error disables and trims the affected device, after which nodes continue on CPU.

CUDA output can differ slightly from CPU output because the hit path uses CUDA activation quantization and matvec arithmetic. Do not expect bit-identical logits or token streams. In particular, a small rounding difference can change a near-tie greedy token.

Public writes to a cached host weight buffer invalidate the affected byte range before the write begins. Invalidation cancels overlapping queued fills, waits for overlapping active reads and transfers, and removes overlapping slots and demand records. The same process runs before a host allocation is released or stops being a weight buffer. Callers must still obey the normal backend synchronization rules when mutating a weight used by concurrent graph execution. Scheduler teardown stops admission, cancels queued work, waits for active graph scopes and nodes, joins fill workers, and then frees device storage.

The normal CUDA allocator may trim an active expert cache as a last attempt to satisfy an allocation. Trimming frees all cache storage on that device and leaves it disabled for the rest of the session. A session becomes permanently dormant when no nonzero device budget remains, or when `auto` drops below two devices with nonzero budgets; any remaining cache devices are then trimmed as well.

## Diagnostics and developer controls

The pool log reports the physical CUDA device, weight type, expert size, slot count, and allocated bytes. Session teardown reports hits, misses, queue activity, fills, evictions, and fallback counters for devices that processed cache nodes. Set `GGML_CUDA_MOE_CACHE_STATS=N` to print the same counters every `N` collection calls.

The following environment variables are implementation controls, not a stable command-line interface. They are read when a scheduler creates its cache session.

| Variable | Default | Meaning |
| --- | ---: | --- |
| `GGML_CUDA_MOE_CACHE_RESERVE_MB` | `3072` | VRAM left outside the cache on each device |
| `GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB` | `1024` | Minimum bytes per expert, in KiB |
| `GGML_CUDA_MOE_CACHE_MAX_BATCH` | `1` | Maximum tokens in an eligible node |
| `GGML_CUDA_MOE_CACHE_INSERTS` | `8` | Maximum admissions per node |
| `GGML_CUDA_MOE_CACHE_ADMIT_AFTER` | `2` | Miss count required before admission |
| `GGML_CUDA_MOE_CACHE_THROTTLE` | `8` | Fresh misses required before replacing a full-pool entry |
| `GGML_CUDA_MOE_CACHE_QUEUE` | `128` | Maximum queued jobs per device |
| `GGML_CUDA_MOE_CACHE_QUEUE_MB` | `512` | Maximum queued source bytes per device |
| `GGML_CUDA_MOE_CACHE_STATS` | `0` | Collection-call interval for periodic statistics, or `0` for teardown only |
| `GGML_CUDA_MOE_CACHE_NDEV` | all | Maximum selected CUDA devices used by a session |
| `GGML_CUDA_MOE_CACHE_SERIAL_FILL` | `1` | Serialize fills across devices in a session |
| `GGML_CUDA_MOE_CACHE_MIN_CC` | mode dependent | Override the minimum compute capability encoded as `major * 100 + minor * 10` |

Directly setting `GGML_CUDA_MOE_CACHE_MODE`, `GGML_CUDA_MOE_CACHE`, or `GGML_CUDA_MOE_CACHE_BUDGET_MB` controls the backend session but does not change the model loader's repacking choice. Prefer `--moe-cache` when using a llama.cpp program.

Keep `GGML_OP_OFFLOAD_MIN_BATCH` above the decode batch size. Setting it to `1` can make the scheduler offload the complete `MUL_MAT_ID` operation to CUDA before the CPU path can split cache hits from misses.

`GGML_CUDA_MOE_CACHE_FAIL` is for fallback testing only. It accepts `dispatch`, `collect`, `insert`, or `slab`; comma-separated stages and `all` are also accepted. A CUDA build can exercise the synthetic success and failure paths with:

```sh
CUDA_VISIBLE_DEVICES=0 ./build/bin/test-moe-cache
```

## Benchmarking

Cache measurements need decode warmup. Pool creation waits for graph-shape discovery, expert admission needs repeated demand, and queued fills finish asynchronously. A one-token warmup usually measures a cold cache rather than steady state.

For an operational comparison of the default policy, keep all placement, thread, batch, and context arguments identical and vary only `off` versus `auto`:

```sh
CUDA_VISIBLE_DEVICES=0,1,2 ./build/bin/llama-bench \
    -m /path/to/model.gguf -p 0 -n 300 --n-gen-warmup 256 -r 5 \
    -ngl 99 -ncmoe MODEL_MOE_LAYER_COUNT -t CPU_THREAD_COUNT -fa on \
    --moe-cache off,auto -o json
```

Replace the two uppercase placeholders with integers appropriate for the model and host. Check the logs to confirm that all dense layers are on CUDA, routed experts are on CPU, and the `auto` arm actually allocated pools. Record GPU models, PCIe topology, CPU model, memory channels and speed, model quantization, exact placement, build revision, and all environment overrides.

For an isolated comparison with the same canonical CPU weights in both arms, use a fixed cache budget and explicitly disable repacking:

```sh
CUDA_VISIBLE_DEVICES=0,1,2 ./build/bin/llama-bench \
    -m /path/to/model.gguf -p 0 -n 300 --n-gen-warmup 256 -r 5 \
    -ngl 99 -ncmoe MODEL_MOE_LAYER_COUNT -t CPU_THREAD_COUNT -fa on \
    --moe-cache off,4096 --repack off -o json
```

Adjust `4096` to the intended per-device MiB cap. `llama-bench` records the effective repack setting and rejects repacking with cache `on` or a fixed budget.

Use a long enough timed generation to amortize graph discovery and inspect both throughput variation and the final cache counters. Repeat the process after a cold process start when cold-start behavior matters. Do not infer a gain from hit rate alone: activation transfers, result transfers, fill traffic, GPU speed, PCIe speed, and CPU memory bandwidth all affect the result.

An `off` versus `on` comparison without `--repack off` includes the intended repacking-policy change. Treat results that use different repacking or expert placement as an end-to-end configuration comparison, not an isolated measurement of the cache.

## Current limitations

- CUDA only. HIP, MUSA, Metal, Vulkan, and other backends do not register an implementation.
- CPU-resident expert `MUL_MAT_ID` only. There is no fused gate/up/activation path and no GPU-resident output handoff.
- Demand fill only. There is no predictive prefetch or separate prompt-time population path.
- No hot-set file or persistence across scheduler sessions or process restarts.
- Direct writes through a raw host pointer bypass invalidation. Mutate cached weight buffers through the backend tensor and buffer APIs.
- No cache-aware automatic expert placement.
- No runtime performance bail-out. An eligible but unprofitable cache remains active unless it fails, is trimmed, or is disabled by configuration.
- Every scheduler owns an independent cache. Multiple contexts can therefore reserve separate VRAM pools.
- Generic operation offload can bypass the cache if `GGML_OP_OFFLOAD_MIN_BATCH` is set low enough to offload decode nodes.
- Performance depends strongly on model shape, quantization, CPU memory bandwidth, PCIe link, CUDA device, spare VRAM, and routing locality.
