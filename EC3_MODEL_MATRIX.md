# EC3 model-matrix verification (2026-06-12, 4x RTX 3090 + EPYC 7R13)

Build: branch v3-expert-cache @ 83918dbee+. "Vanilla" = same binary, LLAMA_EC3=0
(stock fit placement, zero cache code active). All temp-0 identity checks compare
50-60 greedy tokens byte-for-byte.

## Auto mode (zero config, `-fitt 1024`)

| Model | Arch | Size | Setup | EC3 auto | Vanilla | Verdict |
|---|---|---|---|---|---|---|
| GLM-5.1 754B IQ2_M | glm-dsa | 220G | 4 GPU | **17.0** | 14.2 | **+20% (dynamic placement)** |
| Qwen3.5-397B Q3_K_XL | qwen35moe | 167G | 4 GPU | **30.9** | 28.2 | **+10% (dynamic)** |
| Qwen3.5-122B Q2_K_XL | qwen35moe | 39G | 4 GPU | 74.8 | 74.7 | dormant (fits) |
| Qwen3.5-122B | | 39G | 2 GPU | 80.1 | 80.1 | dormant (fits) |
| Qwen3.5-122B | | 39G | 1 GPU (spill) | 57.5 | 56.8 | static chosen (economics rule) |
| MiniMax-M2.7 230B-A10B IQ2_XXS | minimax-m2 | 61G | 4 GPU | 84.6 | 84.6 | dormant (fits) |
| MiniMax-M2.7 | | 61G | 1 GPU (spill) | 45.0 | 45.1 | static chosen (economics rule) |
| Qwen3.6-35B-A3B Q4_K_XL | qwen35moe | 21G | 4 GPU | 148 | 146 | dormant |
| Qwen3-30B-A3B Q4_K_XL | qwen3moe | 17G | 4 GPU | 204.5 | 205.2 | dormant |
| ERNIE-4.5-21B-A3B Q4_K_M | ernie4_5-moe | 12G | 4 GPU | 228.3 | 219.4 | dormant |
| gpt-oss-120b F16 | gpt-oss | 61G | 4 GPU | 121.8 | 121.8 | dormant |
| Llama-4-Scout Q4_K_XL | llama4 | 58G | 4 GPU | 58.33 | 58.32 | dormant |

## Cache contribution WITHIN the offload regime
(`-ngl 99 -ncmoe 99` both arms, cache on vs off — isolates the cache mechanism;
NOT a vs-best-vanilla claim. The vs-best-vanilla numbers are the Auto section
above and the 4-arm forced-spill table below, where vanilla autofit's static
placement is an explicit arm.)

| Model | EC3 | Vanilla | Delta | temp-0 identity |
|---|---|---|---|---|
| GLM-5.1 754B | **19.4±1.1** | 14.0* | **+34-39%** | PPL-validated |
| Qwen3.5-397B | **33.3** | 28.2* | **+18%** | PPL-validated |
| Qwen3.5-122B | **51.3** | 43.5 | **+17.8%** | IDENTICAL |
| Qwen3-30B-A3B | **90.6** | 67.4 | **+34.4%** | IDENTICAL |
| ERNIE-4.5-21B | **111.6** | 77.4 | **+44.3%** | IDENTICAL |
| Qwen3.6-35B-A3B | **89.0** | 77.3 | **+15%** | PPL-validated |
| gpt-oss-120b MXFP4/F16 | **60.7** | 44.1 | **+37.8%** | IDENTICAL (fuse refused on SWIGLU_OAI) |
| gpt-oss-20b | **88.3** | 65.3 | **+35.3%** | IDENTICAL |
| OLMoE-7B-A1B Q3_K_M | **231.4** | 204.2 | **+13.3%** | IDENTICAL (tiny-expert extreme) |
| Llama-4-Scout 109B Q4_K_XL | **37.1** | 25.8 | **+43.6%** | IDENTICAL (top-1 routing, 28MB experts) |

*vanilla figure = best stock config (fitt), stricter than same-placement.

### Forced-spill on ONE GPU (inflated fit margin; 4-arm)

| Model (usable VRAM) | Auto (rule choice) | Vanilla static | Manual dyn + EC3 | Manual dyn vanilla | Rule verdict |
|---|---|---|---|---|---|
| Qwen3-30B (8G) | 101.8 (static) | 103.1 | 90.7 | 69.0 | correct: static > best dynamic |
| ERNIE-21B (6G) | **110.0 (dynamic)** | 108.8 | 97.8 | 78.1 | correct: big-expert exception wins |
| gpt-oss-120b (22G) | 55.5 (dynamic) | 56.2 | — | — | parity (-1.2%, tolerated) |
| gpt-oss-20b (5G) | **88.9 (dynamic)** | 87.8 | — | — | correct: dynamic wins |

### gpt-oss-120b adversarial correctness (forced, 4 GPU)
SWIGLU_OAI + expert biases: fuse learning correctly refused (fused-layers=0),
MXFP4 4.3MB experts cached, **+37.8% (60.7 vs 44.1)**, temp-0 IDENTICAL.
Dormant at 4-GPU auto: 121.84 vs 121.83.

## Bugs found by this sweep (fixed, committed)
1. Single-GPU light-spill placement loss (-6..-18%): per-device dispatch-chain
   serialization — fixed with the calibrated placement economics rule
   (spill ratio >= 1.8 AND (>= 2 devices OR experts >= 2 MiB)).
2. Role-starvation hazard under mixed-quant fragmentation — role-group
   budgeting with intra-group redistribution.

## Pending (downloads in flight)
gpt-oss-120b/20b F16 (SWIGLU_OAI adversarial), Qwen3-235B Q3_K_XL (mild spill),
GLM-4.6 Q4_K_XL (357B spill), Llama-4-Scout (topk=1), DeepSeek-V2-Lite,
granite-4.0-h-small (hybrid mamba), Mixtral 8x7B, OLMoE.
754B/397B re-validation queued for quiet disk (download page-cache contention).

---

# EC3 final clean suite (2026-06-11, downloads paused, final build f80353ae7)

| Model | Regime | Ours | Vanilla | Gain |
|---|---|---|---|---|
| glm51-754b | auto | **17.49** | 13.96 | **+25.3%** |
| glm51-754b | forced | **18.32** | 11.71 | **+56.4%** |
| qwen35-397b | auto | **30.25** | 28.18 | **+7.3%** |
| qwen35-397b | forced | **30.21** | 21.49 | **+40.6%** |
| qwen35-122b | auto | **74.98** | 74.98 | **+0.0%** |
| qwen35-122b | forced | **49.82** | 44.49 | **+12.0%** |
| minimax-m27 | forced | **44.77** | 34.60 | **+29.4%** |
| qwen36-35b | forced | **85.33** | 77.20 | **+10.5%** |
| qwen3-30b | forced | **93.44** | 69.22 | **+35.0%** |
| ernie-21b | forced | **112.00** | 79.45 | **+41.0%** |
| gptoss-120b | forced | **58.59** | 45.53 | **+28.7%** |
| gptoss-20b | forced | **87.02** | 67.36 | **+29.2%** |
| olmoe-7b | forced | **211.91** | 209.26 | **+1.3%** |
| scout-109b | forced | **36.35** | 25.85 | **+40.6%** |
| granite-h | forced | **57.73** | 36.87 | **+56.6%** |
| dsv2-lite | forced | **76.67** | 61.85 | **+24.0%** |

Notes: same -t per model both arms (IQ-quant models t=80, K-quant t=48 — the
shipped guidance). Auto rows = zero-config user experience vs vanilla autofit.
Forced rows = cache contribution in the CPU-offload regime. 122B auto = exact
dormancy parity. New architectures granite-4.0-h (hybrid mamba) and
DeepSeek-V2-Lite: first-try wins (+56.6% / +24.0%), temp-0 outputs IDENTICAL.
16/16 comparisons positive or exact-parity; zero regressions.
