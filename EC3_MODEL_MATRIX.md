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

## Forced CPU-offload regime (`-ngl 99 -ncmoe 99`, same placement both arms)

| Model | EC3 | Vanilla | Delta | temp-0 identity |
|---|---|---|---|---|
| GLM-5.1 754B | **19.4±1.1** | 14.0* | **+34-39%** | PPL-validated |
| Qwen3.5-397B | **33.3** | 28.2* | **+18%** | PPL-validated |
| Qwen3.5-122B | **51.3** | 43.5 | **+17.8%** | IDENTICAL |
| Qwen3-30B-A3B | **90.6** | 67.4 | **+34.4%** | IDENTICAL |
| ERNIE-4.5-21B | **111.6** | 77.4 | **+44.3%** | IDENTICAL |
| Qwen3.6-35B-A3B | **89.0** | 77.3 | **+15%** | PPL-validated |

*vanilla figure = best stock config (fitt), stricter than same-placement.

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
