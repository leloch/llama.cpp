#pragma once

// Expert cache v3 — registration entry point, called from ggml_backend_cuda_reg().
// Populates ggml_expert_cache_v3 (see ggml-backend-expert-cache.h) when
// LLAMA_EC3=1 is set in the environment.

#ifdef __cplusplus
extern "C" {
#endif

void ggml_expert_cache_v3_register(void);

#ifdef __cplusplus
}
#endif
