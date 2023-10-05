#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

void lm_ggml_vk_init(void);

void lm_ggml_vk_preallocate_buffers_graph(struct lm_ggml_tensor * node);
void lm_ggml_vk_preallocate_buffers(void);
void lm_ggml_vk_build_graph(struct lm_ggml_tensor * node);
bool lm_ggml_vk_compute_forward(struct lm_ggml_compute_params * params, struct lm_ggml_tensor * tensor);
void lm_ggml_vk_graph_cleanup(void);

void * lm_ggml_vk_host_malloc(size_t size);
void   lm_ggml_vk_host_free(void * ptr);

void lm_ggml_vk_free_data(const struct lm_ggml_tensor * tensor);

void lm_ggml_vk_transform_tensor(void * data, struct lm_ggml_tensor * tensor);

#ifdef  __cplusplus
}
#endif
