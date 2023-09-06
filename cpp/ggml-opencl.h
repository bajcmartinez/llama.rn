#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

void lm_ggml_cl_init(void);

void   lm_ggml_cl_mul(const struct lm_ggml_tensor * src0, const struct lm_ggml_tensor * src1, struct lm_ggml_tensor * dst);
bool   lm_ggml_cl_can_mul_mat(const struct lm_ggml_tensor * src0, const struct lm_ggml_tensor * src1, struct lm_ggml_tensor * dst);
size_t lm_ggml_cl_mul_mat_get_wsize(const struct lm_ggml_tensor * src0, const struct lm_ggml_tensor * src1, struct lm_ggml_tensor * dst);
void   lm_ggml_cl_mul_mat(const struct lm_ggml_tensor * src0, const struct lm_ggml_tensor * src1, struct lm_ggml_tensor * dst, void * wdata, size_t wsize);

void * lm_ggml_cl_host_malloc(size_t size);
void   lm_ggml_cl_host_free(void * ptr);

void lm_ggml_cl_free_data(const struct lm_ggml_tensor* tensor);

void lm_ggml_cl_transform_tensor(void * data, struct lm_ggml_tensor * tensor);

#ifdef  __cplusplus
}
#endif
