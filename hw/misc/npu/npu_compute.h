/*
 * NPU Compute Kernels
 *
 * Pure C compute functions, no QEMU/kernel dependencies.
 * Used by both QEMU device simulation and standalone tests.
 */

#ifndef NPU_COMPUTE_H
#define NPU_COMPUTE_H

#include <stdint.h>

/* Common flag: bit 0 = fp16 mode */
#define NPU_COMPUTE_FLAG_FP16  (1 << 0)

/* MATMUL flags (maps to npu_inst_matmul_t.flags bits) */
#define NPU_MATMUL_FLAG_FP16        (1 << 0)
#define NPU_MATMUL_FLAG_ACCUMULATE  (1 << 1)
#define NPU_MATMUL_FLAG_TRANSPOSE_A (1 << 2)
#define NPU_MATMUL_FLAG_TRANSPOSE_B (1 << 3)

/* fp16 <-> fp32 conversion */
float    npu_fp16_to_fp32(uint16_t h);
uint16_t npu_fp32_to_fp16(float f);

/*
 * npu_compute_matmul - Matrix multiplication: C = A @ B
 *
 * When FP16 flag is set, A/B/C point to fp16 data in SRAM.
 * Internally promotes to fp32 for computation.
 *
 * Returns: 0 on success, -1 if NULL pointer, -2 if dimension is 0
 */
int npu_compute_matmul(const void* A, const void* B, void* C,
                       uint32_t m, uint32_t n, uint32_t k, uint8_t flags);

/*
 * npu_compute_relu - Element-wise ReLU: dst[i] = max(0, src[i])
 *
 * When FP16 flag is set, src/dst point to fp16 data.
 *
 * Returns: 0 on success, -1 if NULL pointer, -2 if num_elements is 0
 */
int npu_compute_relu(const void* src, void* dst,
                     uint32_t num_elements, uint8_t flags);

/*
 * Arithmetic Operations (element-wise)
 */

/* ADD: dst[i] = a[i] + b[i] */
int npu_compute_add(const void* src_a, const void* src_b, void* dst,
                    uint32_t num_elements, uint8_t flags);

/* SUB: dst[i] = a[i] - b[i] */
int npu_compute_sub(const void* src_a, const void* src_b, void* dst,
                    uint32_t num_elements, uint8_t flags);

/* MUL: dst[i] = a[i] * b[i] */
int npu_compute_mul(const void* src_a, const void* src_b, void* dst,
                    uint32_t num_elements, uint8_t flags);

/* DIV: dst[i] = a[i] / b[i] */
int npu_compute_div(const void* src_a, const void* src_b, void* dst,
                    uint32_t num_elements, uint8_t flags);

/* FMADD: dst[i] = a[i] * b[i] + c[i] */
int npu_compute_fmadd(const void* src_a, const void* src_b, const void* src_c,
                      void* dst, uint32_t num_elements, uint8_t flags);

/* CLAMP: dst[i] = clamp(src[i], min, max) */
int npu_compute_clamp(const void* src, void* dst,
                      uint32_t num_elements, float min_val, float max_val,
                      uint8_t flags);

/* EXP: dst[i] = exp(src[i]) */
int npu_compute_exp(const void* src, void* dst,
                    uint32_t num_elements, uint8_t flags);

/* LOG: dst[i] = log(src[i]) */
int npu_compute_log(const void* src, void* dst,
                    uint32_t num_elements, uint8_t flags);

/* SQRT: dst[i] = sqrt(src[i]) */
int npu_compute_sqrt(const void* src, void* dst,
                     uint32_t num_elements, uint8_t flags);

/* RSQRT: dst[i] = 1/sqrt(src[i]) */
int npu_compute_rsqrt(const void* src, void* dst,
                      uint32_t num_elements, uint8_t flags);

/* ABS: dst[i] = |src[i]| */
int npu_compute_abs(const void* src, void* dst,
                    uint32_t num_elements, uint8_t flags);

/* NEG: dst[i] = -src[i] */
int npu_compute_neg(const void* src, void* dst,
                    uint32_t num_elements, uint8_t flags);

/*
 * Activation Functions
 */

/* SIGMOID: dst[i] = 1 / (1 + exp(-src[i])) */
int npu_compute_sigmoid(const void* src, void* dst,
                        uint32_t num_elements, uint8_t flags);

/* TANH: dst[i] = tanh(src[i]) */
int npu_compute_tanh(const void* src, void* dst,
                     uint32_t num_elements, uint8_t flags);

/* GELU: dst[i] = GELU(src[i]) using tanh approximation */
int npu_compute_gelu(const void* src, void* dst,
                     uint32_t num_elements, uint8_t flags);

/* SOFTMAX: dst = softmax(src) along last axis */
int npu_compute_softmax(const void* src, void* dst,
                        uint32_t batch_size, uint32_t axis_size, uint8_t flags);

/* SWISH: dst[i] = src[i] * sigmoid(src[i]) */
int npu_compute_swish(const void* src, void* dst,
                      uint32_t num_elements, uint8_t flags);

/* MISH: dst[i] = src[i] * tanh(softplus(src[i])) */
int npu_compute_mish(const void* src, void* dst,
                     uint32_t num_elements, uint8_t flags);

/*
 * Reduction Operations
 */

/* REDUCE_SUM: sum along specified axis */
int npu_compute_reduce_sum(const void* src, void* dst,
                           uint32_t outer_size, uint32_t reduce_size,
                           uint32_t inner_size, uint8_t flags);

/* REDUCE_MEAN: mean along specified axis */
int npu_compute_reduce_mean(const void* src, void* dst,
                            uint32_t outer_size, uint32_t reduce_size,
                            uint32_t inner_size, uint8_t flags);

/* REDUCE_MAX: max along specified axis */
int npu_compute_reduce_max(const void* src, void* dst,
                           uint32_t outer_size, uint32_t reduce_size,
                           uint32_t inner_size, uint8_t flags);

/*
 * Layout Operations
 */

/* TRANSPOSE: permute tensor axes (up to 4D) */
int npu_compute_transpose(const void* src, void* dst,
                          const uint32_t* dims, const uint8_t* perm,
                          uint8_t num_dims, uint8_t flags);

/* RESHAPE: copy tensor with new shape (metadata-only if contiguous) */
int npu_compute_reshape(const void* src, void* dst,
                        uint32_t total_elements, uint8_t flags);

/* CONCAT: concatenate two tensors along axis */
int npu_compute_concat(const void* src_0, const void* src_1, void* dst,
                       uint32_t outer_size, uint16_t concat_size_0,
                       uint16_t concat_size_1, uint32_t inner_size, uint8_t flags);

/* SPLIT: split tensor into two along axis */
int npu_compute_split(const void* src, void* dst_0, void* dst_1,
                      uint32_t outer_size, uint16_t split_size,
                      uint32_t inner_size, uint8_t flags);

/*
 * Pooling Operations
 */

/* MAXPOOL: 2D max pooling */
int npu_compute_maxpool(const void* src, void* dst,
                        uint16_t in_c, uint16_t in_h, uint16_t in_w,
                        uint8_t kernel_h, uint8_t kernel_w,
                        uint8_t stride_h, uint8_t stride_w, uint8_t flags);

/* AVGPOOL: 2D average pooling */
int npu_compute_avgpool(const void* src, void* dst,
                        uint16_t in_c, uint16_t in_h, uint16_t in_w,
                        uint8_t kernel_h, uint8_t kernel_w,
                        uint8_t stride_h, uint8_t stride_w, uint8_t flags);

/*
 * Normalization Operations
 */

/* BATCHNORM: dst = (src - mean) / sqrt(var + eps) * gamma + beta */
int npu_compute_batchnorm(const void* src, const void* mean, const void* var,
                          const void* gamma, const void* beta, void* dst,
                          uint32_t num_channels, uint32_t spatial_size,
                          float epsilon, uint8_t flags);

/* LAYERNORM: normalize along last dimension */
int npu_compute_layernorm(const void* src, const void* gamma, const void* beta,
                          void* dst, uint32_t batch_size, uint32_t normalized_shape,
                          float epsilon, uint8_t flags);

/* RMSNORM: dst = src / rms(src) * weight */
int npu_compute_rmsnorm(const void* src, const void* weight, void* dst,
                        uint32_t batch_size, uint32_t normalized_shape,
                        float epsilon, uint8_t flags);

/* GROUPNORM: normalize within channel groups */
int npu_compute_groupnorm(const void* src, const void* gamma, const void* beta,
                          void* dst, uint32_t num_groups, uint32_t num_channels,
                          uint32_t spatial_size, float epsilon, uint8_t flags);

/* INSTANCENORM: normalize per channel per sample */
int npu_compute_instancenorm(const void* src, const void* gamma, const void* beta,
                             void* dst, uint32_t num_channels, uint32_t spatial_size,
                             float epsilon, uint8_t flags);

/*
 * Convolution Operations
 */

/* CONV2D: 2D convolution with optional bias and ReLU fusion */
int npu_compute_conv2d(const void* input, const void* weight, const void* bias,
                       void* dst, uint16_t in_c, uint16_t in_h, uint16_t in_w,
                       uint16_t out_c, uint8_t kernel_h, uint8_t kernel_w,
                       uint8_t pad_h, uint8_t pad_w,
                       uint8_t stride_h, uint8_t stride_w, uint8_t flags);

/* DEPTHWISE_CONV: Depthwise separable convolution (MobileNet) */
int npu_compute_depthwise_conv(const void* input, const void* weight, void* dst,
                               uint16_t channels, uint16_t in_h, uint16_t in_w,
                               uint8_t kernel_h, uint8_t kernel_w,
                               uint8_t pad_h, uint8_t pad_w,
                               uint8_t stride_h, uint8_t stride_w, uint8_t flags);

/*
 * Linear Algebra (extended)
 */

/* GEMM: C = alpha * A @ B + beta * C */
int npu_compute_gemm(const void* A, const void* B, void* C,
                     uint32_t m, uint32_t n, uint32_t k,
                     float alpha, float beta, uint8_t flags);

/* DOT: dst = sum(a[i] * b[i]) */
int npu_compute_dot(const void* src_a, const void* src_b, void* dst,
                    uint32_t num_elements, uint8_t flags);

/*
 * Attention Operations
 */

/* SCALED_DOT_PRODUCT_ATTENTION: dst = softmax(Q @ K^T / scale) @ V */
int npu_compute_sdpa(const void* Q, const void* K, const void* V, void* dst,
                     uint32_t num_heads, uint32_t seq_len_q, uint32_t seq_len_kv,
                     uint32_t head_dim, float scale, uint8_t flags);

/*
 * KV Cache Operations (VLM/VLA)
 */

/* KV_CACHE_APPEND: append new K/V vectors to cache at cur_seq_pos */
int npu_compute_kv_cache_append(const void* k_new, const void* v_new,
                                void* k_cache, void* v_cache,
                                uint32_t cur_seq_pos, uint16_t num_kv_heads,
                                uint16_t head_dim, uint32_t max_seq_len,
                                uint8_t flags);

/* KV_CACHE_ATTENTION: decode attention with cached K/V (seq_len_q=1) */
int npu_compute_kv_cache_attention(const void* Q, const void* k_cache,
                                   const void* v_cache, void* dst,
                                   uint16_t num_heads, uint16_t num_kv_heads,
                                   uint32_t cur_seq_len, uint16_t head_dim,
                                   uint16_t max_seq_len, float scale,
                                   uint8_t flags);

/* KV_CACHE_RESET: zero cache memory */
int npu_compute_kv_cache_reset(void* k_cache, void* v_cache,
                               uint16_t num_kv_heads, uint16_t head_dim,
                               uint32_t max_seq_len, uint8_t flags);

/*
 * Token Processing Operations (VLM/VLA)
 */

/* EMBEDDING_LOOKUP: dst = table[token_id] */
int npu_compute_embedding_lookup(const void* table, void* dst,
                                 uint32_t token_id, uint32_t vocab_size,
                                 uint32_t embed_dim, uint8_t flags);

/* TOKEN_SAMPLE: sample token from logits (greedy/top-k/top-p) */
int npu_compute_token_sample(const void* logits, void* dst_token,
                             uint32_t vocab_size, uint8_t mode,
                             float temperature, uint16_t top_k,
                             uint32_t seed, uint8_t flags);

/* ROTARY_EMBEDDING: apply RoPE in-place */
int npu_compute_rotary_embedding(const void* src, void* dst,
                                 uint32_t position, uint16_t num_heads,
                                 uint16_t head_dim, float theta_base,
                                 uint8_t flags);

/*
 * Data Type Operations
 */

/* CAST: convert between data types */
int npu_compute_cast(const void* src, void* dst,
                     uint32_t num_elements, uint8_t src_dtype, uint8_t dst_dtype);

/* QUANTIZE: fp32 -> int8 */
int npu_compute_quantize(const void* src, void* dst,
                         const void* scale, const void* zero_point,
                         uint32_t num_elements, uint32_t num_channels,
                         uint8_t flags);

/* DEQUANTIZE: int8 -> fp32 */
int npu_compute_dequantize(const void* src, void* dst,
                           const void* scale, const void* zero_point,
                           uint32_t num_elements, uint32_t num_channels,
                           uint8_t flags);

/*
 * Data Movement / Manipulation Operations
 */

/* GATHER: dst[outer][i][inner] = src[outer][indices[i]][inner] */
int npu_compute_gather(const void* src, const void* indices, void* dst,
                       uint32_t outer_size, uint32_t gather_size,
                       uint32_t inner_size, uint32_t num_indices, uint8_t flags);

/* SLICE: extract sub-tensor along one axis */
int npu_compute_slice(const void* src, void* dst,
                      uint32_t outer_size, uint32_t src_axis_size,
                      uint32_t inner_size, uint32_t start,
                      uint32_t length, uint32_t step, uint8_t flags);

/* PAD: pad tensor with constant value (up to 4D) */
int npu_compute_pad(const void* src, void* dst,
                    const uint16_t* src_dims, const uint16_t* pad_before,
                    const uint16_t* pad_after, uint8_t num_dims,
                    uint32_t pad_value_bits, uint8_t flags);

/* WHERE: dst[i] = cond[i] ? true_val[i] : false_val[i] */
int npu_compute_where(const void* cond, const void* true_val,
                      const void* false_val, void* dst,
                      uint32_t num_elements, uint8_t flags);

#endif /* NPU_COMPUTE_H */
