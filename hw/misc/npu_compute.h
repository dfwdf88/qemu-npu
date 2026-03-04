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

#endif /* NPU_COMPUTE_H */
