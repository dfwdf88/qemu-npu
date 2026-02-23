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

#endif /* NPU_COMPUTE_H */
