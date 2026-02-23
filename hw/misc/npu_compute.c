/*
 * NPU Compute Kernels - Implementation
 *
 * Naive reference implementations for correctness testing.
 * Not optimized — real hardware would use dedicated accelerators.
 */

#include "npu_compute.h"
#include <stddef.h>
#include <string.h>

/* ========================================
 * fp16 <-> fp32 conversion (IEEE 754)
 * ======================================== */

float npu_fp16_to_fp32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    float result;

    if (exp == 0) {
        if (mant == 0) {
            /* +/- zero */
            f = sign;
        } else {
            /* Denormalized: convert to normalized fp32 */
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        /* Inf / NaN */
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        /* Normalized */
        f = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }

    memcpy(&result, &f, sizeof(float));
    return result;
}

uint16_t npu_fp32_to_fp16(float f)
{
    uint32_t bits;
    uint32_t sign, exp, mant;

    memcpy(&bits, &f, sizeof(uint32_t));

    sign = (bits >> 16) & 0x8000;
    exp  = (bits >> 23) & 0xFF;
    mant = bits & 0x7FFFFF;

    if (exp == 0xFF) {
        /* Inf / NaN */
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x200 : 0));
    }

    /* Rebias exponent: fp32 bias=127, fp16 bias=15 */
    int new_exp = (int)exp - 127 + 15;

    if (new_exp >= 0x1F) {
        /* Overflow → Inf */
        return (uint16_t)(sign | 0x7C00);
    }

    if (new_exp <= 0) {
        /* Underflow → zero (no denorm rounding for simplicity) */
        return (uint16_t)sign;
    }

    return (uint16_t)(sign | ((uint32_t)new_exp << 10) | (mant >> 13));
}

/* ========================================
 * Helper: access element with transpose
 * ======================================== */

static inline float mat_get_fp32(const float* M, uint32_t r, uint32_t c,
                                 uint32_t cols, int transposed)
{
    if (transposed)
        return M[c * cols + r];
    return M[r * cols + c];
}

static inline float mat_get_fp16(const uint16_t* M, uint32_t r, uint32_t c,
                                 uint32_t cols, int transposed)
{
    if (transposed)
        return npu_fp16_to_fp32(M[c * cols + r]);
    return npu_fp16_to_fp32(M[r * cols + c]);
}

/* ========================================
 * MATMUL
 * ======================================== */

int npu_compute_matmul(const void* A, const void* B, void* C,
                       uint32_t m, uint32_t n, uint32_t k, uint8_t flags)
{
    uint32_t i, j, p;
    int transpose_a = (flags & NPU_MATMUL_FLAG_TRANSPOSE_A) != 0;
    int transpose_b = (flags & NPU_MATMUL_FLAG_TRANSPOSE_B) != 0;
    int accumulate  = (flags & NPU_MATMUL_FLAG_ACCUMULATE) != 0;
    int fp16        = (flags & NPU_MATMUL_FLAG_FP16) != 0;

    uint32_t a_cols = transpose_a ? m : k;
    uint32_t b_cols = transpose_b ? k : n;

    if (!A || !B || !C)
        return -1;

    if (m == 0 || n == 0 || k == 0)
        return -2;

    if (fp16) {
        const uint16_t *Ah = (const uint16_t *)A;
        const uint16_t *Bh = (const uint16_t *)B;
        uint16_t *Ch = (uint16_t *)C;

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                float sum = 0.0f;
                for (p = 0; p < k; p++) {
                    float a_val = mat_get_fp16(Ah, i, p, a_cols, transpose_a);
                    float b_val = mat_get_fp16(Bh, p, j, b_cols, transpose_b);
                    sum += a_val * b_val;
                }
                if (accumulate)
                    sum += npu_fp16_to_fp32(Ch[i * n + j]);
                Ch[i * n + j] = npu_fp32_to_fp16(sum);
            }
        }
    } else {
        const float *Af = (const float *)A;
        const float *Bf = (const float *)B;
        float *Cf = (float *)C;

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                float sum = 0.0f;
                for (p = 0; p < k; p++) {
                    float a_val = mat_get_fp32(Af, i, p, a_cols, transpose_a);
                    float b_val = mat_get_fp32(Bf, p, j, b_cols, transpose_b);
                    sum += a_val * b_val;
                }
                if (accumulate)
                    Cf[i * n + j] += sum;
                else
                    Cf[i * n + j] = sum;
            }
        }
    }

    return 0;
}

/* ========================================
 * RELU
 * ======================================== */

int npu_compute_relu(const void* src, void* dst,
                     uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src || !dst)
        return -1;

    if (num_elements == 0)
        return -2;

    if (fp16) {
        const uint16_t *sh = (const uint16_t *)src;
        uint16_t *dh = (uint16_t *)dst;

        for (i = 0; i < num_elements; i++) {
            float val = npu_fp16_to_fp32(sh[i]);
            dh[i] = npu_fp32_to_fp16(val > 0.0f ? val : 0.0f);
        }
    } else {
        const float *sf = (const float *)src;
        float *df = (float *)dst;

        for (i = 0; i < num_elements; i++)
            df[i] = sf[i] > 0.0f ? sf[i] : 0.0f;
    }

    return 0;
}
