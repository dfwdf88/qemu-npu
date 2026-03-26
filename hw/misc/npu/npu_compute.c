/*
 * NPU Compute Kernels - Implementation
 *
 * Naive reference implementations for correctness testing.
 * Not optimized — real hardware would use dedicated accelerators.
 */

#include "npu_compute.h"
#include <stddef.h>
#include <string.h>
#include <math.h>

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

/* ========================================
 * Arithmetic Operations
 * ======================================== */

int npu_compute_add(const void* src_a, const void* src_b, void* dst,
                    uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src_a || !src_b || !dst)
        return -1;

    if (num_elements == 0)
        return -2;

    if (fp16) {
        const uint16_t *ah = (const uint16_t *)src_a;
        const uint16_t *bh = (const uint16_t *)src_b;
        uint16_t *dh = (uint16_t *)dst;

        for (i = 0; i < num_elements; i++) {
            float a_val = npu_fp16_to_fp32(ah[i]);
            float b_val = npu_fp16_to_fp32(bh[i]);
            dh[i] = npu_fp32_to_fp16(a_val + b_val);
        }
    } else {
        const float *af = (const float *)src_a;
        const float *bf = (const float *)src_b;
        float *df = (float *)dst;

        for (i = 0; i < num_elements; i++)
            df[i] = af[i] + bf[i];
    }

    return 0;
}

int npu_compute_sub(const void* src_a, const void* src_b, void* dst,
                    uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src_a || !src_b || !dst)
        return -1;

    if (num_elements == 0)
        return -2;

    if (fp16) {
        const uint16_t *ah = (const uint16_t *)src_a;
        const uint16_t *bh = (const uint16_t *)src_b;
        uint16_t *dh = (uint16_t *)dst;

        for (i = 0; i < num_elements; i++) {
            float a_val = npu_fp16_to_fp32(ah[i]);
            float b_val = npu_fp16_to_fp32(bh[i]);
            dh[i] = npu_fp32_to_fp16(a_val - b_val);
        }
    } else {
        const float *af = (const float *)src_a;
        const float *bf = (const float *)src_b;
        float *df = (float *)dst;

        for (i = 0; i < num_elements; i++)
            df[i] = af[i] - bf[i];
    }

    return 0;
}

int npu_compute_mul(const void* src_a, const void* src_b, void* dst,
                    uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src_a || !src_b || !dst)
        return -1;

    if (num_elements == 0)
        return -2;

    if (fp16) {
        const uint16_t *ah = (const uint16_t *)src_a;
        const uint16_t *bh = (const uint16_t *)src_b;
        uint16_t *dh = (uint16_t *)dst;

        for (i = 0; i < num_elements; i++) {
            float a_val = npu_fp16_to_fp32(ah[i]);
            float b_val = npu_fp16_to_fp32(bh[i]);
            dh[i] = npu_fp32_to_fp16(a_val * b_val);
        }
    } else {
        const float *af = (const float *)src_a;
        const float *bf = (const float *)src_b;
        float *df = (float *)dst;

        for (i = 0; i < num_elements; i++)
            df[i] = af[i] * bf[i];
    }

    return 0;
}

int npu_compute_div(const void* src_a, const void* src_b, void* dst,
                    uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src_a || !src_b || !dst)
        return -1;

    if (num_elements == 0)
        return -2;

    if (fp16) {
        const uint16_t *ah = (const uint16_t *)src_a;
        const uint16_t *bh = (const uint16_t *)src_b;
        uint16_t *dh = (uint16_t *)dst;

        for (i = 0; i < num_elements; i++) {
            float a_val = npu_fp16_to_fp32(ah[i]);
            float b_val = npu_fp16_to_fp32(bh[i]);
            dh[i] = npu_fp32_to_fp16(a_val / b_val);
        }
    } else {
        const float *af = (const float *)src_a;
        const float *bf = (const float *)src_b;
        float *df = (float *)dst;

        for (i = 0; i < num_elements; i++)
            df[i] = af[i] / bf[i];
    }

    return 0;
}

/* ========================================
 * Activation Functions
 * ======================================== */

int npu_compute_sigmoid(const void* src, void* dst,
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
            dh[i] = npu_fp32_to_fp16(1.0f / (1.0f + expf(-val)));
        }
    } else {
        const float *sf = (const float *)src;
        float *df = (float *)dst;

        for (i = 0; i < num_elements; i++)
            df[i] = 1.0f / (1.0f + expf(-sf[i]));
    }

    return 0;
}

int npu_compute_tanh(const void* src, void* dst,
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
            dh[i] = npu_fp32_to_fp16(tanhf(val));
        }
    } else {
        const float *sf = (const float *)src;
        float *df = (float *)dst;

        for (i = 0; i < num_elements; i++)
            df[i] = tanhf(sf[i]);
    }

    return 0;
}

int npu_compute_gelu(const void* src, void* dst,
                     uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;
    const float sqrt_2_over_pi = 0.7978845608f;  /* sqrt(2/π) */
    const float coeff = 0.044715f;

    if (!src || !dst)
        return -1;

    if (num_elements == 0)
        return -2;

    if (fp16) {
        const uint16_t *sh = (const uint16_t *)src;
        uint16_t *dh = (uint16_t *)dst;

        for (i = 0; i < num_elements; i++) {
            float x = npu_fp16_to_fp32(sh[i]);
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            float result = 0.5f * x * (1.0f + tanhf(inner));
            dh[i] = npu_fp32_to_fp16(result);
        }
    } else {
        const float *sf = (const float *)src;
        float *df = (float *)dst;

        for (i = 0; i < num_elements; i++) {
            float x = sf[i];
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            df[i] = 0.5f * x * (1.0f + tanhf(inner));
        }
    }

    return 0;
}

int npu_compute_softmax(const void* src, void* dst,
                        uint32_t batch_size, uint32_t axis_size, uint8_t flags)
{
    uint32_t b, i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src || !dst)
        return -1;

    if (batch_size == 0 || axis_size == 0)
        return -2;

    if (fp16) {
        const uint16_t *sh = (const uint16_t *)src;
        uint16_t *dh = (uint16_t *)dst;

        for (b = 0; b < batch_size; b++) {
            const uint16_t *vec_in = &sh[b * axis_size];
            uint16_t *vec_out = &dh[b * axis_size];

            /* Find max for numerical stability */
            float max_val = npu_fp16_to_fp32(vec_in[0]);
            for (i = 1; i < axis_size; i++) {
                float val = npu_fp16_to_fp32(vec_in[i]);
                if (val > max_val)
                    max_val = val;
            }

            /* Two-pass: compute exp+sum, then normalize (use output as temp) */
            float sum = 0.0f;
            for (i = 0; i < axis_size; i++) {
                float val = npu_fp16_to_fp32(vec_in[i]);
                float ev = expf(val - max_val);
                /* Store fp32 exp value temporarily as fp16 in output */
                vec_out[i] = npu_fp32_to_fp16(ev);
                sum += ev;
            }

            /* Normalize */
            for (i = 0; i < axis_size; i++) {
                float ev = npu_fp16_to_fp32(vec_out[i]);
                vec_out[i] = npu_fp32_to_fp16(ev / sum);
            }
        }
    } else {
        const float *sf = (const float *)src;
        float *df = (float *)dst;

        for (b = 0; b < batch_size; b++) {
            const float *vec_in = &sf[b * axis_size];
            float *vec_out = &df[b * axis_size];

            /* Find max for numerical stability */
            float max_val = vec_in[0];
            for (i = 1; i < axis_size; i++) {
                if (vec_in[i] > max_val)
                    max_val = vec_in[i];
            }

            /* Compute exp and sum */
            float sum = 0.0f;
            for (i = 0; i < axis_size; i++) {
                vec_out[i] = expf(vec_in[i] - max_val);
                sum += vec_out[i];
            }

            /* Normalize */
            for (i = 0; i < axis_size; i++)
                vec_out[i] /= sum;
        }
    }

    return 0;
}

/* ========================================
 * Reduction Operations
 * ======================================== */

int npu_compute_reduce_sum(const void* src, void* dst,
                           uint32_t outer_size, uint32_t reduce_size,
                           uint32_t inner_size, uint8_t flags)
{
    uint32_t o, r, i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src || !dst)
        return -1;

    if (outer_size == 0 || reduce_size == 0 || inner_size == 0)
        return -2;

    if (fp16) {
        const uint16_t *sh = (const uint16_t *)src;
        uint16_t *dh = (uint16_t *)dst;

        for (o = 0; o < outer_size; o++) {
            for (i = 0; i < inner_size; i++) {
                float sum = 0.0f;
                for (r = 0; r < reduce_size; r++) {
                    uint32_t src_idx = o * reduce_size * inner_size + r * inner_size + i;
                    sum += npu_fp16_to_fp32(sh[src_idx]);
                }
                uint32_t dst_idx = o * inner_size + i;
                dh[dst_idx] = npu_fp32_to_fp16(sum);
            }
        }
    } else {
        const float *sf = (const float *)src;
        float *df = (float *)dst;

        for (o = 0; o < outer_size; o++) {
            for (i = 0; i < inner_size; i++) {
                float sum = 0.0f;
                for (r = 0; r < reduce_size; r++) {
                    uint32_t src_idx = o * reduce_size * inner_size + r * inner_size + i;
                    sum += sf[src_idx];
                }
                uint32_t dst_idx = o * inner_size + i;
                df[dst_idx] = sum;
            }
        }
    }

    return 0;
}

int npu_compute_reduce_mean(const void* src, void* dst,
                            uint32_t outer_size, uint32_t reduce_size,
                            uint32_t inner_size, uint8_t flags)
{
    uint32_t o, r, i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src || !dst)
        return -1;

    if (outer_size == 0 || reduce_size == 0 || inner_size == 0)
        return -2;

    if (fp16) {
        const uint16_t *sh = (const uint16_t *)src;
        uint16_t *dh = (uint16_t *)dst;

        for (o = 0; o < outer_size; o++) {
            for (i = 0; i < inner_size; i++) {
                float sum = 0.0f;
                for (r = 0; r < reduce_size; r++) {
                    uint32_t src_idx = o * reduce_size * inner_size + r * inner_size + i;
                    sum += npu_fp16_to_fp32(sh[src_idx]);
                }
                uint32_t dst_idx = o * inner_size + i;
                dh[dst_idx] = npu_fp32_to_fp16(sum / (float)reduce_size);
            }
        }
    } else {
        const float *sf = (const float *)src;
        float *df = (float *)dst;

        for (o = 0; o < outer_size; o++) {
            for (i = 0; i < inner_size; i++) {
                float sum = 0.0f;
                for (r = 0; r < reduce_size; r++) {
                    uint32_t src_idx = o * reduce_size * inner_size + r * inner_size + i;
                    sum += sf[src_idx];
                }
                uint32_t dst_idx = o * inner_size + i;
                df[dst_idx] = sum / (float)reduce_size;
            }
        }
    }

    return 0;
}

int npu_compute_reduce_max(const void* src, void* dst,
                           uint32_t outer_size, uint32_t reduce_size,
                           uint32_t inner_size, uint8_t flags)
{
    uint32_t o, r, i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src || !dst)
        return -1;

    if (outer_size == 0 || reduce_size == 0 || inner_size == 0)
        return -2;

    if (fp16) {
        const uint16_t *sh = (const uint16_t *)src;
        uint16_t *dh = (uint16_t *)dst;

        for (o = 0; o < outer_size; o++) {
            for (i = 0; i < inner_size; i++) {
                uint32_t first_idx = o * reduce_size * inner_size + i;
                float max_val = npu_fp16_to_fp32(sh[first_idx]);
                for (r = 1; r < reduce_size; r++) {
                    uint32_t src_idx = o * reduce_size * inner_size + r * inner_size + i;
                    float val = npu_fp16_to_fp32(sh[src_idx]);
                    if (val > max_val)
                        max_val = val;
                }
                uint32_t dst_idx = o * inner_size + i;
                dh[dst_idx] = npu_fp32_to_fp16(max_val);
            }
        }
    } else {
        const float *sf = (const float *)src;
        float *df = (float *)dst;

        for (o = 0; o < outer_size; o++) {
            for (i = 0; i < inner_size; i++) {
                uint32_t first_idx = o * reduce_size * inner_size + i;
                float max_val = sf[first_idx];
                for (r = 1; r < reduce_size; r++) {
                    uint32_t src_idx = o * reduce_size * inner_size + r * inner_size + i;
                    if (sf[src_idx] > max_val)
                        max_val = sf[src_idx];
                }
                uint32_t dst_idx = o * inner_size + i;
                df[dst_idx] = max_val;
            }
        }
    }

    return 0;
}

/* ========================================
 * Layout Operations
 * ======================================== */

int npu_compute_transpose(const void* src, void* dst,
                          const uint32_t* dims, const uint8_t* perm,
                          uint8_t num_dims, uint8_t flags)
{
    uint32_t i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src || !dst || !dims || !perm)
        return -1;

    if (num_dims < 2 || num_dims > 4)
        return -2;

    /* Calculate total elements */
    uint32_t total = 1;
    for (i = 0; i < num_dims; i++)
        total *= dims[i];

    /* For simplicity, handle common case: 2D transpose */
    if (num_dims == 2 && perm[0] == 1 && perm[1] == 0) {
        uint32_t rows = dims[0];
        uint32_t cols = dims[1];

        if (fp16) {
            const uint16_t *sh = (const uint16_t *)src;
            uint16_t *dh = (uint16_t *)dst;
            for (uint32_t r = 0; r < rows; r++) {
                for (uint32_t c = 0; c < cols; c++) {
                    dh[c * rows + r] = sh[r * cols + c];
                }
            }
        } else {
            const float *sf = (const float *)src;
            float *df = (float *)dst;
            for (uint32_t r = 0; r < rows; r++) {
                for (uint32_t c = 0; c < cols; c++) {
                    df[c * rows + r] = sf[r * cols + c];
                }
            }
        }
    } else {
        /* General N-D transpose - copy element by element with index mapping */
        for (i = 0; i < total; i++) {
            /* Calculate multi-dimensional index from linear index */
            uint32_t indices[4] = {0};
            uint32_t temp = i;
            for (int d = num_dims - 1; d >= 0; d--) {
                indices[d] = temp % dims[d];
                temp /= dims[d];
            }

            /* Permute indices */
            uint32_t perm_indices[4] = {0};
            for (uint32_t d = 0; d < num_dims; d++) {
                perm_indices[d] = indices[perm[d]];
            }

            /* Calculate output linear index */
            uint32_t out_idx = 0;
            uint32_t stride = 1;
            for (int d = num_dims - 1; d >= 0; d--) {
                out_idx += perm_indices[d] * stride;
                stride *= dims[perm[d]];
            }

            /* Copy element */
            if (fp16) {
                ((uint16_t *)dst)[out_idx] = ((const uint16_t *)src)[i];
            } else {
                ((float *)dst)[out_idx] = ((const float *)src)[i];
            }
        }
    }

    return 0;
}

int npu_compute_reshape(const void* src, void* dst,
                        uint32_t total_elements, uint8_t flags)
{
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src || !dst)
        return -1;

    if (total_elements == 0)
        return -2;

    /* Reshape is just a contiguous copy */
    uint32_t elem_size = fp16 ? 2 : 4;
    memcpy(dst, src, total_elements * elem_size);

    return 0;
}

int npu_compute_concat(const void* src_0, const void* src_1, void* dst,
                       uint32_t outer_size, uint16_t concat_size_0,
                       uint16_t concat_size_1, uint32_t inner_size, uint8_t flags)
{
    uint32_t o, c, i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src_0 || !src_1 || !dst)
        return -1;

    if (outer_size == 0 || (concat_size_0 == 0 && concat_size_1 == 0) || inner_size == 0)
        return -2;

    if (fp16) {
        const uint16_t *s0 = (const uint16_t *)src_0;
        const uint16_t *s1 = (const uint16_t *)src_1;
        uint16_t *dh = (uint16_t *)dst;

        for (o = 0; o < outer_size; o++) {
            for (i = 0; i < inner_size; i++) {
                /* Copy from src_0 */
                for (c = 0; c < concat_size_0; c++) {
                    uint32_t src_idx = o * concat_size_0 * inner_size + c * inner_size + i;
                    uint32_t dst_idx = o * (concat_size_0 + concat_size_1) * inner_size + c * inner_size + i;
                    dh[dst_idx] = s0[src_idx];
                }
                /* Copy from src_1 */
                for (c = 0; c < concat_size_1; c++) {
                    uint32_t src_idx = o * concat_size_1 * inner_size + c * inner_size + i;
                    uint32_t dst_idx = o * (concat_size_0 + concat_size_1) * inner_size + (concat_size_0 + c) * inner_size + i;
                    dh[dst_idx] = s1[src_idx];
                }
            }
        }
    } else {
        const float *s0 = (const float *)src_0;
        const float *s1 = (const float *)src_1;
        float *df = (float *)dst;

        for (o = 0; o < outer_size; o++) {
            for (i = 0; i < inner_size; i++) {
                /* Copy from src_0 */
                for (c = 0; c < concat_size_0; c++) {
                    uint32_t src_idx = o * concat_size_0 * inner_size + c * inner_size + i;
                    uint32_t dst_idx = o * (concat_size_0 + concat_size_1) * inner_size + c * inner_size + i;
                    df[dst_idx] = s0[src_idx];
                }
                /* Copy from src_1 */
                for (c = 0; c < concat_size_1; c++) {
                    uint32_t src_idx = o * concat_size_1 * inner_size + c * inner_size + i;
                    uint32_t dst_idx = o * (concat_size_0 + concat_size_1) * inner_size + (concat_size_0 + c) * inner_size + i;
                    df[dst_idx] = s1[src_idx];
                }
            }
        }
    }

    return 0;
}

int npu_compute_split(const void* src, void* dst_0, void* dst_1,
                      uint32_t outer_size, uint16_t split_size,
                      uint32_t inner_size, uint8_t flags)
{
    uint32_t o, s, i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src || !dst_0 || !dst_1)
        return -1;

    if (outer_size == 0 || split_size == 0 || inner_size == 0)
        return -2;

    if (fp16) {
        const uint16_t *sh = (const uint16_t *)src;
        uint16_t *d0 = (uint16_t *)dst_0;
        uint16_t *d1 = (uint16_t *)dst_1;

        for (o = 0; o < outer_size; o++) {
            for (i = 0; i < inner_size; i++) {
                /* Copy to dst_0 (first split_size elements along axis) */
                for (s = 0; s < split_size; s++) {
                    uint32_t src_idx = o * (2 * split_size) * inner_size + s * inner_size + i;
                    uint32_t dst_idx = o * split_size * inner_size + s * inner_size + i;
                    d0[dst_idx] = sh[src_idx];
                }
                /* Copy to dst_1 (second split_size elements along axis) */
                for (s = 0; s < split_size; s++) {
                    uint32_t src_idx = o * (2 * split_size) * inner_size + (split_size + s) * inner_size + i;
                    uint32_t dst_idx = o * split_size * inner_size + s * inner_size + i;
                    d1[dst_idx] = sh[src_idx];
                }
            }
        }
    } else {
        const float *sf = (const float *)src;
        float *d0 = (float *)dst_0;
        float *d1 = (float *)dst_1;

        for (o = 0; o < outer_size; o++) {
            for (i = 0; i < inner_size; i++) {
                /* Copy to dst_0 */
                for (s = 0; s < split_size; s++) {
                    uint32_t src_idx = o * (2 * split_size) * inner_size + s * inner_size + i;
                    uint32_t dst_idx = o * split_size * inner_size + s * inner_size + i;
                    d0[dst_idx] = sf[src_idx];
                }
                /* Copy to dst_1 */
                for (s = 0; s < split_size; s++) {
                    uint32_t src_idx = o * (2 * split_size) * inner_size + (split_size + s) * inner_size + i;
                    uint32_t dst_idx = o * split_size * inner_size + s * inner_size + i;
                    d1[dst_idx] = sf[src_idx];
                }
            }
        }
    }

    return 0;
}

/* ========================================
 * Pooling Operations
 * ======================================== */

int npu_compute_maxpool(const void* src, void* dst,
                        uint16_t in_c, uint16_t in_h, uint16_t in_w,
                        uint8_t kernel_h, uint8_t kernel_w,
                        uint8_t stride_h, uint8_t stride_w, uint8_t flags)
{
    uint16_t c, out_h, out_w, kh, kw;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

    if (!src || !dst)
        return -1;

    if (in_c == 0 || in_h == 0 || in_w == 0)
        return -2;

    /* Calculate output dimensions */
    uint16_t out_h_dim = (in_h - kernel_h) / stride_h + 1;
    uint16_t out_w_dim = (in_w - kernel_w) / stride_w + 1;

    if (fp16) {
        const uint16_t *sh = (const uint16_t *)src;
        uint16_t *dh = (uint16_t *)dst;

        for (c = 0; c < in_c; c++) {
            for (out_h = 0; out_h < out_h_dim; out_h++) {
                for (out_w = 0; out_w < out_w_dim; out_w++) {
                    uint16_t in_h_start = out_h * stride_h;
                    uint16_t in_w_start = out_w * stride_w;

                    /* Find max in kernel window */
                    float max_val = -INFINITY;
                    for (kh = 0; kh < kernel_h; kh++) {
                        for (kw = 0; kw < kernel_w; kw++) {
                            uint32_t src_idx = c * in_h * in_w + (in_h_start + kh) * in_w + (in_w_start + kw);
                            float val = npu_fp16_to_fp32(sh[src_idx]);
                            if (val > max_val)
                                max_val = val;
                        }
                    }

                    uint32_t dst_idx = c * out_h_dim * out_w_dim + out_h * out_w_dim + out_w;
                    dh[dst_idx] = npu_fp32_to_fp16(max_val);
                }
            }
        }
    } else {
        const float *sf = (const float *)src;
        float *df = (float *)dst;

        for (c = 0; c < in_c; c++) {
            for (out_h = 0; out_h < out_h_dim; out_h++) {
                for (out_w = 0; out_w < out_w_dim; out_w++) {
                    uint16_t in_h_start = out_h * stride_h;
                    uint16_t in_w_start = out_w * stride_w;

                    /* Find max in kernel window */
                    float max_val = -INFINITY;
                    for (kh = 0; kh < kernel_h; kh++) {
                        for (kw = 0; kw < kernel_w; kw++) {
                            uint32_t src_idx = c * in_h * in_w + (in_h_start + kh) * in_w + (in_w_start + kw);
                            if (sf[src_idx] > max_val)
                                max_val = sf[src_idx];
                        }
                    }

                    uint32_t dst_idx = c * out_h_dim * out_w_dim + out_h * out_w_dim + out_w;
                    df[dst_idx] = max_val;
                }
            }
        }
    }

    return 0;
}

int npu_compute_avgpool(const void* src, void* dst,
                        uint16_t in_c, uint16_t in_h, uint16_t in_w,
                        uint8_t kernel_h, uint8_t kernel_w,
                        uint8_t stride_h, uint8_t stride_w, uint8_t flags)
{
    uint16_t c, out_h, out_w, kh, kw;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;
    float pool_size = (float)(kernel_h * kernel_w);

    if (!src || !dst)
        return -1;

    if (in_c == 0 || in_h == 0 || in_w == 0)
        return -2;

    /* Calculate output dimensions */
    uint16_t out_h_dim = (in_h - kernel_h) / stride_h + 1;
    uint16_t out_w_dim = (in_w - kernel_w) / stride_w + 1;

    if (fp16) {
        const uint16_t *sh = (const uint16_t *)src;
        uint16_t *dh = (uint16_t *)dst;

        for (c = 0; c < in_c; c++) {
            for (out_h = 0; out_h < out_h_dim; out_h++) {
                for (out_w = 0; out_w < out_w_dim; out_w++) {
                    uint16_t in_h_start = out_h * stride_h;
                    uint16_t in_w_start = out_w * stride_w;

                    /* Calculate average in kernel window */
                    float sum = 0.0f;
                    for (kh = 0; kh < kernel_h; kh++) {
                        for (kw = 0; kw < kernel_w; kw++) {
                            uint32_t src_idx = c * in_h * in_w + (in_h_start + kh) * in_w + (in_w_start + kw);
                            sum += npu_fp16_to_fp32(sh[src_idx]);
                        }
                    }

                    uint32_t dst_idx = c * out_h_dim * out_w_dim + out_h * out_w_dim + out_w;
                    dh[dst_idx] = npu_fp32_to_fp16(sum / pool_size);
                }
            }
        }
    } else {
        const float *sf = (const float *)src;
        float *df = (float *)dst;

        for (c = 0; c < in_c; c++) {
            for (out_h = 0; out_h < out_h_dim; out_h++) {
                for (out_w = 0; out_w < out_w_dim; out_w++) {
                    uint16_t in_h_start = out_h * stride_h;
                    uint16_t in_w_start = out_w * stride_w;

                    /* Calculate average in kernel window */
                    float sum = 0.0f;
                    for (kh = 0; kh < kernel_h; kh++) {
                        for (kw = 0; kw < kernel_w; kw++) {
                            uint32_t src_idx = c * in_h * in_w + (in_h_start + kh) * in_w + (in_w_start + kw);
                            sum += sf[src_idx];
                        }
                    }

                    uint32_t dst_idx = c * out_h_dim * out_w_dim + out_h * out_w_dim + out_w;
                    df[dst_idx] = sum / pool_size;
                }
            }
        }
    }

    return 0;
}

int npu_compute_batchnorm(const void* src, const void* mean, const void* var,
                          const void* gamma, const void* beta, void* dst,
                          uint32_t num_channels, uint32_t spatial_size,
                          float epsilon, uint8_t flags)
{
    uint32_t c, s;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;
    int affine = (flags & 0x02) != 0;  /* bit 1 = affine flag */

    if (!src || !mean || !var || !dst)
        return -1;
    if (affine && (!gamma || !beta))
        return -1;
    if (num_channels == 0 || spatial_size == 0)
        return -2;

    if (fp16) {
        const uint16_t *sh = (const uint16_t *)src;
        const uint16_t *mh = (const uint16_t *)mean;
        const uint16_t *vh = (const uint16_t *)var;
        const uint16_t *gh = affine ? (const uint16_t *)gamma : NULL;
        const uint16_t *bh = affine ? (const uint16_t *)beta : NULL;
        uint16_t *dh = (uint16_t *)dst;

        for (c = 0; c < num_channels; c++) {
            float m = npu_fp16_to_fp32(mh[c]);
            float v = npu_fp16_to_fp32(vh[c]);
            float g = affine ? npu_fp16_to_fp32(gh[c]) : 1.0f;
            float b = affine ? npu_fp16_to_fp32(bh[c]) : 0.0f;
            float std_inv = 1.0f / sqrtf(v + epsilon);

            for (s = 0; s < spatial_size; s++) {
                uint32_t idx = c * spatial_size + s;
                float x = npu_fp16_to_fp32(sh[idx]);
                float normalized = (x - m) * std_inv;
                float result = normalized * g + b;
                dh[idx] = npu_fp32_to_fp16(result);
            }
        }
    } else {
        const float *sf = (const float *)src;
        const float *mf = (const float *)mean;
        const float *vf = (const float *)var;
        const float *gf = affine ? (const float *)gamma : NULL;
        const float *bf = affine ? (const float *)beta : NULL;
        float *df = (float *)dst;

        for (c = 0; c < num_channels; c++) {
            float m = mf[c];
            float v = vf[c];
            float g = affine ? gf[c] : 1.0f;
            float b = affine ? bf[c] : 0.0f;
            float std_inv = 1.0f / sqrtf(v + epsilon);

            for (s = 0; s < spatial_size; s++) {
                uint32_t idx = c * spatial_size + s;
                float x = sf[idx];
                float normalized = (x - m) * std_inv;
                float result = normalized * g + b;
                df[idx] = result;
            }
        }
    }

    return 0;
}

int npu_compute_layernorm(const void* src, const void* gamma, const void* beta,
                          void* dst, uint32_t batch_size, uint32_t normalized_shape,
                          float epsilon, uint8_t flags)
{
    uint32_t b, i;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;
    int affine = (flags & 0x02) != 0;  /* bit 1 = affine flag */

    if (!src || !dst)
        return -1;
    if (affine && (!gamma || !beta))
        return -1;
    if (batch_size == 0 || normalized_shape == 0)
        return -2;

    if (fp16) {
        const uint16_t *sh = (const uint16_t *)src;
        const uint16_t *gh = affine ? (const uint16_t *)gamma : NULL;
        const uint16_t *bh = affine ? (const uint16_t *)beta : NULL;
        uint16_t *dh = (uint16_t *)dst;

        for (b = 0; b < batch_size; b++) {
            /* Calculate mean */
            float sum = 0.0f;
            for (i = 0; i < normalized_shape; i++) {
                sum += npu_fp16_to_fp32(sh[b * normalized_shape + i]);
            }
            float mean = sum / normalized_shape;

            /* Calculate variance */
            float var_sum = 0.0f;
            for (i = 0; i < normalized_shape; i++) {
                float x = npu_fp16_to_fp32(sh[b * normalized_shape + i]);
                float diff = x - mean;
                var_sum += diff * diff;
            }
            float variance = var_sum / normalized_shape;
            float std_inv = 1.0f / sqrtf(variance + epsilon);

            /* Normalize and apply affine transform */
            for (i = 0; i < normalized_shape; i++) {
                float x = npu_fp16_to_fp32(sh[b * normalized_shape + i]);
                float normalized = (x - mean) * std_inv;
                float g = affine ? npu_fp16_to_fp32(gh[i]) : 1.0f;
                float b_val = affine ? npu_fp16_to_fp32(bh[i]) : 0.0f;
                float result = normalized * g + b_val;
                dh[b * normalized_shape + i] = npu_fp32_to_fp16(result);
            }
        }
    } else {
        const float *sf = (const float *)src;
        const float *gf = affine ? (const float *)gamma : NULL;
        const float *bf = affine ? (const float *)beta : NULL;
        float *df = (float *)dst;

        for (b = 0; b < batch_size; b++) {
            /* Calculate mean */
            float sum = 0.0f;
            for (i = 0; i < normalized_shape; i++) {
                sum += sf[b * normalized_shape + i];
            }
            float mean = sum / normalized_shape;

            /* Calculate variance */
            float var_sum = 0.0f;
            for (i = 0; i < normalized_shape; i++) {
                float x = sf[b * normalized_shape + i];
                float diff = x - mean;
                var_sum += diff * diff;
            }
            float variance = var_sum / normalized_shape;
            float std_inv = 1.0f / sqrtf(variance + epsilon);

            /* Normalize and apply affine transform */
            for (i = 0; i < normalized_shape; i++) {
                float x = sf[b * normalized_shape + i];
                float normalized = (x - mean) * std_inv;
                float g = affine ? gf[i] : 1.0f;
                float b_val = affine ? bf[i] : 0.0f;
                float result = normalized * g + b_val;
                df[b * normalized_shape + i] = result;
            }
        }
    }

    return 0;
}
/* Convolution operations - appended to npu_compute.c */

int npu_compute_conv2d(const void* input, const void* weight, const void* bias,
                       void* dst, uint16_t in_c, uint16_t in_h, uint16_t in_w,
                       uint16_t out_c, uint8_t kernel_h, uint8_t kernel_w,
                       uint8_t pad_h, uint8_t pad_w,
                       uint8_t stride_h, uint8_t stride_w, uint8_t flags)
{
    uint16_t oc, oh, ow, ic, kh, kw;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;
    int relu_fused = (flags & 0x02) != 0;  /* bit 1 = relu_fused */
    int bias_en = (flags & 0x04) != 0;     /* bit 2 = bias_en */

    if (!input || !weight || !dst)
        return -1;
    if (bias_en && !bias)
        return -1;
    if (in_c == 0 || in_h == 0 || in_w == 0 || out_c == 0)
        return -2;
    if (kernel_h == 0 || kernel_w == 0)
        return -2;

    /* Calculate output dimensions */
    uint16_t out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    uint16_t out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    if (fp16) {
        const uint16_t *ih = (const uint16_t *)input;
        const uint16_t *wh = (const uint16_t *)weight;
        const uint16_t *bh = bias_en ? (const uint16_t *)bias : NULL;
        uint16_t *dh = (uint16_t *)dst;

        for (oc = 0; oc < out_c; oc++) {
            for (oh = 0; oh < out_h; oh++) {
                for (ow = 0; ow < out_w; ow++) {
                    float sum = bias_en ? npu_fp16_to_fp32(bh[oc]) : 0.0f;

                    /* Convolution operation */
                    for (ic = 0; ic < in_c; ic++) {
                        for (kh = 0; kh < kernel_h; kh++) {
                            for (kw = 0; kw < kernel_w; kw++) {
                                int16_t in_h_idx = (int16_t)(oh * stride_h) - pad_h + kh;
                                int16_t in_w_idx = (int16_t)(ow * stride_w) - pad_w + kw;

                                /* Zero padding */
                                if (in_h_idx < 0 || in_h_idx >= in_h ||
                                    in_w_idx < 0 || in_w_idx >= in_w) {
                                    continue;
                                }

                                uint32_t input_idx = ic * in_h * in_w + in_h_idx * in_w + in_w_idx;
                                uint32_t weight_idx = oc * in_c * kernel_h * kernel_w +
                                                     ic * kernel_h * kernel_w +
                                                     kh * kernel_w + kw;

                                sum += npu_fp16_to_fp32(ih[input_idx]) *
                                      npu_fp16_to_fp32(wh[weight_idx]);
                            }
                        }
                    }

                    /* Apply ReLU if fused */
                    if (relu_fused && sum < 0.0f)
                        sum = 0.0f;

                    uint32_t out_idx = oc * out_h * out_w + oh * out_w + ow;
                    dh[out_idx] = npu_fp32_to_fp16(sum);
                }
            }
        }
    } else {
        const float *ifl = (const float *)input;
        const float *wfl = (const float *)weight;
        const float *bfl = bias_en ? (const float *)bias : NULL;
        float *dfl = (float *)dst;

        for (oc = 0; oc < out_c; oc++) {
            for (oh = 0; oh < out_h; oh++) {
                for (ow = 0; ow < out_w; ow++) {
                    float sum = bias_en ? bfl[oc] : 0.0f;

                    /* Convolution operation */
                    for (ic = 0; ic < in_c; ic++) {
                        for (kh = 0; kh < kernel_h; kh++) {
                            for (kw = 0; kw < kernel_w; kw++) {
                                int16_t in_h_idx = (int16_t)(oh * stride_h) - pad_h + kh;
                                int16_t in_w_idx = (int16_t)(ow * stride_w) - pad_w + kw;

                                /* Zero padding */
                                if (in_h_idx < 0 || in_h_idx >= in_h ||
                                    in_w_idx < 0 || in_w_idx >= in_w) {
                                    continue;
                                }

                                uint32_t input_idx = ic * in_h * in_w + in_h_idx * in_w + in_w_idx;
                                uint32_t weight_idx = oc * in_c * kernel_h * kernel_w +
                                                     ic * kernel_h * kernel_w +
                                                     kh * kernel_w + kw;

                                sum += ifl[input_idx] * wfl[weight_idx];
                            }
                        }
                    }

                    /* Apply ReLU if fused */
                    if (relu_fused && sum < 0.0f)
                        sum = 0.0f;

                    uint32_t out_idx = oc * out_h * out_w + oh * out_w + ow;
                    dfl[out_idx] = sum;
                }
            }
        }
    }

    return 0;
}

int npu_compute_depthwise_conv(const void* input, const void* weight, void* dst,
                               uint16_t channels, uint16_t in_h, uint16_t in_w,
                               uint8_t kernel_h, uint8_t kernel_w,
                               uint8_t pad_h, uint8_t pad_w,
                               uint8_t stride_h, uint8_t stride_w, uint8_t flags)
{
    uint16_t c, oh, ow;
    uint8_t kh, kw;
    int fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;
    int relu_fused = (flags & 0x02) != 0;  /* bit 1 = relu_fused */

    if (!input || !weight || !dst)
        return -1;
    if (channels == 0 || in_h == 0 || in_w == 0)
        return -2;
    if (kernel_h == 0 || kernel_w == 0)
        return -2;

    /* Calculate output dimensions */
    uint16_t out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    uint16_t out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    if (fp16) {
        const uint16_t *ih = (const uint16_t *)input;
        const uint16_t *wh = (const uint16_t *)weight;
        uint16_t *dh = (uint16_t *)dst;

        /* Depthwise: each channel has its own kernel */
        for (c = 0; c < channels; c++) {
            for (oh = 0; oh < out_h; oh++) {
                for (ow = 0; ow < out_w; ow++) {
                    float sum = 0.0f;

                    /* Apply kernel to single channel */
                    for (kh = 0; kh < kernel_h; kh++) {
                        for (kw = 0; kw < kernel_w; kw++) {
                            int16_t in_h_idx = (int16_t)(oh * stride_h) - pad_h + kh;
                            int16_t in_w_idx = (int16_t)(ow * stride_w) - pad_w + kw;

                            /* Zero padding */
                            if (in_h_idx < 0 || in_h_idx >= in_h ||
                                in_w_idx < 0 || in_w_idx >= in_w) {
                                continue;
                            }

                            uint32_t input_idx = c * in_h * in_w + in_h_idx * in_w + in_w_idx;
                            uint32_t weight_idx = c * kernel_h * kernel_w + kh * kernel_w + kw;

                            sum += npu_fp16_to_fp32(ih[input_idx]) *
                                  npu_fp16_to_fp32(wh[weight_idx]);
                        }
                    }

                    /* Apply ReLU if fused */
                    if (relu_fused && sum < 0.0f)
                        sum = 0.0f;

                    uint32_t out_idx = c * out_h * out_w + oh * out_w + ow;
                    dh[out_idx] = npu_fp32_to_fp16(sum);
                }
            }
        }
    } else {
        const float *ifl = (const float *)input;
        const float *wfl = (const float *)weight;
        float *dfl = (float *)dst;

        /* Depthwise: each channel has its own kernel */
        for (c = 0; c < channels; c++) {
            for (oh = 0; oh < out_h; oh++) {
                for (ow = 0; ow < out_w; ow++) {
                    float sum = 0.0f;

                    /* Apply kernel to single channel */
                    for (kh = 0; kh < kernel_h; kh++) {
                        for (kw = 0; kw < kernel_w; kw++) {
                            int16_t in_h_idx = (int16_t)(oh * stride_h) - pad_h + kh;
                            int16_t in_w_idx = (int16_t)(ow * stride_w) - pad_w + kw;

                            /* Zero padding */
                            if (in_h_idx < 0 || in_h_idx >= in_h ||
                                in_w_idx < 0 || in_w_idx >= in_w) {
                                continue;
                            }

                            uint32_t input_idx = c * in_h * in_w + in_h_idx * in_w + in_w_idx;
                            uint32_t weight_idx = c * kernel_h * kernel_w + kh * kernel_w + kw;

                            sum += ifl[input_idx] * wfl[weight_idx];
                        }
                    }

                    /* Apply ReLU if fused */
                    if (relu_fused && sum < 0.0f)
                        sum = 0.0f;

                    uint32_t out_idx = c * out_h * out_w + oh * out_w + ow;
                    dfl[out_idx] = sum;
                }
            }
        }
    }

    return 0;
}

/* ========================================
 * FMADD: dst[i] = a[i] * b[i] + c[i]
 * ======================================== */

int npu_compute_fmadd(const void* src_a, const void* src_b, const void* src_c,
                      void* dst, uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    if (!src_a || !src_b || !src_c || !dst)
        return -1;
    if (num_elements == 0)
        return -2;

    if (flags & NPU_COMPUTE_FLAG_FP16) {
        const uint16_t *a = (const uint16_t *)src_a;
        const uint16_t *b = (const uint16_t *)src_b;
        const uint16_t *c = (const uint16_t *)src_c;
        uint16_t *d = (uint16_t *)dst;
        for (i = 0; i < num_elements; i++) {
            float va = npu_fp16_to_fp32(a[i]);
            float vb = npu_fp16_to_fp32(b[i]);
            float vc = npu_fp16_to_fp32(c[i]);
            d[i] = npu_fp32_to_fp16(va * vb + vc);
        }
    } else {
        const float *a = (const float *)src_a;
        const float *b = (const float *)src_b;
        const float *c = (const float *)src_c;
        float *d = (float *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = a[i] * b[i] + c[i];
    }
    return 0;
}

/* ========================================
 * CLAMP: dst[i] = clamp(src[i], min, max)
 * ======================================== */

int npu_compute_clamp(const void* src, void* dst,
                      uint32_t num_elements, float min_val, float max_val,
                      uint8_t flags)
{
    uint32_t i;
    if (!src || !dst)
        return -1;
    if (num_elements == 0)
        return -2;

    if (flags & NPU_COMPUTE_FLAG_FP16) {
        const uint16_t *s = (const uint16_t *)src;
        uint16_t *d = (uint16_t *)dst;
        for (i = 0; i < num_elements; i++) {
            float v = npu_fp16_to_fp32(s[i]);
            if (v < min_val) v = min_val;
            if (v > max_val) v = max_val;
            d[i] = npu_fp32_to_fp16(v);
        }
    } else {
        const float *s = (const float *)src;
        float *d = (float *)dst;
        for (i = 0; i < num_elements; i++) {
            float v = s[i];
            if (v < min_val) v = min_val;
            if (v > max_val) v = max_val;
            d[i] = v;
        }
    }
    return 0;
}

/* ========================================
 * EXP: dst[i] = exp(src[i])
 * ======================================== */

int npu_compute_exp(const void* src, void* dst,
                    uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    if (!src || !dst)
        return -1;
    if (num_elements == 0)
        return -2;

    if (flags & NPU_COMPUTE_FLAG_FP16) {
        const uint16_t *s = (const uint16_t *)src;
        uint16_t *d = (uint16_t *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = npu_fp32_to_fp16(expf(npu_fp16_to_fp32(s[i])));
    } else {
        const float *s = (const float *)src;
        float *d = (float *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = expf(s[i]);
    }
    return 0;
}

/* ========================================
 * LOG: dst[i] = log(src[i])
 * ======================================== */

int npu_compute_log(const void* src, void* dst,
                    uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    if (!src || !dst)
        return -1;
    if (num_elements == 0)
        return -2;

    if (flags & NPU_COMPUTE_FLAG_FP16) {
        const uint16_t *s = (const uint16_t *)src;
        uint16_t *d = (uint16_t *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = npu_fp32_to_fp16(logf(npu_fp16_to_fp32(s[i])));
    } else {
        const float *s = (const float *)src;
        float *d = (float *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = logf(s[i]);
    }
    return 0;
}

/* ========================================
 * SQRT: dst[i] = sqrt(src[i])
 * ======================================== */

int npu_compute_sqrt(const void* src, void* dst,
                     uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    if (!src || !dst)
        return -1;
    if (num_elements == 0)
        return -2;

    if (flags & NPU_COMPUTE_FLAG_FP16) {
        const uint16_t *s = (const uint16_t *)src;
        uint16_t *d = (uint16_t *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = npu_fp32_to_fp16(sqrtf(npu_fp16_to_fp32(s[i])));
    } else {
        const float *s = (const float *)src;
        float *d = (float *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = sqrtf(s[i]);
    }
    return 0;
}

/* ========================================
 * RSQRT: dst[i] = 1/sqrt(src[i])
 * ======================================== */

int npu_compute_rsqrt(const void* src, void* dst,
                      uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    if (!src || !dst)
        return -1;
    if (num_elements == 0)
        return -2;

    if (flags & NPU_COMPUTE_FLAG_FP16) {
        const uint16_t *s = (const uint16_t *)src;
        uint16_t *d = (uint16_t *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = npu_fp32_to_fp16(1.0f / sqrtf(npu_fp16_to_fp32(s[i])));
    } else {
        const float *s = (const float *)src;
        float *d = (float *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = 1.0f / sqrtf(s[i]);
    }
    return 0;
}

/* ========================================
 * ABS: dst[i] = |src[i]|
 * ======================================== */

int npu_compute_abs(const void* src, void* dst,
                    uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    if (!src || !dst)
        return -1;
    if (num_elements == 0)
        return -2;

    if (flags & NPU_COMPUTE_FLAG_FP16) {
        const uint16_t *s = (const uint16_t *)src;
        uint16_t *d = (uint16_t *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = npu_fp32_to_fp16(fabsf(npu_fp16_to_fp32(s[i])));
    } else {
        const float *s = (const float *)src;
        float *d = (float *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = fabsf(s[i]);
    }
    return 0;
}

/* ========================================
 * NEG: dst[i] = -src[i]
 * ======================================== */

int npu_compute_neg(const void* src, void* dst,
                    uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    if (!src || !dst)
        return -1;
    if (num_elements == 0)
        return -2;

    if (flags & NPU_COMPUTE_FLAG_FP16) {
        const uint16_t *s = (const uint16_t *)src;
        uint16_t *d = (uint16_t *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = npu_fp32_to_fp16(-npu_fp16_to_fp32(s[i]));
    } else {
        const float *s = (const float *)src;
        float *d = (float *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = -s[i];
    }
    return 0;
}

/* ========================================
 * SWISH: dst[i] = src[i] * sigmoid(src[i])
 * ======================================== */

int npu_compute_swish(const void* src, void* dst,
                      uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    if (!src || !dst)
        return -1;
    if (num_elements == 0)
        return -2;

    if (flags & NPU_COMPUTE_FLAG_FP16) {
        const uint16_t *s = (const uint16_t *)src;
        uint16_t *d = (uint16_t *)dst;
        for (i = 0; i < num_elements; i++) {
            float x = npu_fp16_to_fp32(s[i]);
            d[i] = npu_fp32_to_fp16(x / (1.0f + expf(-x)));
        }
    } else {
        const float *s = (const float *)src;
        float *d = (float *)dst;
        for (i = 0; i < num_elements; i++)
            d[i] = s[i] / (1.0f + expf(-s[i]));
    }
    return 0;
}

/* ========================================
 * MISH: dst[i] = src[i] * tanh(softplus(src[i]))
 * ======================================== */

int npu_compute_mish(const void* src, void* dst,
                     uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    if (!src || !dst)
        return -1;
    if (num_elements == 0)
        return -2;

    if (flags & NPU_COMPUTE_FLAG_FP16) {
        const uint16_t *s = (const uint16_t *)src;
        uint16_t *d = (uint16_t *)dst;
        for (i = 0; i < num_elements; i++) {
            float x = npu_fp16_to_fp32(s[i]);
            float sp = logf(1.0f + expf(x));
            d[i] = npu_fp32_to_fp16(x * tanhf(sp));
        }
    } else {
        const float *s = (const float *)src;
        float *d = (float *)dst;
        for (i = 0; i < num_elements; i++) {
            float sp = logf(1.0f + expf(s[i]));
            d[i] = s[i] * tanhf(sp);
        }
    }
    return 0;
}

/* ========================================
 * GEMM: C = alpha * A @ B + beta * C
 * ======================================== */

int npu_compute_gemm(const void* A, const void* B, void* C,
                     uint32_t m, uint32_t n, uint32_t k,
                     float alpha, float beta, uint8_t flags)
{
    uint32_t i, j, p;
    int transpose_a = (flags & 0x02) != 0;
    int transpose_b = (flags & 0x04) != 0;
    int fp16        = (flags & NPU_COMPUTE_FLAG_FP16) != 0;

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
                for (p = 0; p < k; p++)
                    sum += mat_get_fp16(Ah, i, p, a_cols, transpose_a) *
                           mat_get_fp16(Bh, p, j, b_cols, transpose_b);
                float old_c = npu_fp16_to_fp32(Ch[i * n + j]);
                Ch[i * n + j] = npu_fp32_to_fp16(alpha * sum + beta * old_c);
            }
        }
    } else {
        const float *Af = (const float *)A;
        const float *Bf = (const float *)B;
        float *Cf = (float *)C;

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                float sum = 0.0f;
                for (p = 0; p < k; p++)
                    sum += mat_get_fp32(Af, i, p, a_cols, transpose_a) *
                           mat_get_fp32(Bf, p, j, b_cols, transpose_b);
                Cf[i * n + j] = alpha * sum + beta * Cf[i * n + j];
            }
        }
    }
    return 0;
}

/* ========================================
 * DOT: dst = sum(a[i] * b[i])
 * ======================================== */

int npu_compute_dot(const void* src_a, const void* src_b, void* dst,
                    uint32_t num_elements, uint8_t flags)
{
    uint32_t i;
    if (!src_a || !src_b || !dst)
        return -1;
    if (num_elements == 0)
        return -2;

    float sum = 0.0f;
    if (flags & NPU_COMPUTE_FLAG_FP16) {
        const uint16_t *a = (const uint16_t *)src_a;
        const uint16_t *b = (const uint16_t *)src_b;
        for (i = 0; i < num_elements; i++)
            sum += npu_fp16_to_fp32(a[i]) * npu_fp16_to_fp32(b[i]);
        *(uint16_t *)dst = npu_fp32_to_fp16(sum);
    } else {
        const float *a = (const float *)src_a;
        const float *b = (const float *)src_b;
        for (i = 0; i < num_elements; i++)
            sum += a[i] * b[i];
        *(float *)dst = sum;
    }
    return 0;
}

/* ========================================
 * RMSNORM: dst = src / rms(src) * weight
 * ======================================== */

int npu_compute_rmsnorm(const void* src, const void* weight, void* dst,
                        uint32_t batch_size, uint32_t normalized_shape,
                        float epsilon, uint8_t flags)
{
    uint32_t b, i;
    if (!src || !dst)
        return -1;

    const float *sf = (const float *)src;
    const float *wf = (const float *)weight;
    float *df = (float *)dst;

    for (b = 0; b < batch_size; b++) {
        const float *row = &sf[b * normalized_shape];
        float *out = &df[b * normalized_shape];

        float sum_sq = 0.0f;
        for (i = 0; i < normalized_shape; i++)
            sum_sq += row[i] * row[i];
        float rms = sqrtf(sum_sq / normalized_shape + epsilon);

        for (i = 0; i < normalized_shape; i++) {
            float val = row[i] / rms;
            if (wf)
                val *= wf[i];
            out[i] = val;
        }
    }
    return 0;
}

/* ========================================
 * GROUPNORM: normalize within channel groups
 * ======================================== */

int npu_compute_groupnorm(const void* src, const void* gamma, const void* beta,
                          void* dst, uint32_t num_groups, uint32_t num_channels,
                          uint32_t spatial_size, float epsilon, uint8_t flags)
{
    uint32_t g, c, s;
    if (!src || !dst)
        return -1;

    const float *sf = (const float *)src;
    const float *gf = (const float *)gamma;
    const float *bf = (const float *)beta;
    float *df = (float *)dst;
    int affine = (flags & 0x02) != 0;

    uint32_t channels_per_group = num_channels / num_groups;
    uint32_t group_size = channels_per_group * spatial_size;

    for (g = 0; g < num_groups; g++) {
        uint32_t group_start = g * group_size;

        float mean = 0.0f;
        for (s = 0; s < group_size; s++)
            mean += sf[group_start + s];
        mean /= group_size;

        float var = 0.0f;
        for (s = 0; s < group_size; s++) {
            float diff = sf[group_start + s] - mean;
            var += diff * diff;
        }
        var /= group_size;

        float inv_std = 1.0f / sqrtf(var + epsilon);

        for (c = 0; c < channels_per_group; c++) {
            uint32_t ch_idx = g * channels_per_group + c;
            for (s = 0; s < spatial_size; s++) {
                uint32_t idx = group_start + c * spatial_size + s;
                float val = (sf[idx] - mean) * inv_std;
                if (affine && gf && bf)
                    val = val * gf[ch_idx] + bf[ch_idx];
                df[idx] = val;
            }
        }
    }
    return 0;
}

/* ========================================
 * INSTANCENORM: normalize per channel
 * ======================================== */

int npu_compute_instancenorm(const void* src, const void* gamma, const void* beta,
                             void* dst, uint32_t num_channels, uint32_t spatial_size,
                             float epsilon, uint8_t flags)
{
    uint32_t c, s;
    if (!src || !dst)
        return -1;

    const float *sf = (const float *)src;
    const float *gf = (const float *)gamma;
    const float *bf = (const float *)beta;
    float *df = (float *)dst;
    int affine = (flags & 0x02) != 0;

    for (c = 0; c < num_channels; c++) {
        uint32_t ch_start = c * spatial_size;

        float mean = 0.0f;
        for (s = 0; s < spatial_size; s++)
            mean += sf[ch_start + s];
        mean /= spatial_size;

        float var = 0.0f;
        for (s = 0; s < spatial_size; s++) {
            float diff = sf[ch_start + s] - mean;
            var += diff * diff;
        }
        var /= spatial_size;

        float inv_std = 1.0f / sqrtf(var + epsilon);

        for (s = 0; s < spatial_size; s++) {
            float val = (sf[ch_start + s] - mean) * inv_std;
            if (affine && gf && bf)
                val = val * gf[c] + bf[c];
            df[ch_start + s] = val;
        }
    }
    return 0;
}

/* ========================================
 * SCALED_DOT_PRODUCT_ATTENTION
 * dst = softmax(Q @ K^T / scale) @ V
 * ======================================== */

int npu_compute_sdpa(const void* Q, const void* K, const void* V, void* dst,
                     uint32_t num_heads, uint32_t seq_len_q, uint32_t seq_len_kv,
                     uint32_t head_dim, float scale, uint8_t flags)
{
    uint32_t h, i, j, d;
    int causal = (flags & 0x02) != 0;

    if (!Q || !K || !V || !dst)
        return -1;

    const float *Qf = (const float *)Q;
    const float *Kf = (const float *)K;
    const float *Vf = (const float *)V;
    float *Df = (float *)dst;

    uint32_t qk_stride = seq_len_q * head_dim;
    uint32_t kv_stride = seq_len_kv * head_dim;

    for (h = 0; h < num_heads; h++) {
        const float *q = &Qf[h * qk_stride];
        const float *k = &Kf[h * kv_stride];
        const float *v = &Vf[h * kv_stride];
        float *out = &Df[h * qk_stride];

        for (i = 0; i < seq_len_q; i++) {
            float scores[1024];
            float max_score = -1e30f;

            for (j = 0; j < seq_len_kv; j++) {
                float dot = 0.0f;
                for (d = 0; d < head_dim; d++)
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                dot *= scale;

                if (causal && j > i)
                    dot = -1e30f;

                scores[j] = dot;
                if (dot > max_score)
                    max_score = dot;
            }

            float sum_exp = 0.0f;
            for (j = 0; j < seq_len_kv; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum_exp += scores[j];
            }
            for (j = 0; j < seq_len_kv; j++)
                scores[j] /= sum_exp;

            for (d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (j = 0; j < seq_len_kv; j++)
                    val += scores[j] * v[j * head_dim + d];
                out[i * head_dim + d] = val;
            }
        }
    }
    return 0;
}

/* ========================================
 * CAST: convert between data types
 * dtype: 0=fp32 1=fp16 2=int8 3=int32
 * ======================================== */

int npu_compute_cast(const void* src, void* dst,
                     uint32_t num_elements, uint8_t src_dtype, uint8_t dst_dtype)
{
    uint32_t i;
    if (!src || !dst)
        return -1;
    if (num_elements == 0)
        return -2;
    if (src_dtype == dst_dtype) {
        uint32_t elem_sizes[] = {4, 2, 1, 4};
        memcpy(dst, src, (size_t)num_elements * elem_sizes[src_dtype]);
        return 0;
    }

    for (i = 0; i < num_elements; i++) {
        float val;
        switch (src_dtype) {
            case 0: val = ((const float *)src)[i]; break;
            case 1: val = npu_fp16_to_fp32(((const uint16_t *)src)[i]); break;
            case 2: val = (float)((const int8_t *)src)[i]; break;
            case 3: val = (float)((const int32_t *)src)[i]; break;
            default: return -1;
        }
        switch (dst_dtype) {
            case 0: ((float *)dst)[i] = val; break;
            case 1: ((uint16_t *)dst)[i] = npu_fp32_to_fp16(val); break;
            case 2: {
                int v = (int)(val + (val >= 0 ? 0.5f : -0.5f));
                if (v > 127) v = 127;
                if (v < -128) v = -128;
                ((int8_t *)dst)[i] = (int8_t)v;
                break;
            }
            case 3: ((int32_t *)dst)[i] = (int32_t)val; break;
            default: return -1;
        }
    }
    return 0;
}

/* ========================================
 * QUANTIZE: fp32 -> int8
 * ======================================== */

int npu_compute_quantize(const void* src, void* dst,
                         const void* scale, const void* zero_point,
                         uint32_t num_elements, uint32_t num_channels,
                         uint8_t flags)
{
    uint32_t i;
    int per_channel = (flags & 0x01) != 0;

    if (!src || !dst || !scale || !zero_point)
        return -1;

    const float *sf = (const float *)src;
    int8_t *di = (int8_t *)dst;
    const float *sc = (const float *)scale;
    const int8_t *zp = (const int8_t *)zero_point;

    uint32_t elements_per_channel = per_channel ? num_elements / num_channels : 0;

    for (i = 0; i < num_elements; i++) {
        uint32_t ch = per_channel ? (i / elements_per_channel) : 0;
        float s = sc[ch];
        int z = zp[ch];
        int v = (int)roundf(sf[i] / s) + z;
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        di[i] = (int8_t)v;
    }
    return 0;
}

/* ========================================
 * DEQUANTIZE: int8 -> fp32
 * ======================================== */

int npu_compute_dequantize(const void* src, void* dst,
                           const void* scale, const void* zero_point,
                           uint32_t num_elements, uint32_t num_channels,
                           uint8_t flags)
{
    uint32_t i;
    int per_channel = (flags & 0x01) != 0;

    if (!src || !dst || !scale || !zero_point)
        return -1;

    const int8_t *si = (const int8_t *)src;
    float *df = (float *)dst;
    const float *sc = (const float *)scale;
    const int8_t *zp = (const int8_t *)zero_point;

    uint32_t elements_per_channel = per_channel ? num_elements / num_channels : 0;

    for (i = 0; i < num_elements; i++) {
        uint32_t ch = per_channel ? (i / elements_per_channel) : 0;
        df[i] = ((float)si[i] - (float)zp[ch]) * sc[ch];
    }
    return 0;
}

/* ========================================
 * Data Movement / Manipulation Operations
 * ======================================== */

int npu_compute_gather(const void* src, const void* indices, void* dst,
                       uint32_t outer_size, uint32_t gather_size,
                       uint32_t inner_size, uint32_t num_indices, uint8_t flags)
{
    int is_fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;
    uint32_t elem_size = is_fp16 ? 2 : 4;
    uint32_t o, idx_i;

    if (!src || !indices || !dst)
        return -1;

    const uint8_t *sp = (const uint8_t *)src;
    const uint32_t *ip = (const uint32_t *)indices;
    uint8_t *dp = (uint8_t *)dst;

    for (o = 0; o < outer_size; o++) {
        for (idx_i = 0; idx_i < num_indices; idx_i++) {
            uint32_t gather_idx = ip[o * num_indices + idx_i];
            if (gather_idx >= gather_size)
                return -1;

            uint32_t src_off = (o * gather_size + gather_idx) * inner_size * elem_size;
            uint32_t dst_off = (o * num_indices + idx_i) * inner_size * elem_size;

            memcpy(&dp[dst_off], &sp[src_off], inner_size * elem_size);
        }
    }
    return 0;
}

int npu_compute_slice(const void* src, void* dst,
                      uint32_t outer_size, uint32_t src_axis_size,
                      uint32_t inner_size, uint32_t start,
                      uint32_t length, uint32_t step, uint8_t flags)
{
    int is_fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;
    uint32_t elem_size = is_fp16 ? 2 : 4;
    uint32_t o, s, dst_idx;

    if (!src || !dst)
        return -1;
    if (step == 0)
        return -1;

    const uint8_t *sp = (const uint8_t *)src;
    uint8_t *dp = (uint8_t *)dst;

    for (o = 0; o < outer_size; o++) {
        dst_idx = 0;
        for (s = 0; s < length; s++) {
            uint32_t src_pos = start + s * step;
            if (src_pos >= src_axis_size)
                break;

            uint32_t src_off = (o * src_axis_size + src_pos) * inner_size * elem_size;
            uint32_t dst_off = (o * length + dst_idx) * inner_size * elem_size;

            memcpy(&dp[dst_off], &sp[src_off], inner_size * elem_size);
            dst_idx++;
        }
    }
    return 0;
}

int npu_compute_pad(const void* src, void* dst,
                    const uint16_t* src_dims, const uint16_t* pad_before,
                    const uint16_t* pad_after, uint8_t num_dims,
                    uint32_t pad_value_bits, uint8_t flags)
{
    int is_fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;
    uint32_t elem_size = is_fp16 ? 2 : 4;
    uint32_t dst_dims[4], dst_total;
    uint32_t d, i;

    if (!src || !dst || !src_dims || !pad_before || !pad_after)
        return -1;
    if (num_dims == 0 || num_dims > 4)
        return -1;

    /* Calculate destination dimensions and total size */
    dst_total = 1;
    for (d = 0; d < num_dims; d++) {
        dst_dims[d] = pad_before[d] + src_dims[d] + pad_after[d];
        dst_total *= dst_dims[d];
    }

    /* Fill entire destination with pad value */
    if (is_fp16) {
        uint16_t pv = (uint16_t)(pad_value_bits & 0xFFFF);
        uint16_t *dp = (uint16_t *)dst;
        for (i = 0; i < dst_total; i++) dp[i] = pv;
    } else {
        uint32_t *dp = (uint32_t *)dst;
        for (i = 0; i < dst_total; i++) dp[i] = pad_value_bits;
    }

    /* Copy source data into padded region */
    /* Simplified for 1D-4D: iterate source and compute dst position */
    {
        const uint8_t *sp = (const uint8_t *)src;
        uint8_t *dp = (uint8_t *)dst;
        uint32_t src_strides[4] = {1, 1, 1, 1};
        uint32_t dst_strides[4] = {1, 1, 1, 1};

        for (d = num_dims - 1; d > 0; d--) {
            src_strides[d - 1] = src_strides[d] * src_dims[d];
            dst_strides[d - 1] = dst_strides[d] * dst_dims[d];
        }

        uint32_t src_total = 1;
        for (d = 0; d < num_dims; d++) src_total *= src_dims[d];

        for (i = 0; i < src_total; i++) {
            /* Convert flat index to per-dim coords */
            uint32_t src_off = 0, dst_off = 0, rem = i;
            for (d = 0; d < num_dims; d++) {
                uint32_t coord = rem / src_strides[d];
                rem = rem % src_strides[d];
                src_off += coord * src_strides[d];
                dst_off += (coord + pad_before[d]) * dst_strides[d];
            }
            memcpy(&dp[dst_off * elem_size], &sp[src_off * elem_size], elem_size);
        }
    }
    return 0;
}

int npu_compute_where(const void* cond, const void* true_val,
                      const void* false_val, void* dst,
                      uint32_t num_elements, uint8_t flags)
{
    int is_fp16 = (flags & NPU_COMPUTE_FLAG_FP16) != 0;
    uint32_t elem_size = is_fp16 ? 2 : 4;
    uint32_t i;

    if (!cond || !true_val || !false_val || !dst)
        return -1;

    const uint8_t *cp = (const uint8_t *)cond;
    const uint8_t *tp = (const uint8_t *)true_val;
    const uint8_t *fp = (const uint8_t *)false_val;
    uint8_t *dp = (uint8_t *)dst;

    for (i = 0; i < num_elements; i++) {
        const uint8_t *selected = cp[i] ? &tp[i * elem_size] : &fp[i * elem_size];
        memcpy(&dp[i * elem_size], selected, elem_size);
    }
    return 0;
}

/* ================================================================== */
/* KV Cache Operations                                                */
/* ================================================================== */

int npu_compute_kv_cache_append(const void* k_new, const void* v_new,
                                void* k_cache, void* v_cache,
                                uint32_t cur_seq_pos, uint16_t num_kv_heads,
                                uint16_t head_dim, uint32_t max_seq_len,
                                uint8_t flags)
{
    uint32_t h;
    uint32_t elem_size = (flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    uint32_t row_bytes = head_dim * elem_size;

    if (!k_new || !v_new || !k_cache || !v_cache)
        return -1;
    if (cur_seq_pos >= max_seq_len)
        return -2;

    /* Cache layout: (num_kv_heads, max_seq_len, head_dim) row-major */
    for (h = 0; h < num_kv_heads; h++) {
        uint32_t cache_offset = (h * max_seq_len + cur_seq_pos) * row_bytes;
        uint32_t new_offset = h * row_bytes;
        memcpy((uint8_t*)k_cache + cache_offset, (const uint8_t*)k_new + new_offset, row_bytes);
        memcpy((uint8_t*)v_cache + cache_offset, (const uint8_t*)v_new + new_offset, row_bytes);
    }
    return 0;
}

int npu_compute_kv_cache_attention(const void* Q, const void* k_cache,
                                   const void* v_cache, void* dst,
                                   uint16_t num_heads, uint16_t num_kv_heads,
                                   uint32_t cur_seq_len, uint16_t head_dim,
                                   uint16_t max_seq_len, float scale,
                                   uint8_t flags)
{
    uint32_t h, s, d;
    uint32_t gqa_ratio;
    float score, max_score, sum_exp;

    if (!Q || !k_cache || !v_cache || !dst)
        return -1;
    if (cur_seq_len == 0 || num_kv_heads == 0)
        return -2;

    gqa_ratio = num_heads / num_kv_heads;

    /* For each query head */
    for (h = 0; h < num_heads; h++) {
        uint32_t kv_h = h / gqa_ratio;  /* GQA mapping */

        /* Q: (num_heads, head_dim), single token */
        /* K cache: (num_kv_heads, max_seq_len, head_dim) */

        /* Compute attention scores: Q[h] @ K_cache[kv_h, :cur_seq_len]^T */
        float *scores = (float*)alloca(cur_seq_len * sizeof(float));

        max_score = -1e30f;
        for (s = 0; s < cur_seq_len; s++) {
            score = 0.0f;
            for (d = 0; d < (uint32_t)head_dim; d++) {
                float q_val = npu_fp16_to_fp32(((const uint16_t*)Q)[h * head_dim + d]);
                uint32_t k_idx = (kv_h * max_seq_len + s) * head_dim + d;
                float k_val = npu_fp16_to_fp32(((const uint16_t*)k_cache)[k_idx]);
                score += q_val * k_val;
            }
            score *= scale;
            scores[s] = score;
            if (score > max_score) max_score = score;
        }

        /* Softmax */
        sum_exp = 0.0f;
        for (s = 0; s < cur_seq_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum_exp += scores[s];
        }
        for (s = 0; s < cur_seq_len; s++) {
            scores[s] /= sum_exp;
        }

        /* Output = attn @ V_cache[kv_h, :cur_seq_len] */
        for (d = 0; d < (uint32_t)head_dim; d++) {
            float val = 0.0f;
            for (s = 0; s < cur_seq_len; s++) {
                uint32_t v_idx = (kv_h * max_seq_len + s) * head_dim + d;
                float v_val = npu_fp16_to_fp32(((const uint16_t*)v_cache)[v_idx]);
                val += scores[s] * v_val;
            }
            ((uint16_t*)dst)[h * head_dim + d] = npu_fp32_to_fp16(val);
        }
    }
    return 0;
}

int npu_compute_kv_cache_reset(void* k_cache, void* v_cache,
                               uint16_t num_kv_heads, uint16_t head_dim,
                               uint32_t max_seq_len, uint8_t flags)
{
    if (!k_cache || !v_cache)
        return -1;

    if (flags & 0x01) {  /* zero_memory flag */
        uint32_t total = num_kv_heads * max_seq_len * head_dim * 2;  /* fp16 */
        memset(k_cache, 0, total);
        memset(v_cache, 0, total);
    }
    return 0;
}

/* ================================================================== */
/* Token Processing Operations                                        */
/* ================================================================== */

int npu_compute_embedding_lookup(const void* table, void* dst,
                                 uint32_t token_id, uint32_t vocab_size,
                                 uint32_t embed_dim, uint8_t flags)
{
    uint32_t elem_size;

    if (!table || !dst)
        return -1;
    if (token_id >= vocab_size)
        return -2;

    elem_size = (flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    memcpy(dst, (const uint8_t*)table + token_id * embed_dim * elem_size,
           embed_dim * elem_size);
    return 0;
}

int npu_compute_token_sample(const void* logits, void* dst_token,
                             uint32_t vocab_size, uint8_t mode,
                             float temperature, uint16_t top_k,
                             uint32_t seed, uint8_t flags)
{
    uint32_t i, best_idx;
    float best_val;

    if (!logits || !dst_token)
        return -1;
    if (vocab_size == 0)
        return -2;

    /* Greedy (argmax) — sufficient for functional emulation */
    best_idx = 0;
    best_val = (flags & NPU_COMPUTE_FLAG_FP16)
        ? npu_fp16_to_fp32(((const uint16_t*)logits)[0])
        : ((const float*)logits)[0];

    for (i = 1; i < vocab_size; i++) {
        float val = (flags & NPU_COMPUTE_FLAG_FP16)
            ? npu_fp16_to_fp32(((const uint16_t*)logits)[i])
            : ((const float*)logits)[i];
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    /* Write token ID as u32 */
    *(uint32_t*)dst_token = best_idx;
    return 0;
}

int npu_compute_rotary_embedding(const void* src, void* dst,
                                 uint32_t position, uint16_t num_heads,
                                 uint16_t head_dim, float theta_base,
                                 uint8_t flags)
{
    uint32_t h, d;
    uint32_t half_dim;

    if (!src || !dst)
        return -1;
    if (head_dim % 2 != 0)
        return -2;

    half_dim = head_dim / 2;

    for (h = 0; h < num_heads; h++) {
        for (d = 0; d < half_dim; d++) {
            float freq = 1.0f / powf(theta_base, (float)(2 * d) / (float)head_dim);
            float angle = (float)position * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            uint32_t idx0 = h * head_dim + d * 2;
            uint32_t idx1 = h * head_dim + d * 2 + 1;

            float x0 = npu_fp16_to_fp32(((const uint16_t*)src)[idx0]);
            float x1 = npu_fp16_to_fp32(((const uint16_t*)src)[idx1]);

            ((uint16_t*)dst)[idx0] = npu_fp32_to_fp16(x0 * cos_a - x1 * sin_a);
            ((uint16_t*)dst)[idx1] = npu_fp32_to_fp16(x0 * sin_a + x1 * cos_a);
        }
    }
    return 0;
}
