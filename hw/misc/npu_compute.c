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

            /* Compute exp and sum */
            float sum = 0.0f;
            float exp_vals[axis_size];  /* VLA for temporary storage */
            for (i = 0; i < axis_size; i++) {
                float val = npu_fp16_to_fp32(vec_in[i]);
                exp_vals[i] = expf(val - max_val);
                sum += exp_vals[i];
            }

            /* Normalize */
            for (i = 0; i < axis_size; i++)
                vec_out[i] = npu_fp32_to_fp16(exp_vals[i] / sum);
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
