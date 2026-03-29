/*
 * NPU Instruction Execution Engine - Implementation
 *
 * Fetches 64-byte instructions from a program buffer,
 * decodes opcodes via npu_inst_t union, and dispatches
 * to compute kernels (hw/compute/) or DMA callbacks.
 *
 * Uses a direct dispatch table (256 entries) for O(1) opcode lookup.
 * Inspired by NVIDIA PTX and Intel AVX-512 instruction dispatch.
 */

#include "npu_engine.h"
#include "npu_spec.h"
#include "npu_compute.h"

#include <string.h>
#include <math.h>

/* ================================================================== */
/* Instruction Handler Context                                        */
/* ================================================================== */

typedef struct {
    npu_engine_t *engine;
    const npu_inst_t *inst;
    npu_dma_read_fn dma_read;
    npu_dma_write_fn dma_write;
    void *opaque;
} npu_exec_ctx_t;

/* Function pointer type for instruction handlers */
typedef int (*npu_handler_fn)(npu_exec_ctx_t *ctx);

void npu_engine_init(npu_engine_t *engine)
{
    memset(engine->sram, 0, NPU_SRAM_SIZE);
    engine->pc = 0;
    engine->halted = 0;
    engine->exit_code = 0;
    engine->error = 0;
}

void npu_engine_reset(npu_engine_t *engine)
{
    engine->pc = 0;
    engine->halted = 0;
    engine->exit_code = 0;
    engine->error = 0;
}

/* ================================================================== */
/* Helper Functions                                                   */
/* ================================================================== */

/* Check that [offset, offset+size) fits in SRAM */
static int sram_bounds_ok(uint64_t offset, uint32_t size)
{
    if (offset + size > NPU_SRAM_SIZE)
        return 0;
    return 1;
}

/* ================================================================== */
/* Instruction Handlers                                               */
/* ================================================================== */

static int handle_nop(npu_exec_ctx_t *ctx)
{
    (void)ctx;  /* Unused */
    /* NOP: no operation */
    return 0;
}

static int handle_halt(npu_exec_ctx_t *ctx)
{
    ctx->engine->halted = 1;
    ctx->engine->exit_code = ctx->inst->halt.exit_code;
    return 0;
}

static int handle_load(npu_exec_ctx_t *ctx)
{
    const npu_inst_load_t *inst = &ctx->inst->load;
    if (!sram_bounds_ok(inst->npu_addr, inst->size_bytes))
        return -1;
    return ctx->dma_read(ctx->opaque, inst->host_addr,
                         &ctx->engine->sram[inst->npu_addr], inst->size_bytes);
}

static int handle_store(npu_exec_ctx_t *ctx)
{
    const npu_inst_store_t *inst = &ctx->inst->store;
    if (!sram_bounds_ok(inst->npu_addr, inst->size_bytes))
        return -1;
    return ctx->dma_write(ctx->opaque, inst->host_addr,
                          &ctx->engine->sram[inst->npu_addr], inst->size_bytes);
}

static int handle_matmul(npu_exec_ctx_t *ctx)
{
    const npu_inst_matmul_t *inst = &ctx->inst->matmul;
    uint64_t a_off = inst->src_a_addr;
    uint64_t b_off = inst->src_b_addr;
    uint64_t c_off = inst->dst_addr;
    uint32_t m = inst->m;
    uint32_t n = inst->n;
    uint32_t k = inst->k;

    /* Bounds check: fp16=2 bytes, fp32=4 bytes per element */
    uint32_t elem_size = (inst->flags & NPU_MATMUL_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(a_off, m * k * elem_size))
        return -1;
    if (!sram_bounds_ok(b_off, k * n * elem_size))
        return -1;
    if (!sram_bounds_ok(c_off, m * n * elem_size))
        return -1;

    return npu_compute_matmul(&ctx->engine->sram[a_off], &ctx->engine->sram[b_off],
                              &ctx->engine->sram[c_off], m, n, k, inst->flags);
}

static int handle_relu(npu_exec_ctx_t *ctx)
{
    const npu_inst_relu_t *inst = &ctx->inst->relu;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_relu(&ctx->engine->sram[inst->src_addr],
                            &ctx->engine->sram[inst->dst_addr],
                            inst->num_elements, inst->flags);
}

static int handle_add(npu_exec_ctx_t *ctx)
{
    const npu_inst_add_t *inst = &ctx->inst->add;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_a_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->src_b_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_add(&ctx->engine->sram[inst->src_a_addr],
                           &ctx->engine->sram[inst->src_b_addr],
                           &ctx->engine->sram[inst->dst_addr],
                           inst->num_elements, inst->flags);
}

static int handle_sub(npu_exec_ctx_t *ctx)
{
    const npu_inst_sub_t *inst = &ctx->inst->sub;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_a_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->src_b_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_sub(&ctx->engine->sram[inst->src_a_addr],
                           &ctx->engine->sram[inst->src_b_addr],
                           &ctx->engine->sram[inst->dst_addr],
                           inst->num_elements, inst->flags);
}

static int handle_mul(npu_exec_ctx_t *ctx)
{
    const npu_inst_mul_t *inst = &ctx->inst->mul;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_a_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->src_b_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_mul(&ctx->engine->sram[inst->src_a_addr],
                           &ctx->engine->sram[inst->src_b_addr],
                           &ctx->engine->sram[inst->dst_addr],
                           inst->num_elements, inst->flags);
}

static int handle_div(npu_exec_ctx_t *ctx)
{
    const npu_inst_div_t *inst = &ctx->inst->div;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_a_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->src_b_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_div(&ctx->engine->sram[inst->src_a_addr],
                           &ctx->engine->sram[inst->src_b_addr],
                           &ctx->engine->sram[inst->dst_addr],
                           inst->num_elements, inst->flags);
}

static int handle_sigmoid(npu_exec_ctx_t *ctx)
{
    const npu_inst_sigmoid_t *inst = &ctx->inst->sigmoid;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_sigmoid(&ctx->engine->sram[inst->src_addr],
                               &ctx->engine->sram[inst->dst_addr],
                               inst->num_elements, inst->flags);
}

static int handle_tanh(npu_exec_ctx_t *ctx)
{
    const npu_inst_tanh_t *inst = &ctx->inst->tanh;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_tanh(&ctx->engine->sram[inst->src_addr],
                            &ctx->engine->sram[inst->dst_addr],
                            inst->num_elements, inst->flags);
}

static int handle_gelu(npu_exec_ctx_t *ctx)
{
    const npu_inst_gelu_t *inst = &ctx->inst->gelu;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_gelu(&ctx->engine->sram[inst->src_addr],
                            &ctx->engine->sram[inst->dst_addr],
                            inst->num_elements, inst->flags);
}

static int handle_softmax(npu_exec_ctx_t *ctx)
{
    const npu_inst_softmax_t *inst = &ctx->inst->softmax;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    uint32_t total_elements = inst->batch_size * inst->axis_size;
    if (!sram_bounds_ok(inst->src_addr, total_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, total_elements * elem_size))
        return -1;

    return npu_compute_softmax(&ctx->engine->sram[inst->src_addr],
                               &ctx->engine->sram[inst->dst_addr],
                               inst->batch_size, inst->axis_size, inst->flags);
}

static int handle_reduce_sum(npu_exec_ctx_t *ctx)
{
    const npu_inst_reduce_sum_t *inst = &ctx->inst->reduce_sum;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    uint32_t src_elements = inst->outer_size * inst->reduce_size * inst->inner_size;
    uint32_t dst_elements = inst->outer_size * inst->inner_size;
    if (!sram_bounds_ok(inst->src_addr, src_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, dst_elements * elem_size))
        return -1;

    return npu_compute_reduce_sum(&ctx->engine->sram[inst->src_addr],
                                  &ctx->engine->sram[inst->dst_addr],
                                  inst->outer_size, inst->reduce_size,
                                  inst->inner_size, inst->flags);
}

static int handle_reduce_mean(npu_exec_ctx_t *ctx)
{
    const npu_inst_reduce_mean_t *inst = &ctx->inst->reduce_mean;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    uint32_t src_elements = inst->outer_size * inst->reduce_size * inst->inner_size;
    uint32_t dst_elements = inst->outer_size * inst->inner_size;
    if (!sram_bounds_ok(inst->src_addr, src_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, dst_elements * elem_size))
        return -1;

    return npu_compute_reduce_mean(&ctx->engine->sram[inst->src_addr],
                                   &ctx->engine->sram[inst->dst_addr],
                                   inst->outer_size, inst->reduce_size,
                                   inst->inner_size, inst->flags);
}

static int handle_reduce_max(npu_exec_ctx_t *ctx)
{
    const npu_inst_reduce_max_t *inst = &ctx->inst->reduce_max;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    uint32_t src_elements = inst->outer_size * inst->reduce_size * inst->inner_size;
    uint32_t dst_elements = inst->outer_size * inst->inner_size;
    if (!sram_bounds_ok(inst->src_addr, src_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, dst_elements * elem_size))
        return -1;

    return npu_compute_reduce_max(&ctx->engine->sram[inst->src_addr],
                                  &ctx->engine->sram[inst->dst_addr],
                                  inst->outer_size, inst->reduce_size,
                                  inst->inner_size, inst->flags);
}

static int handle_transpose(npu_exec_ctx_t *ctx)
{
    const npu_inst_transpose_t *inst = &ctx->inst->transpose;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;

    /* Calculate total elements */
    uint32_t dims[4] = {inst->dim_0, inst->dim_1, inst->dim_2, inst->dim_3};
    uint32_t total = 1;
    for (uint32_t i = 0; i < inst->num_dims; i++)
        total *= dims[i];

    if (!sram_bounds_ok(inst->src_addr, total * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, total * elem_size))
        return -1;

    uint8_t perm[4] = {inst->perm_0, inst->perm_1, inst->perm_2, inst->perm_3};
    return npu_compute_transpose(&ctx->engine->sram[inst->src_addr],
                                 &ctx->engine->sram[inst->dst_addr],
                                 dims, perm, inst->num_dims, inst->flags);
}

static int handle_reshape(npu_exec_ctx_t *ctx)
{
    const npu_inst_reshape_t *inst = &ctx->inst->reshape;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->total_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->total_elements * elem_size))
        return -1;

    return npu_compute_reshape(&ctx->engine->sram[inst->src_addr],
                               &ctx->engine->sram[inst->dst_addr],
                               inst->total_elements, inst->flags);
}

static int handle_concat(npu_exec_ctx_t *ctx)
{
    const npu_inst_concat_t *inst = &ctx->inst->concat;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    uint32_t src0_elements = inst->outer_size * inst->concat_size_0 * inst->inner_size;
    uint32_t src1_elements = inst->outer_size * inst->concat_size_1 * inst->inner_size;
    uint32_t dst_elements = inst->outer_size * (inst->concat_size_0 + inst->concat_size_1) * inst->inner_size;

    if (!sram_bounds_ok(inst->src_addr_0, src0_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->src_addr_1, src1_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, dst_elements * elem_size))
        return -1;

    return npu_compute_concat(&ctx->engine->sram[inst->src_addr_0],
                              &ctx->engine->sram[inst->src_addr_1],
                              &ctx->engine->sram[inst->dst_addr],
                              inst->outer_size, inst->concat_size_0,
                              inst->concat_size_1, inst->inner_size, inst->flags);
}

static int handle_split(npu_exec_ctx_t *ctx)
{
    const npu_inst_split_t *inst = &ctx->inst->split;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    uint32_t src_elements = inst->outer_size * (2 * inst->split_size) * inst->inner_size;
    uint32_t dst_elements = inst->outer_size * inst->split_size * inst->inner_size;

    if (!sram_bounds_ok(inst->src_addr, src_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr_0, dst_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr_1, dst_elements * elem_size))
        return -1;

    return npu_compute_split(&ctx->engine->sram[inst->src_addr],
                             &ctx->engine->sram[inst->dst_addr_0],
                             &ctx->engine->sram[inst->dst_addr_1],
                             inst->outer_size, inst->split_size,
                             inst->inner_size, inst->flags);
}

static int handle_maxpool(npu_exec_ctx_t *ctx)
{
    const npu_inst_maxpool_t *inst = &ctx->inst->maxpool;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;

    /* Calculate output dimensions */
    uint16_t out_h = (inst->in_h - inst->kernel_h) / inst->stride_h + 1;
    uint16_t out_w = (inst->in_w - inst->kernel_w) / inst->stride_w + 1;

    uint32_t src_elements = inst->in_c * inst->in_h * inst->in_w;
    uint32_t dst_elements = inst->in_c * out_h * out_w;

    if (!sram_bounds_ok(inst->src_addr, src_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, dst_elements * elem_size))
        return -1;

    return npu_compute_maxpool(&ctx->engine->sram[inst->src_addr],
                               &ctx->engine->sram[inst->dst_addr],
                               inst->in_c, inst->in_h, inst->in_w,
                               inst->kernel_h, inst->kernel_w,
                               inst->stride_h, inst->stride_w, inst->flags);
}

static int handle_avgpool(npu_exec_ctx_t *ctx)
{
    const npu_inst_avgpool_t *inst = &ctx->inst->avgpool;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;

    /* Calculate output dimensions */
    uint16_t out_h = (inst->in_h - inst->kernel_h) / inst->stride_h + 1;
    uint16_t out_w = (inst->in_w - inst->kernel_w) / inst->stride_w + 1;

    uint32_t src_elements = inst->in_c * inst->in_h * inst->in_w;
    uint32_t dst_elements = inst->in_c * out_h * out_w;

    if (!sram_bounds_ok(inst->src_addr, src_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, dst_elements * elem_size))
        return -1;

    return npu_compute_avgpool(&ctx->engine->sram[inst->src_addr],
                               &ctx->engine->sram[inst->dst_addr],
                               inst->in_c, inst->in_h, inst->in_w,
                               inst->kernel_h, inst->kernel_w,
                               inst->stride_h, inst->stride_w, inst->flags);
}

static int handle_batchnorm(npu_exec_ctx_t *ctx)
{
    const npu_inst_batchnorm_t *inst = &ctx->inst->batchnorm;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    int affine = (inst->flags & 0x02) != 0;

    uint32_t src_elements = inst->num_channels * inst->spatial_size;

    if (!sram_bounds_ok(inst->src_addr, src_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->mean_addr, inst->num_channels * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->var_addr, inst->num_channels * elem_size))
        return -1;
    if (affine) {
        if (!sram_bounds_ok(inst->gamma_addr, inst->num_channels * elem_size))
            return -1;
        if (!sram_bounds_ok(inst->beta_addr, inst->num_channels * elem_size))
            return -1;
    }
    if (!sram_bounds_ok(inst->dst_addr, src_elements * elem_size))
        return -1;

    /* Extract epsilon from fp32 bit representation */
    float epsilon;
    memcpy(&epsilon, &inst->epsilon_fp32, sizeof(float));

    void *gamma = affine ? &ctx->engine->sram[inst->gamma_addr] : NULL;
    void *beta = affine ? &ctx->engine->sram[inst->beta_addr] : NULL;

    return npu_compute_batchnorm(&ctx->engine->sram[inst->src_addr],
                                 &ctx->engine->sram[inst->mean_addr],
                                 &ctx->engine->sram[inst->var_addr],
                                 gamma, beta,
                                 &ctx->engine->sram[inst->dst_addr],
                                 inst->num_channels, inst->spatial_size,
                                 epsilon, inst->flags);
}

static int handle_layernorm(npu_exec_ctx_t *ctx)
{
    const npu_inst_layernorm_t *inst = &ctx->inst->layernorm;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    int affine = (inst->flags & 0x02) != 0;

    uint32_t total_elements = inst->batch_size * inst->normalized_shape;

    if (!sram_bounds_ok(inst->src_addr, total_elements * elem_size))
        return -1;
    if (affine) {
        if (!sram_bounds_ok(inst->gamma_addr, inst->normalized_shape * elem_size))
            return -1;
        if (!sram_bounds_ok(inst->beta_addr, inst->normalized_shape * elem_size))
            return -1;
    }
    if (!sram_bounds_ok(inst->dst_addr, total_elements * elem_size))
        return -1;

    /* Extract epsilon from fp32 bit representation */
    float epsilon;
    memcpy(&epsilon, &inst->epsilon_fp32, sizeof(float));

    void *gamma = affine ? &ctx->engine->sram[inst->gamma_addr] : NULL;
    void *beta = affine ? &ctx->engine->sram[inst->beta_addr] : NULL;

    return npu_compute_layernorm(&ctx->engine->sram[inst->src_addr],
                                 gamma, beta,
                                 &ctx->engine->sram[inst->dst_addr],
                                 inst->batch_size, inst->normalized_shape,
                                 epsilon, inst->flags);
}

static int handle_conv2d(npu_exec_ctx_t *ctx)
{
    const npu_inst_conv2d_t *inst = &ctx->inst->conv2d;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    int bias_en = (inst->flags & 0x04) != 0;

    /* Calculate output dimensions */
    uint16_t out_h = (inst->in_h + 2 * inst->pad_h - inst->kernel_h) / inst->stride_h + 1;
    uint16_t out_w = (inst->in_w + 2 * inst->pad_w - inst->kernel_w) / inst->stride_w + 1;

    uint32_t input_elements = inst->in_c * inst->in_h * inst->in_w;
    uint32_t weight_elements = inst->out_c * inst->in_c * inst->kernel_h * inst->kernel_w;
    uint32_t output_elements = inst->out_c * out_h * out_w;

    if (!sram_bounds_ok(inst->input_addr, input_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->weight_addr, weight_elements * elem_size))
        return -1;
    if (bias_en && !sram_bounds_ok(inst->bias_addr, inst->out_c * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, output_elements * elem_size))
        return -1;

    void *bias = bias_en ? &ctx->engine->sram[inst->bias_addr] : NULL;

    return npu_compute_conv2d(&ctx->engine->sram[inst->input_addr],
                              &ctx->engine->sram[inst->weight_addr],
                              bias,
                              &ctx->engine->sram[inst->dst_addr],
                              inst->in_c, inst->in_h, inst->in_w,
                              inst->out_c, inst->kernel_h, inst->kernel_w,
                              inst->pad_h, inst->pad_w,
                              inst->stride_h, inst->stride_w, inst->flags);
}

static int handle_fmadd(npu_exec_ctx_t *ctx)
{
    const npu_inst_fmadd_t *inst = &ctx->inst->fmadd;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_a_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->src_b_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->src_c_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_fmadd(&ctx->engine->sram[inst->src_a_addr],
                              &ctx->engine->sram[inst->src_b_addr],
                              &ctx->engine->sram[inst->src_c_addr],
                              &ctx->engine->sram[inst->dst_addr],
                              inst->num_elements, inst->flags);
}

static int handle_clamp(npu_exec_ctx_t *ctx)
{
    const npu_inst_clamp_t *inst = &ctx->inst->clamp;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    float min_val, max_val;
    memcpy(&min_val, &inst->min_val_fp32, sizeof(float));
    memcpy(&max_val, &inst->max_val_fp32, sizeof(float));

    return npu_compute_clamp(&ctx->engine->sram[inst->src_addr],
                              &ctx->engine->sram[inst->dst_addr],
                              inst->num_elements, min_val, max_val, inst->flags);
}

static int handle_exp(npu_exec_ctx_t *ctx)
{
    const npu_inst_exp_t *inst = &ctx->inst->exp;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_exp(&ctx->engine->sram[inst->src_addr],
                            &ctx->engine->sram[inst->dst_addr],
                            inst->num_elements, inst->flags);
}

static int handle_log(npu_exec_ctx_t *ctx)
{
    const npu_inst_log_t *inst = &ctx->inst->log;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_log(&ctx->engine->sram[inst->src_addr],
                            &ctx->engine->sram[inst->dst_addr],
                            inst->num_elements, inst->flags);
}

static int handle_sqrt(npu_exec_ctx_t *ctx)
{
    const npu_inst_sqrt_t *inst = &ctx->inst->sqrt;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_sqrt(&ctx->engine->sram[inst->src_addr],
                             &ctx->engine->sram[inst->dst_addr],
                             inst->num_elements, inst->flags);
}

static int handle_rsqrt(npu_exec_ctx_t *ctx)
{
    const npu_inst_rsqrt_t *inst = &ctx->inst->rsqrt;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_rsqrt(&ctx->engine->sram[inst->src_addr],
                              &ctx->engine->sram[inst->dst_addr],
                              inst->num_elements, inst->flags);
}

static int handle_abs(npu_exec_ctx_t *ctx)
{
    const npu_inst_abs_t *inst = &ctx->inst->abs;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_abs(&ctx->engine->sram[inst->src_addr],
                            &ctx->engine->sram[inst->dst_addr],
                            inst->num_elements, inst->flags);
}

static int handle_neg(npu_exec_ctx_t *ctx)
{
    const npu_inst_neg_t *inst = &ctx->inst->neg;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_neg(&ctx->engine->sram[inst->src_addr],
                            &ctx->engine->sram[inst->dst_addr],
                            inst->num_elements, inst->flags);
}

static int handle_gemm(npu_exec_ctx_t *ctx)
{
    const npu_inst_gemm_t *inst = &ctx->inst->gemm;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_a_addr, inst->m * inst->k * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->src_b_addr, inst->k * inst->n * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->m * inst->n * elem_size))
        return -1;

    float alpha, beta;
    memcpy(&alpha, &inst->alpha_fp32, sizeof(float));
    memcpy(&beta, &inst->beta_fp32, sizeof(float));

    return npu_compute_gemm(&ctx->engine->sram[inst->src_a_addr],
                             &ctx->engine->sram[inst->src_b_addr],
                             &ctx->engine->sram[inst->dst_addr],
                             inst->m, inst->n, inst->k,
                             alpha, beta, inst->flags);
}

static int handle_dot(npu_exec_ctx_t *ctx)
{
    const npu_inst_dot_t *inst = &ctx->inst->dot;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_a_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->src_b_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, elem_size))
        return -1;

    return npu_compute_dot(&ctx->engine->sram[inst->src_a_addr],
                            &ctx->engine->sram[inst->src_b_addr],
                            &ctx->engine->sram[inst->dst_addr],
                            inst->num_elements, inst->flags);
}

static int handle_swish(npu_exec_ctx_t *ctx)
{
    const npu_inst_swish_t *inst = &ctx->inst->swish;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_swish(&ctx->engine->sram[inst->src_addr],
                              &ctx->engine->sram[inst->dst_addr],
                              inst->num_elements, inst->flags);
}

static int handle_mish(npu_exec_ctx_t *ctx)
{
    const npu_inst_mish_t *inst = &ctx->inst->mish;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_mish(&ctx->engine->sram[inst->src_addr],
                             &ctx->engine->sram[inst->dst_addr],
                             inst->num_elements, inst->flags);
}

static int handle_rmsnorm(npu_exec_ctx_t *ctx)
{
    const npu_inst_rmsnorm_t *inst = &ctx->inst->rmsnorm;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    int has_weight = (inst->flags & 0x02) != 0;

    uint32_t total_elements = inst->batch_size * inst->normalized_shape;
    if (!sram_bounds_ok(inst->src_addr, total_elements * elem_size))
        return -1;
    if (has_weight && !sram_bounds_ok(inst->weight_addr, inst->normalized_shape * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, total_elements * elem_size))
        return -1;

    float epsilon;
    memcpy(&epsilon, &inst->epsilon_fp32, sizeof(float));

    void *weight = has_weight ? &ctx->engine->sram[inst->weight_addr] : NULL;

    return npu_compute_rmsnorm(&ctx->engine->sram[inst->src_addr],
                                weight,
                                &ctx->engine->sram[inst->dst_addr],
                                inst->batch_size, inst->normalized_shape,
                                epsilon, inst->flags);
}

static int handle_groupnorm(npu_exec_ctx_t *ctx)
{
    const npu_inst_groupnorm_t *inst = &ctx->inst->groupnorm;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    int affine = (inst->flags & 0x02) != 0;

    uint32_t total_elements = inst->num_channels * inst->spatial_size;
    if (!sram_bounds_ok(inst->src_addr, total_elements * elem_size))
        return -1;
    if (affine) {
        if (!sram_bounds_ok(inst->gamma_addr, inst->num_channels * elem_size))
            return -1;
        if (!sram_bounds_ok(inst->beta_addr, inst->num_channels * elem_size))
            return -1;
    }
    if (!sram_bounds_ok(inst->dst_addr, total_elements * elem_size))
        return -1;

    float epsilon;
    memcpy(&epsilon, &inst->epsilon_fp32, sizeof(float));

    void *gamma = affine ? &ctx->engine->sram[inst->gamma_addr] : NULL;
    void *beta = affine ? &ctx->engine->sram[inst->beta_addr] : NULL;

    return npu_compute_groupnorm(&ctx->engine->sram[inst->src_addr],
                                  gamma, beta,
                                  &ctx->engine->sram[inst->dst_addr],
                                  inst->num_groups, inst->num_channels,
                                  inst->spatial_size, epsilon, inst->flags);
}

static int handle_instancenorm(npu_exec_ctx_t *ctx)
{
    const npu_inst_instancenorm_t *inst = &ctx->inst->instancenorm;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    int affine = (inst->flags & 0x02) != 0;

    uint32_t total_elements = inst->num_channels * inst->spatial_size;
    if (!sram_bounds_ok(inst->src_addr, total_elements * elem_size))
        return -1;
    if (affine) {
        if (!sram_bounds_ok(inst->gamma_addr, inst->num_channels * elem_size))
            return -1;
        if (!sram_bounds_ok(inst->beta_addr, inst->num_channels * elem_size))
            return -1;
    }
    if (!sram_bounds_ok(inst->dst_addr, total_elements * elem_size))
        return -1;

    float epsilon;
    memcpy(&epsilon, &inst->epsilon_fp32, sizeof(float));

    void *gamma = affine ? &ctx->engine->sram[inst->gamma_addr] : NULL;
    void *beta = affine ? &ctx->engine->sram[inst->beta_addr] : NULL;

    return npu_compute_instancenorm(&ctx->engine->sram[inst->src_addr],
                                     gamma, beta,
                                     &ctx->engine->sram[inst->dst_addr],
                                     inst->num_channels, inst->spatial_size,
                                     epsilon, inst->flags);
}

static int handle_sdpa(npu_exec_ctx_t *ctx)
{
    const npu_inst_scaled_dot_product_attention_t *inst = &ctx->inst->scaled_dot_product_attention;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;

    uint32_t q_elements = inst->num_heads * inst->seq_len_q * inst->head_dim;
    uint32_t kv_elements = inst->num_heads * inst->seq_len_kv * inst->head_dim;

    if (!sram_bounds_ok(inst->q_addr, q_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->k_addr, kv_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->v_addr, kv_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, q_elements * elem_size))
        return -1;

    float scale;
    memcpy(&scale, &inst->scale_fp32, sizeof(float));
    if (scale == 0.0f)
        scale = 1.0f / sqrtf((float)inst->head_dim);

    return npu_compute_sdpa(&ctx->engine->sram[inst->q_addr],
                             &ctx->engine->sram[inst->k_addr],
                             &ctx->engine->sram[inst->v_addr],
                             &ctx->engine->sram[inst->dst_addr],
                             inst->num_heads, inst->seq_len_q, inst->seq_len_kv,
                             inst->head_dim, scale, inst->flags);
}

static int handle_cast(npu_exec_ctx_t *ctx)
{
    const npu_inst_cast_t *inst = &ctx->inst->cast;
    uint32_t src_elem_sizes[] = {4, 2, 1, 4};
    uint32_t dst_elem_sizes[] = {4, 2, 1, 4};

    if (inst->src_dtype > 3 || inst->dst_dtype > 3)
        return -1;

    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * src_elem_sizes[inst->src_dtype]))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * dst_elem_sizes[inst->dst_dtype]))
        return -1;

    return npu_compute_cast(&ctx->engine->sram[inst->src_addr],
                             &ctx->engine->sram[inst->dst_addr],
                             inst->num_elements, inst->src_dtype, inst->dst_dtype);
}

static int handle_quantize(npu_exec_ctx_t *ctx)
{
    const npu_inst_quantize_t *inst = &ctx->inst->quantize;
    int per_channel = (inst->flags & 0x01) != 0;
    uint32_t num_scale = per_channel ? inst->num_channels : 1;

    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * sizeof(float)))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements))
        return -1;
    if (!sram_bounds_ok(inst->scale_addr, num_scale * sizeof(float)))
        return -1;
    if (!sram_bounds_ok(inst->zero_point_addr, num_scale))
        return -1;

    return npu_compute_quantize(&ctx->engine->sram[inst->src_addr],
                                 &ctx->engine->sram[inst->dst_addr],
                                 &ctx->engine->sram[inst->scale_addr],
                                 &ctx->engine->sram[inst->zero_point_addr],
                                 inst->num_elements, inst->num_channels,
                                 inst->flags);
}

static int handle_dequantize(npu_exec_ctx_t *ctx)
{
    const npu_inst_dequantize_t *inst = &ctx->inst->dequantize;
    int per_channel = (inst->flags & 0x01) != 0;
    uint32_t num_scale = per_channel ? inst->num_channels : 1;

    if (!sram_bounds_ok(inst->src_addr, inst->num_elements))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * sizeof(float)))
        return -1;
    if (!sram_bounds_ok(inst->scale_addr, num_scale * sizeof(float)))
        return -1;
    if (!sram_bounds_ok(inst->zero_point_addr, num_scale))
        return -1;

    return npu_compute_dequantize(&ctx->engine->sram[inst->src_addr],
                                   &ctx->engine->sram[inst->dst_addr],
                                   &ctx->engine->sram[inst->scale_addr],
                                   &ctx->engine->sram[inst->zero_point_addr],
                                   inst->num_elements, inst->num_channels,
                                   inst->flags);
}

static int handle_depthwise_conv(npu_exec_ctx_t *ctx)
{
    const npu_inst_depthwise_conv_t *inst = &ctx->inst->depthwise_conv;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;

    /* Calculate output dimensions */
    uint16_t out_h = (inst->in_h + 2 * inst->pad_h - inst->kernel_h) / inst->stride_h + 1;
    uint16_t out_w = (inst->in_w + 2 * inst->pad_w - inst->kernel_w) / inst->stride_w + 1;

    uint32_t input_elements = inst->channels * inst->in_h * inst->in_w;
    uint32_t weight_elements = inst->channels * inst->kernel_h * inst->kernel_w;
    uint32_t output_elements = inst->channels * out_h * out_w;

    if (!sram_bounds_ok(inst->input_addr, input_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->weight_addr, weight_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, output_elements * elem_size))
        return -1;

    return npu_compute_depthwise_conv(&ctx->engine->sram[inst->input_addr],
                                      &ctx->engine->sram[inst->weight_addr],
                                      &ctx->engine->sram[inst->dst_addr],
                                      inst->channels, inst->in_h, inst->in_w,
                                      inst->kernel_h, inst->kernel_w,
                                      inst->pad_h, inst->pad_w,
                                      inst->stride_h, inst->stride_w, inst->flags);
}

/* ================================================================== */
/* Data Movement / Manipulation Handlers                              */
/* ================================================================== */

static int handle_gather(npu_exec_ctx_t *ctx)
{
    const npu_inst_gather_t *inst = &ctx->inst->gather;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;

    uint32_t src_bytes = inst->outer_size * inst->gather_size * inst->inner_size * elem_size;
    uint32_t idx_bytes = inst->outer_size * inst->num_indices * sizeof(uint32_t);
    uint32_t dst_bytes = inst->outer_size * inst->num_indices * inst->inner_size * elem_size;

    if (!sram_bounds_ok(inst->src_addr, src_bytes))
        return -1;
    if (!sram_bounds_ok(inst->idx_addr, idx_bytes))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, dst_bytes))
        return -1;

    return npu_compute_gather(&ctx->engine->sram[inst->src_addr],
                              &ctx->engine->sram[inst->idx_addr],
                              &ctx->engine->sram[inst->dst_addr],
                              inst->outer_size, inst->gather_size,
                              inst->inner_size, inst->num_indices, inst->flags);
}

static int handle_slice(npu_exec_ctx_t *ctx)
{
    const npu_inst_slice_t *inst = &ctx->inst->slice;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;

    uint32_t src_bytes = inst->outer_size * inst->src_axis_size * inst->inner_size * elem_size;
    uint32_t dst_bytes = inst->outer_size * inst->length * inst->inner_size * elem_size;

    if (!sram_bounds_ok(inst->src_addr, src_bytes))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, dst_bytes))
        return -1;

    return npu_compute_slice(&ctx->engine->sram[inst->src_addr],
                             &ctx->engine->sram[inst->dst_addr],
                             inst->outer_size, inst->src_axis_size,
                             inst->inner_size, inst->start,
                             inst->length, inst->step, inst->flags);
}

static int handle_pad(npu_exec_ctx_t *ctx)
{
    const npu_inst_pad_t *inst = &ctx->inst->pad;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;

    uint16_t src_dims[4] = { inst->src_dim0, inst->src_dim1, inst->src_dim2, inst->src_dim3 };
    uint16_t pad_before[4] = { inst->pad_before0, inst->pad_before1, inst->pad_before2, inst->pad_before3 };
    uint16_t pad_after[4] = { inst->pad_after0, inst->pad_after1, inst->pad_after2, inst->pad_after3 };

    uint32_t src_total = 1, dst_total = 1;
    uint8_t d;
    for (d = 0; d < inst->num_dims; d++) {
        src_total *= src_dims[d];
        dst_total *= (pad_before[d] + src_dims[d] + pad_after[d]);
    }

    if (!sram_bounds_ok(inst->src_addr, src_total * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, dst_total * elem_size))
        return -1;

    return npu_compute_pad(&ctx->engine->sram[inst->src_addr],
                           &ctx->engine->sram[inst->dst_addr],
                           src_dims, pad_before, pad_after,
                           inst->num_dims, inst->pad_value, inst->flags);
}

static int handle_where(npu_exec_ctx_t *ctx)
{
    const npu_inst_where_t *inst = &ctx->inst->where;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;

    if (!sram_bounds_ok(inst->cond_addr, inst->num_elements))
        return -1;
    if (!sram_bounds_ok(inst->true_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->false_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_where(&ctx->engine->sram[inst->cond_addr],
                             &ctx->engine->sram[inst->true_addr],
                             &ctx->engine->sram[inst->false_addr],
                             &ctx->engine->sram[inst->dst_addr],
                             inst->num_elements, inst->flags);
}

/* ================================================================== */
/* VLM/VLA Instruction Handlers                                       */
/* ================================================================== */

static int handle_prefetch(npu_exec_ctx_t *ctx)
{
    /* PREFETCH behaves like LOAD in emulation (no async overlap) */
    const npu_inst_prefetch_t *inst = &ctx->inst->prefetch;
    if (!sram_bounds_ok(inst->npu_addr, inst->size_bytes))
        return -1;
    return ctx->dma_read(ctx->opaque, inst->host_addr,
                         &ctx->engine->sram[inst->npu_addr], inst->size_bytes);
}

static int handle_kv_cache_append(npu_exec_ctx_t *ctx)
{
    const npu_inst_kv_cache_append_t *inst = &ctx->inst->kv_cache_append;
    uint16_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    uint32_t kv_vec_bytes = inst->num_kv_heads * inst->head_dim * elem_size;
    uint32_t cache_bytes = inst->num_kv_heads * inst->max_seq_len * inst->head_dim * elem_size;

    if (!sram_bounds_ok(inst->k_new_addr, kv_vec_bytes))
        return -1;
    if (!sram_bounds_ok(inst->v_new_addr, kv_vec_bytes))
        return -1;
    if (!sram_bounds_ok(inst->k_cache_addr, cache_bytes))
        return -1;
    if (!sram_bounds_ok(inst->v_cache_addr, cache_bytes))
        return -1;

    return npu_compute_kv_cache_append(
        &ctx->engine->sram[inst->k_new_addr],
        &ctx->engine->sram[inst->v_new_addr],
        &ctx->engine->sram[inst->k_cache_addr],
        &ctx->engine->sram[inst->v_cache_addr],
        inst->cur_seq_pos, inst->num_kv_heads, inst->head_dim,
        inst->max_seq_len, inst->flags);
}

static int handle_kv_cache_attention(npu_exec_ctx_t *ctx)
{
    const npu_inst_kv_cache_attention_t *inst = &ctx->inst->kv_cache_attention;
    uint16_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    uint32_t q_bytes = inst->num_heads * inst->head_dim * elem_size;
    uint32_t cache_bytes = inst->num_kv_heads * inst->max_seq_len * inst->head_dim * elem_size;

    if (!sram_bounds_ok(inst->q_addr, q_bytes))
        return -1;
    if (!sram_bounds_ok(inst->k_cache_addr, cache_bytes))
        return -1;
    if (!sram_bounds_ok(inst->v_cache_addr, cache_bytes))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, q_bytes))
        return -1;

    float scale;
    memcpy(&scale, &inst->scale_fp32, sizeof(float));
    if (scale == 0.0f)
        scale = 1.0f / sqrtf((float)inst->head_dim);

    return npu_compute_kv_cache_attention(
        &ctx->engine->sram[inst->q_addr],
        &ctx->engine->sram[inst->k_cache_addr],
        &ctx->engine->sram[inst->v_cache_addr],
        &ctx->engine->sram[inst->dst_addr],
        inst->num_heads, inst->num_kv_heads, inst->cur_seq_len,
        inst->head_dim, inst->max_seq_len, scale, inst->flags);
}

static int handle_kv_cache_reset(npu_exec_ctx_t *ctx)
{
    const npu_inst_kv_cache_reset_t *inst = &ctx->inst->kv_cache_reset;
    uint32_t cache_bytes = inst->num_kv_heads * inst->max_seq_len * inst->head_dim * 2;

    if (!sram_bounds_ok(inst->k_cache_addr, cache_bytes))
        return -1;
    if (!sram_bounds_ok(inst->v_cache_addr, cache_bytes))
        return -1;

    return npu_compute_kv_cache_reset(
        &ctx->engine->sram[inst->k_cache_addr],
        &ctx->engine->sram[inst->v_cache_addr],
        inst->num_kv_heads, inst->head_dim, inst->max_seq_len, inst->flags);
}

static int handle_embedding_lookup(npu_exec_ctx_t *ctx)
{
    const npu_inst_embedding_lookup_t *inst = &ctx->inst->embedding_lookup;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    uint32_t table_bytes = inst->vocab_size * inst->embed_dim * elem_size;
    uint32_t row_bytes = inst->embed_dim * elem_size;

    if (!sram_bounds_ok(inst->table_addr, table_bytes))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, row_bytes))
        return -1;

    return npu_compute_embedding_lookup(
        &ctx->engine->sram[inst->table_addr],
        &ctx->engine->sram[inst->dst_addr],
        inst->token_id, inst->vocab_size, inst->embed_dim, inst->flags);
}

static int handle_token_sample(npu_exec_ctx_t *ctx)
{
    const npu_inst_token_sample_t *inst = &ctx->inst->token_sample;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    uint32_t logits_bytes = inst->vocab_size * elem_size;

    if (!sram_bounds_ok(inst->logits_addr, logits_bytes))
        return -1;
    if (!sram_bounds_ok(inst->dst_token_addr, 4))
        return -1;

    float temperature;
    memcpy(&temperature, &inst->temperature_fp32, sizeof(float));

    return npu_compute_token_sample(
        &ctx->engine->sram[inst->logits_addr],
        &ctx->engine->sram[inst->dst_token_addr],
        inst->vocab_size, inst->sub_opcode, temperature,
        inst->top_k, inst->seed, inst->flags);
}

static int handle_rotary_embedding(npu_exec_ctx_t *ctx)
{
    const npu_inst_rotary_embedding_t *inst = &ctx->inst->rotary_embedding;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    uint32_t total_bytes = inst->num_heads * inst->head_dim * elem_size;

    if (!sram_bounds_ok(inst->src_addr, total_bytes))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, total_bytes))
        return -1;

    float theta_base;
    memcpy(&theta_base, &inst->theta_base_fp32, sizeof(float));
    if (theta_base == 0.0f)
        theta_base = 10000.0f;

    return npu_compute_rotary_embedding(
        &ctx->engine->sram[inst->src_addr],
        &ctx->engine->sram[inst->dst_addr],
        inst->position, inst->num_heads, inst->head_dim,
        theta_base, inst->flags);
}

/* ================================================================== */
/* Opcode Dispatch Table                                              */
/* ================================================================== */

/*
 * Direct 256-entry dispatch table for O(1) opcode lookup.
 * Memory footprint: 256 * 8 bytes = 2KB (fits in L1 cache).
 * Inspired by NVIDIA PTX instruction dispatch architecture.
 */
static npu_handler_fn g_opcode_handlers[256] = {
    /* 0x00-0x0F: Control */
    [NPU_OP_NOP]  = handle_nop,
    [NPU_OP_HALT] = handle_halt,

    /* 0x10-0x1F: Arithmetic */
    [NPU_OP_ADD]   = handle_add,
    [NPU_OP_SUB]   = handle_sub,
    [NPU_OP_MUL]   = handle_mul,
    [NPU_OP_DIV]   = handle_div,
    [NPU_OP_FMADD] = handle_fmadd,
    [NPU_OP_CLAMP] = handle_clamp,
    [NPU_OP_EXP]   = handle_exp,
    [NPU_OP_LOG]   = handle_log,
    [NPU_OP_SQRT]  = handle_sqrt,
    [NPU_OP_RSQRT] = handle_rsqrt,
    [NPU_OP_ABS]   = handle_abs,
    [NPU_OP_NEG]   = handle_neg,

    /* 0x20-0x2F: Linalg */
    [NPU_OP_MATMUL] = handle_matmul,
    [NPU_OP_GEMM]   = handle_gemm,
    [NPU_OP_DOT]    = handle_dot,

    /* 0x30-0x3F: Convolution */
    [NPU_OP_CONV2D]         = handle_conv2d,
    [NPU_OP_DEPTHWISE_CONV] = handle_depthwise_conv,

    /* 0x40-0x4F: Activation */
    [NPU_OP_RELU]    = handle_relu,
    [NPU_OP_SIGMOID] = handle_sigmoid,
    [NPU_OP_TANH]    = handle_tanh,
    [NPU_OP_GELU]    = handle_gelu,
    [NPU_OP_SWISH]   = handle_swish,
    [NPU_OP_SOFTMAX] = handle_softmax,
    [NPU_OP_MISH]    = handle_mish,

    /* 0x50-0x5F: Normalization */
    [NPU_OP_BATCHNORM]    = handle_batchnorm,
    [NPU_OP_LAYERNORM]    = handle_layernorm,
    [NPU_OP_RMSNORM]      = handle_rmsnorm,
    [NPU_OP_GROUPNORM]    = handle_groupnorm,
    [NPU_OP_INSTANCENORM] = handle_instancenorm,

    /* 0x60-0x6F: Pooling */
    [NPU_OP_MAXPOOL] = handle_maxpool,
    [NPU_OP_AVGPOOL] = handle_avgpool,

    /* 0x70-0x7F: Reduction */
    [NPU_OP_REDUCE_SUM]  = handle_reduce_sum,
    [NPU_OP_REDUCE_MEAN] = handle_reduce_mean,
    [NPU_OP_REDUCE_MAX]  = handle_reduce_max,

    /* 0x80-0x8F: Memory */
    [NPU_OP_LOAD]     = handle_load,
    [NPU_OP_STORE]    = handle_store,
    [NPU_OP_PREFETCH] = handle_prefetch,

    /* 0x90-0x9F: Layout */
    [NPU_OP_TRANSPOSE] = handle_transpose,
    [NPU_OP_RESHAPE]   = handle_reshape,
    [NPU_OP_CONCAT]    = handle_concat,
    [NPU_OP_SPLIT]     = handle_split,
    [NPU_OP_GATHER]    = handle_gather,
    [NPU_OP_SLICE]     = handle_slice,
    [NPU_OP_PAD]       = handle_pad,
    [NPU_OP_WHERE]     = handle_where,

    /* 0xA0-0xAF: Attention */
    [NPU_OP_SCALED_DOT_PRODUCT_ATTENTION] = handle_sdpa,
    [NPU_OP_KV_CACHE_APPEND]    = handle_kv_cache_append,
    [NPU_OP_KV_CACHE_ATTENTION] = handle_kv_cache_attention,
    [NPU_OP_KV_CACHE_RESET]     = handle_kv_cache_reset,

    /* 0xB0-0xBF: Data Type */
    [NPU_OP_CAST]       = handle_cast,
    [NPU_OP_QUANTIZE]   = handle_quantize,
    [NPU_OP_DEQUANTIZE] = handle_dequantize,

    /* 0xC0-0xCF: Token */
    [NPU_OP_EMBEDDING_LOOKUP]  = handle_embedding_lookup,
    [NPU_OP_TOKEN_SAMPLE]      = handle_token_sample,
    [NPU_OP_ROTARY_EMBEDDING]  = handle_rotary_embedding,
};

/* ================================================================== */
/* Engine Execution Loop                                              */
/* ================================================================== */

int npu_engine_run(npu_engine_t *engine,
                   const uint8_t *program, uint32_t program_size,
                   npu_dma_read_fn dma_read,
                   npu_dma_write_fn dma_write,
                   void *opaque)
{
    uint32_t num_insts;
    int ret;

    if (!engine || !program)
        return -1;
    if (program_size % NPU_INST_BYTES != 0)
        return -1;

    num_insts = program_size / NPU_INST_BYTES;
    npu_engine_reset(engine);

    #define NPU_MAX_ITERATIONS 10000000  /* Safety limit: 10M instructions */
    uint32_t iterations = 0;

    while (engine->pc < num_insts && !engine->halted) {
        if (++iterations > NPU_MAX_ITERATIONS) {
            engine->error = 1;
            return -1;
        }
        const npu_inst_t *inst =
            (const npu_inst_t *)&program[engine->pc * NPU_INST_BYTES];

        /* O(1) dispatch via direct table lookup */
        npu_handler_fn handler = g_opcode_handlers[inst->opcode];
        if (!handler) {
            /* Unknown/unassigned opcode */
            engine->error = 1;
            return -1;
        }

        /* Execute instruction handler */
        npu_exec_ctx_t ctx = {
            .engine = engine,
            .inst = inst,
            .dma_read = dma_read,
            .dma_write = dma_write,
            .opaque = opaque,
        };

        ret = handler(&ctx);
        if (ret != 0) {
            engine->error = 1;
            return -1;
        }

        /* HALT instruction sets halted flag and returns early */
        if (engine->halted) {
            return 0;
        }

        engine->pc++;
    }

    /* Fell off end without HALT */
    if (!engine->halted) {
        engine->error = 1;
        return -1;
    }

    return 0;
}
