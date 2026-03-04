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

/* Stub handler for unimplemented operations */
static int handle_unimplemented(npu_exec_ctx_t *ctx)
{
    (void)ctx;  /* Unused - will be used when implemented */
    /* TODO: implement compute kernel for this operation */
    return -1;
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
    [NPU_OP_ADD] = handle_add,
    [NPU_OP_SUB] = handle_sub,
    [NPU_OP_MUL] = handle_mul,
    [NPU_OP_DIV] = handle_div,

    /* 0x20-0x2F: Linalg */
    [NPU_OP_MATMUL] = handle_matmul,

    /* 0x30-0x3F: Convolution */
    [NPU_OP_CONV2D]         = handle_conv2d,
    [NPU_OP_DEPTHWISE_CONV] = handle_depthwise_conv,

    /* 0x40-0x4F: Activation */
    [NPU_OP_RELU]    = handle_relu,
    [NPU_OP_SIGMOID] = handle_sigmoid,
    [NPU_OP_TANH]    = handle_tanh,
    [NPU_OP_GELU]    = handle_gelu,
    [NPU_OP_SOFTMAX] = handle_softmax,

    /* 0x50-0x5F: Normalization */
    [NPU_OP_BATCHNORM] = handle_batchnorm,
    [NPU_OP_LAYERNORM] = handle_layernorm,

    /* 0x60-0x6F: Pooling */
    [NPU_OP_MAXPOOL] = handle_maxpool,
    [NPU_OP_AVGPOOL] = handle_avgpool,

    /* 0x70-0x7F: Reduction */
    [NPU_OP_REDUCE_SUM]  = handle_reduce_sum,
    [NPU_OP_REDUCE_MEAN] = handle_reduce_mean,
    [NPU_OP_REDUCE_MAX]  = handle_reduce_max,

    /* 0x80-0x8F: Memory */
    [NPU_OP_LOAD]  = handle_load,
    [NPU_OP_STORE] = handle_store,

    /* 0x90-0x9F: Layout */
    [NPU_OP_TRANSPOSE] = handle_transpose,
    [NPU_OP_RESHAPE]   = handle_reshape,
    [NPU_OP_CONCAT]    = handle_concat,
    [NPU_OP_SPLIT]     = handle_split,

    /* Remaining entries (0xA0-0xFF) are NULL by default */
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

    while (engine->pc < num_insts && !engine->halted) {
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
