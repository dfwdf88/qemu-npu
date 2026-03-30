/*
 * LPU Instruction Execution Engine - Implementation
 *
 * Fetches 64-byte instructions from a program buffer,
 * decodes opcodes via lpu_inst_t union, and dispatches
 * to compute kernels (reused from NPU npu_compute.c).
 *
 * Uses a direct dispatch table (256 entries) for O(1) opcode lookup.
 * The LPU ISA is a 21-instruction subset of the NPU ISA with
 * identical opcode values and 64-byte instruction format.
 */

#include "lpu_engine.h"
#include "lpu_spec.h"
#include "../npu/npu_compute.h"

#include <string.h>
#include <math.h>

/* ================================================================== */
/* Instruction Handler Context                                        */
/* ================================================================== */

typedef struct {
    lpu_engine_t *engine;
    const lpu_inst_t *inst;
    lpu_dma_read_fn dma_read;
    lpu_dma_write_fn dma_write;
    void *opaque;
} lpu_exec_ctx_t;

/* Function pointer type for instruction handlers */
typedef int (*lpu_handler_fn)(lpu_exec_ctx_t *ctx);

void lpu_engine_init(lpu_engine_t *engine)
{
    memset(engine->sram, 0, LPU_SRAM_SIZE);
    engine->pc = 0;
    engine->halted = 0;
    engine->exit_code = 0;
    engine->error = 0;
}

void lpu_engine_reset(lpu_engine_t *engine)
{
    /* DO NOT clear SRAM -- KV cache persistence across decode steps */
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
    if (offset + size > LPU_SRAM_SIZE)
        return 0;
    return 1;
}

/* ================================================================== */
/* Instruction Handlers                                               */
/* ================================================================== */

/* 0x00 NOP */
static int handle_nop(lpu_exec_ctx_t *ctx)
{
    (void)ctx;  /* Unused */
    return 0;
}

/* 0x01 HALT */
static int handle_halt(lpu_exec_ctx_t *ctx)
{
    ctx->engine->halted = 1;
    ctx->engine->exit_code = ctx->inst->halt.exit_code;
    return 0;
}

/* 0x10 ADD */
static int handle_add(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_add_t *inst = &ctx->inst->add;
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

/* 0x12 MUL */
static int handle_mul(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_mul_t *inst = &ctx->inst->mul;
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

/* 0x20 MATMUL */
static int handle_matmul(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_matmul_t *inst = &ctx->inst->matmul;
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

/* 0x21 GEMM */
static int handle_gemm(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_gemm_t *inst = &ctx->inst->gemm;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_a_addr, inst->m * inst->k * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->src_b_addr, inst->k * inst->n * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->m * inst->n * elem_size))
        return -1;

    /* LPU GEMM uses fp16 bit pattern for alpha/beta; convert to float */
    float alpha = npu_fp16_to_fp32(inst->alpha_fp16);
    float beta = npu_fp16_to_fp32(inst->beta_fp16);

    /* npu_compute_gemm reads old C from dst (in-place).
     * If bias is at a separate address, copy it to dst first. */
    if ((inst->flags & LPU_GEMM_FLAG_HAS_BIAS) && inst->bias_addr != inst->dst_addr) {
        uint32_t c_size = inst->m * inst->n * elem_size;
        if (!sram_bounds_ok(inst->bias_addr, c_size))
            return -1;
        memcpy(&ctx->engine->sram[inst->dst_addr],
               &ctx->engine->sram[inst->bias_addr], c_size);
    }

    return npu_compute_gemm(&ctx->engine->sram[inst->src_a_addr],
                             &ctx->engine->sram[inst->src_b_addr],
                             &ctx->engine->sram[inst->dst_addr],
                             inst->m, inst->n, inst->k,
                             alpha, beta, inst->flags);
}

/* 0x43 GELU */
static int handle_gelu(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_gelu_t *inst = &ctx->inst->gelu;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_gelu(&ctx->engine->sram[inst->src_addr],
                            &ctx->engine->sram[inst->dst_addr],
                            inst->num_elements, inst->flags);
}

/* 0x44 SWISH */
static int handle_swish(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_swish_t *inst = &ctx->inst->swish;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_swish(&ctx->engine->sram[inst->src_addr],
                              &ctx->engine->sram[inst->dst_addr],
                              inst->num_elements, inst->flags);
}

/* 0x45 SOFTMAX */
static int handle_softmax(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_softmax_t *inst = &ctx->inst->softmax;
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

/* 0x51 LAYERNORM */
static int handle_layernorm(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_layernorm_t *inst = &ctx->inst->layernorm;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    int has_bias = (inst->flags & 0x02) != 0;

    uint32_t total_elements = inst->batch_size * inst->hidden_size;

    if (!sram_bounds_ok(inst->src_addr, total_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->gamma_addr, inst->hidden_size * elem_size))
        return -1;
    if (has_bias && !sram_bounds_ok(inst->beta_addr, inst->hidden_size * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, total_elements * elem_size))
        return -1;

    /* Extract epsilon from fp32 bit representation */
    float epsilon;
    memcpy(&epsilon, &inst->eps_fp32, sizeof(float));

    void *gamma = &ctx->engine->sram[inst->gamma_addr];
    void *beta = has_bias ? &ctx->engine->sram[inst->beta_addr] : NULL;

    return npu_compute_layernorm(&ctx->engine->sram[inst->src_addr],
                                 gamma, beta,
                                 &ctx->engine->sram[inst->dst_addr],
                                 inst->batch_size, inst->hidden_size,
                                 epsilon, inst->flags);
}

/* 0x52 RMSNORM */
static int handle_rmsnorm(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_rmsnorm_t *inst = &ctx->inst->rmsnorm;
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;

    uint32_t total_elements = inst->batch_size * inst->hidden_size;
    if (!sram_bounds_ok(inst->src_addr, total_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->gamma_addr, inst->hidden_size * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, total_elements * elem_size))
        return -1;

    float epsilon;
    memcpy(&epsilon, &inst->eps_fp32, sizeof(float));

    return npu_compute_rmsnorm(&ctx->engine->sram[inst->src_addr],
                                &ctx->engine->sram[inst->gamma_addr],
                                &ctx->engine->sram[inst->dst_addr],
                                inst->batch_size, inst->hidden_size,
                                epsilon, inst->flags);
}

/* 0x80 LOAD */
static int handle_load(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_load_t *inst = &ctx->inst->load;
    if (!sram_bounds_ok(inst->lpu_addr, inst->size_bytes))
        return -1;
    return ctx->dma_read(ctx->opaque, inst->host_addr,
                         &ctx->engine->sram[inst->lpu_addr], inst->size_bytes);
}

/* 0x81 STORE */
static int handle_store(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_store_t *inst = &ctx->inst->store;
    if (!sram_bounds_ok(inst->lpu_addr, inst->size_bytes))
        return -1;
    return ctx->dma_write(ctx->opaque, inst->host_addr,
                          &ctx->engine->sram[inst->lpu_addr], inst->size_bytes);
}

/* 0x82 PREFETCH */
static int handle_prefetch(lpu_exec_ctx_t *ctx)
{
    /* PREFETCH behaves like LOAD in emulation (no async overlap) */
    const lpu_inst_prefetch_t *inst = &ctx->inst->prefetch;
    if (!sram_bounds_ok(inst->lpu_addr, inst->size_bytes))
        return -1;
    return ctx->dma_read(ctx->opaque, inst->host_addr,
                         &ctx->engine->sram[inst->lpu_addr], inst->size_bytes);
}

/* 0xA0 SDPA */
static int handle_sdpa(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_scaled_dot_product_attention_t *inst = &ctx->inst->scaled_dot_product_attention;
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

/* 0xA1 KV_CACHE_APPEND */
static int handle_kv_cache_append(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_kv_cache_append_t *inst = &ctx->inst->kv_cache_append;
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

/* 0xA2 KV_CACHE_ATTENTION */
static int handle_kv_cache_attention(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_kv_cache_attention_t *inst = &ctx->inst->kv_cache_attention;
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

/* 0xA3 KV_CACHE_RESET */
static int handle_kv_cache_reset(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_kv_cache_reset_t *inst = &ctx->inst->kv_cache_reset;
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

/* 0xC0 EMBEDDING_LOOKUP */
static int handle_embedding_lookup(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_embedding_lookup_t *inst = &ctx->inst->embedding_lookup;
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

/* 0xC1 TOKEN_SAMPLE */
static int handle_token_sample(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_token_sample_t *inst = &ctx->inst->token_sample;
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

/* 0xC2 ROTARY_EMBEDDING */
static int handle_rotary_embedding(lpu_exec_ctx_t *ctx)
{
    const lpu_inst_rotary_embedding_t *inst = &ctx->inst->rotary_embedding;
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
 * Only the 21 LPU opcodes are populated; all others are NULL.
 */
static lpu_handler_fn g_opcode_handlers[256] = {
    /* 0x00-0x01: Control */
    [LPU_OP_NOP]  = handle_nop,
    [LPU_OP_HALT] = handle_halt,

    /* 0x10-0x12: Arithmetic */
    [LPU_OP_ADD] = handle_add,
    [LPU_OP_MUL] = handle_mul,

    /* 0x20-0x21: Linalg */
    [LPU_OP_MATMUL] = handle_matmul,
    [LPU_OP_GEMM]   = handle_gemm,

    /* 0x43-0x45: Activation */
    [LPU_OP_GELU]    = handle_gelu,
    [LPU_OP_SWISH]   = handle_swish,
    [LPU_OP_SOFTMAX] = handle_softmax,

    /* 0x51-0x52: Normalization */
    [LPU_OP_LAYERNORM] = handle_layernorm,
    [LPU_OP_RMSNORM]   = handle_rmsnorm,

    /* 0x80-0x82: Memory */
    [LPU_OP_LOAD]     = handle_load,
    [LPU_OP_STORE]    = handle_store,
    [LPU_OP_PREFETCH] = handle_prefetch,

    /* 0xA0-0xA3: Attention */
    [LPU_OP_SCALED_DOT_PRODUCT_ATTENTION] = handle_sdpa,
    [LPU_OP_KV_CACHE_APPEND]    = handle_kv_cache_append,
    [LPU_OP_KV_CACHE_ATTENTION] = handle_kv_cache_attention,
    [LPU_OP_KV_CACHE_RESET]     = handle_kv_cache_reset,

    /* 0xC0-0xC2: Token */
    [LPU_OP_EMBEDDING_LOOKUP]  = handle_embedding_lookup,
    [LPU_OP_TOKEN_SAMPLE]      = handle_token_sample,
    [LPU_OP_ROTARY_EMBEDDING]  = handle_rotary_embedding,
};

/* ================================================================== */
/* Engine Execution Loop                                              */
/* ================================================================== */

int lpu_engine_run(lpu_engine_t *engine,
                   const uint8_t *program, uint32_t program_size,
                   lpu_dma_read_fn dma_read,
                   lpu_dma_write_fn dma_write,
                   void *opaque)
{
    uint32_t num_insts;
    int ret;

    if (!engine || !program)
        return -1;
    if (program_size % LPU_INST_BYTES != 0)
        return -1;

    num_insts = program_size / LPU_INST_BYTES;
    lpu_engine_reset(engine);

    while (engine->pc < num_insts && !engine->halted) {
        const lpu_inst_t *inst =
            (const lpu_inst_t *)&program[engine->pc * LPU_INST_BYTES];

        /* O(1) dispatch via direct table lookup */
        lpu_handler_fn handler = g_opcode_handlers[inst->opcode];
        if (!handler) {
            /* Unknown/unassigned opcode */
            engine->error = 1;
            return -1;
        }

        /* Execute instruction handler */
        lpu_exec_ctx_t ctx = {
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
