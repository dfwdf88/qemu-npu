/*
 * NPU Instruction Execution Engine - Implementation
 *
 * Fetches 64-byte instructions from a program buffer,
 * decodes opcodes via npu_inst_t union, and dispatches
 * to compute kernels (hw/compute/) or DMA callbacks.
 */

#include "npu_engine.h"
#include "npu_spec.h"
#include "npu_compute.h"

#include <string.h>

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

/* Check that [offset, offset+size) fits in SRAM */
static int sram_bounds_ok(uint64_t offset, uint32_t size)
{
    if (offset + size > NPU_SRAM_SIZE)
        return 0;
    return 1;
}

static int exec_load(npu_engine_t *engine, const npu_inst_load_t *inst,
                     npu_dma_read_fn dma_read, void *opaque)
{
    if (!sram_bounds_ok(inst->npu_addr, inst->size_bytes))
        return -1;
    return dma_read(opaque, inst->host_addr,
                    &engine->sram[inst->npu_addr], inst->size_bytes);
}

static int exec_store(npu_engine_t *engine, const npu_inst_store_t *inst,
                      npu_dma_write_fn dma_write, void *opaque)
{
    if (!sram_bounds_ok(inst->npu_addr, inst->size_bytes))
        return -1;
    return dma_write(opaque, inst->host_addr,
                     &engine->sram[inst->npu_addr], inst->size_bytes);
}

static int exec_matmul(npu_engine_t *engine, const npu_inst_matmul_t *inst)
{
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

    return npu_compute_matmul(&engine->sram[a_off], &engine->sram[b_off],
                              &engine->sram[c_off], m, n, k, inst->flags);
}

static int exec_relu(npu_engine_t *engine, const npu_inst_relu_t *inst)
{
    uint32_t elem_size = (inst->flags & NPU_COMPUTE_FLAG_FP16) ? 2 : 4;
    if (!sram_bounds_ok(inst->src_addr, inst->num_elements * elem_size))
        return -1;
    if (!sram_bounds_ok(inst->dst_addr, inst->num_elements * elem_size))
        return -1;

    return npu_compute_relu(&engine->sram[inst->src_addr],
                            &engine->sram[inst->dst_addr],
                            inst->num_elements, inst->flags);
}

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

        switch (inst->opcode) {
        case NPU_OP_NOP:
            break;

        case NPU_OP_HALT:
            engine->halted = 1;
            engine->exit_code = inst->halt.exit_code;
            return 0;

        case NPU_OP_LOAD:
            ret = exec_load(engine, &inst->load, dma_read, opaque);
            if (ret != 0) {
                engine->error = 1;
                return -1;
            }
            break;

        case NPU_OP_STORE:
            ret = exec_store(engine, &inst->store, dma_write, opaque);
            if (ret != 0) {
                engine->error = 1;
                return -1;
            }
            break;

        case NPU_OP_MATMUL:
            ret = exec_matmul(engine, &inst->matmul);
            if (ret != 0) {
                engine->error = 1;
                return -1;
            }
            break;

        case NPU_OP_RELU:
            ret = exec_relu(engine, &inst->relu);
            if (ret != 0) {
                engine->error = 1;
                return -1;
            }
            break;

        /* TODO: implement when compute kernels are ready */
        case NPU_OP_CONV2D:
        case NPU_OP_ADD:
        case NPU_OP_SOFTMAX:
            engine->error = 1;
            return -1;

        default:
            engine->error = 1;
            return -1;
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
