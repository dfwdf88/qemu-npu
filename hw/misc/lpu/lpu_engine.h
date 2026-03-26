/*
 * LPU Instruction Execution Engine
 *
 * Standalone fetch/decode/execute engine for LPU ISA.
 * No QEMU or kernel dependencies -- callable from QEMU lpu_pci.c
 * and from standalone gtest.
 *
 * The LPU is an inference-optimized subset of the NPU ISA (21 instructions),
 * with a larger SRAM (64MB) for KV cache persistence across decode steps.
 *
 * DMA is abstracted via callbacks so the caller provides
 * the actual host memory access (QEMU pci_dma_read/write
 * or a mock for testing).
 */

#ifndef LPU_ENGINE_H
#define LPU_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#define LPU_SRAM_SIZE  (64 * 1024 * 1024)  /* 64MB local SRAM (KV cache) */
#define LPU_INST_BYTES 64

/* DMA callback types (same signatures as NPU) */
typedef int (*lpu_dma_read_fn)(void *opaque, uint64_t addr,
                               void *buf, uint32_t size);
typedef int (*lpu_dma_write_fn)(void *opaque, uint64_t addr,
                                const void *buf, uint32_t size);

/* Engine state */
typedef struct {
    uint8_t  sram[LPU_SRAM_SIZE];
    uint32_t pc;            /* program counter (instruction index) */
    int      halted;
    uint16_t exit_code;     /* from HALT instruction */
    int      error;         /* non-zero if execution error */
} lpu_engine_t;

/* Initialize engine (clears SRAM, resets state) */
void lpu_engine_init(lpu_engine_t *engine);

/* Reset engine state without clearing SRAM (KV cache persistence) */
void lpu_engine_reset(lpu_engine_t *engine);

/*
 * Execute a program (.bin instruction stream).
 *
 * @engine:       Engine state (SRAM persists across calls)
 * @program:      Flat array of 64-byte instructions
 * @program_size: Total bytes (must be multiple of 64)
 * @dma_read:     Callback to read from host memory (for LOAD)
 * @dma_write:    Callback to write to host memory (for STORE)
 * @opaque:       Context passed to DMA callbacks
 *
 * Returns: 0 = normal HALT, -1 = error (unknown opcode, bad address, etc.)
 */
int lpu_engine_run(lpu_engine_t *engine,
                   const uint8_t *program, uint32_t program_size,
                   lpu_dma_read_fn dma_read,
                   lpu_dma_write_fn dma_write,
                   void *opaque);

#endif /* LPU_ENGINE_H */
