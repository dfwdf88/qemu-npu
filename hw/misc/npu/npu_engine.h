/*
 * NPU Instruction Execution Engine
 *
 * Standalone fetch/decode/execute engine for NPU ISA.
 * No QEMU or kernel dependencies — callable from QEMU edu.c
 * and from standalone gtest.
 *
 * DMA is abstracted via callbacks so the caller provides
 * the actual host memory access (QEMU pci_dma_read/write
 * or a mock for testing).
 */

#ifndef NPU_ENGINE_H
#define NPU_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#define NPU_SRAM_SIZE  (1 * 1024 * 1024)  /* 1MB local SRAM */
#define NPU_INST_BYTES 64

/* DMA callback types (provided by caller) */
typedef int (*npu_dma_read_fn)(void *opaque, uint64_t addr,
                               void *buf, uint32_t size);
typedef int (*npu_dma_write_fn)(void *opaque, uint64_t addr,
                                const void *buf, uint32_t size);

/* Engine state */
typedef struct {
    uint8_t  sram[NPU_SRAM_SIZE];
    uint32_t pc;            /* program counter (instruction index) */
    int      halted;
    uint16_t exit_code;     /* from HALT instruction */
    int      error;         /* non-zero if execution error */
} npu_engine_t;

/* Initialize engine (clears SRAM, resets state) */
void npu_engine_init(npu_engine_t *engine);

/* Reset engine state without clearing SRAM */
void npu_engine_reset(npu_engine_t *engine);

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
int npu_engine_run(npu_engine_t *engine,
                   const uint8_t *program, uint32_t program_size,
                   npu_dma_read_fn dma_read,
                   npu_dma_write_fn dma_write,
                   void *opaque);

#endif /* NPU_ENGINE_H */
