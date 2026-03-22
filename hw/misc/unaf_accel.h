/*
 * UNAF Accelerator Base Device (QOM Abstract Parent)
 *
 * Provides shared PCI boilerplate for all accelerator types.
 * Each chip type (NPU, LPU, ...) inherits from this and only
 * provides its engine_init/engine_run callbacks.
 *
 * Usage (QEMU command line):
 *   -device edu          (NPU, existing)
 *   -device unaf-lpu     (LPU, future)
 *
 * NOTE: This header is prepared for future QOM refactoring.
 * Currently edu.c still operates standalone.
 */

#ifndef UNAF_ACCEL_H
#define UNAF_ACCEL_H

#include <stdint.h>
#include <stddef.h>

/* DMA callback types (used by all accelerator engines) */
typedef int (*accel_dma_read_fn)(void *opaque, uint64_t addr,
                                  void *buf, uint32_t size);
typedef int (*accel_dma_write_fn)(void *opaque, uint64_t addr,
                                   const void *buf, uint32_t size);

/*
 * Engine interface contract:
 * Every accelerator engine must implement these functions.
 * The engine is pure C with no QEMU dependencies.
 *
 * void xxx_engine_init(xxx_engine_t *engine);
 * void xxx_engine_reset(xxx_engine_t *engine);
 * int  xxx_engine_run(xxx_engine_t *engine,
 *                     const uint8_t *program, uint32_t program_size,
 *                     accel_dma_read_fn dma_read,
 *                     accel_dma_write_fn dma_write,
 *                     void *opaque);
 *
 * Returns: 0 = normal HALT, -1 = error
 */

#endif /* UNAF_ACCEL_H */
