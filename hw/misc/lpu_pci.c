/*
 * QEMU LPU PCI Device
 *
 * Language Processing Unit -- an inference-optimized subset of the NPU
 * with 64MB SRAM for KV cache persistence and a 21-instruction ISA.
 *
 * PCI vendor 0x1234 (QEMU), device 0x11e9.
 * Register-compatible with edu.c (NPU) for driver reuse.
 *
 * Copyright (c) 2024-2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/units.h"
#include "hw/pci/pci.h"
#include "hw/pci/msi.h"
#include "qemu/timer.h"
#include "qom/object.h"
#include "qemu/main-loop.h" /* iothread mutex */
#include "qemu/module.h"
#include "qapi/visitor.h"

#include "lpu/lpu_engine.h"

#define TYPE_PCI_LPU_DEVICE "lpu"
typedef struct LpuState LpuState;
DECLARE_INSTANCE_CHECKER(LpuState, LPU,
                         TYPE_PCI_LPU_DEVICE)

#define DMA_IRQ         0x00000100
#define LPU_ERROR_IRQ   0x00000200

#define DMA_START       0x40000
#define DMA_SIZE        4096

struct LpuState {
    PCIDevice pdev;
    MemoryRegion mmio;

    uint32_t irq_status;

#define LPU_DMA_RUN             0x1
#define LPU_DMA_DIR(cmd)        (((cmd) & 0x2) >> 1)
# define LPU_DMA_FROM_PCI       0
# define LPU_DMA_TO_PCI         1
#define LPU_DMA_IRQ             0x4
    struct dma_state {
        dma_addr_t src;
        dma_addr_t dst;
        dma_addr_t cnt;
        dma_addr_t cmd;
    } dma;
    QEMUTimer dma_timer;
    char dma_buf[DMA_SIZE];
    uint64_t dma_mask;

    lpu_engine_t lpu_engine;
};

static bool lpu_msi_enabled(LpuState *lpu)
{
    return msi_enabled(&lpu->pdev);
}

static void lpu_raise_irq(LpuState *lpu, uint32_t val)
{
    lpu->irq_status |= val;
    if (lpu->irq_status) {
        if (lpu_msi_enabled(lpu)) {
            msi_notify(&lpu->pdev, 0);
        } else {
            pci_set_irq(&lpu->pdev, 1);
        }
    }
}

static void lpu_lower_irq(LpuState *lpu, uint32_t val)
{
    lpu->irq_status &= ~val;

    if (!lpu->irq_status && !lpu_msi_enabled(lpu)) {
        pci_set_irq(&lpu->pdev, 0);
    }
}

static void lpu_check_range(uint64_t xfer_start, uint64_t xfer_size,
                uint64_t dma_start, uint64_t dma_size)
{
    uint64_t xfer_end = xfer_start + xfer_size;
    uint64_t dma_end = dma_start + dma_size;

    if (dma_end >= dma_start && xfer_end >= xfer_start &&
        xfer_start >= dma_start && xfer_end <= dma_end) {
        return;
    }

    qemu_log_mask(LOG_GUEST_ERROR,
                  "LPU: DMA range 0x%016"PRIx64"-0x%016"PRIx64
                  " out of bounds (0x%016"PRIx64"-0x%016"PRIx64")!",
                  xfer_start, xfer_end - 1, dma_start, dma_end - 1);
}

static dma_addr_t lpu_clamp_addr(const LpuState *lpu, dma_addr_t addr)
{
    dma_addr_t res = addr & lpu->dma_mask;

    if (addr != res) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "LPU: clamping DMA 0x%016"PRIx64" to 0x%016"PRIx64"!",
                      addr, res);
    }

    return res;
}

static void lpu_dma_timer(void *opaque)
{
    LpuState *lpu = opaque;
    bool raise_irq = false;

    if (!(lpu->dma.cmd & LPU_DMA_RUN)) {
        return;
    }

    if (LPU_DMA_DIR(lpu->dma.cmd) == LPU_DMA_FROM_PCI) {
        uint64_t dst = lpu->dma.dst;
        lpu_check_range(dst, lpu->dma.cnt, DMA_START, DMA_SIZE);
        dst -= DMA_START;
        pci_dma_read(&lpu->pdev, lpu_clamp_addr(lpu, lpu->dma.src),
                lpu->dma_buf + dst, lpu->dma.cnt);
    } else {
        uint64_t src = lpu->dma.src;
        lpu_check_range(src, lpu->dma.cnt, DMA_START, DMA_SIZE);
        src -= DMA_START;
        pci_dma_write(&lpu->pdev, lpu_clamp_addr(lpu, lpu->dma.dst),
                lpu->dma_buf + src, lpu->dma.cnt);
    }

    lpu->dma.cmd &= ~LPU_DMA_RUN;
    if (lpu->dma.cmd & LPU_DMA_IRQ) {
        raise_irq = true;
    }

    if (raise_irq) {
        lpu_raise_irq(lpu, DMA_IRQ);
    }
}

/* LPU DMA callbacks - wrap QEMU pci_dma_read/write for lpu_engine */
static int lpu_npu_dma_read(void *opaque, uint64_t addr,
                            void *buf, uint32_t size)
{
    LpuState *lpu = opaque;
    pci_dma_read(&lpu->pdev, lpu_clamp_addr(lpu, lpu->dma.dst + addr),
                 buf, size);
    return 0;
}

static int lpu_npu_dma_write(void *opaque, uint64_t addr,
                             const void *buf, uint32_t size)
{
    LpuState *lpu = opaque;
    pci_dma_write(&lpu->pdev, lpu_clamp_addr(lpu, lpu->dma.dst + addr),
                  buf, size);
    return 0;
}

static void dma_rw(LpuState *lpu, bool write, dma_addr_t *val, dma_addr_t *dma,
                bool timer)
{
    if (write && (lpu->dma.cmd & LPU_DMA_RUN)) {
        return;
    }

    if (write) {
        *dma = *val;
    } else {
        *val = *dma;
    }

    if (timer) {
        timer_mod(&lpu->dma_timer, qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL) + 100);
    }
}

static uint64_t lpu_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    LpuState *lpu = opaque;
    uint64_t val = ~0ULL;

    if (addr < 0x80 && size != 4) {
        return val;
    }

    if (addr >= 0x80 && size != 4 && size != 8) {
        return val;
    }

    switch (addr) {
    case 0x00:
        val = 0x010000ed;   /* LPU device ID word */
        break;
    case 0x20:
        val = 0;            /* status (no factorial thread in LPU) */
        break;
    case 0x24:
        val = lpu->irq_status;
        break;
    case 0x80:
        dma_rw(lpu, false, &val, &lpu->dma.src, false);
        break;
    case 0x84:
        val = lpu->dma.src >> 32;
        break;
    case 0x88:
        dma_rw(lpu, false, &val, &lpu->dma.dst, false);
        break;
    case 0x8c:
        val = lpu->dma.dst >> 32;
        break;
    case 0x90:
        dma_rw(lpu, false, &val, &lpu->dma.cnt, false);
        break;
    case 0x98:
        dma_rw(lpu, false, &val, &lpu->dma.cmd, false);
        break;
    }

    return val;
}

static void lpu_mmio_write(void *opaque, hwaddr addr, uint64_t val,
                unsigned size)
{
    LpuState *lpu = opaque;

    if (addr < 0x80 && size != 4) {
        return;
    }

    if (addr >= 0x80 && size != 4 && size != 8) {
        return;
    }

    switch (addr) {
    case 0x60:
        lpu_raise_irq(lpu, val);
        break;
    case 0x64:
        lpu_lower_irq(lpu, val);
        break;
    case 0x80:
        dma_rw(lpu, true, &val, &lpu->dma.src, false);
        break;
    case 0x84:
        if (!(lpu->dma.cmd & LPU_DMA_RUN)) {
            lpu->dma.src = (lpu->dma.src & 0xFFFFFFFF) |
                           ((dma_addr_t)val << 32);
        }
        break;
    case 0x88:
        dma_rw(lpu, true, &val, &lpu->dma.dst, false);
        break;
    case 0x8c:
        if (!(lpu->dma.cmd & LPU_DMA_RUN)) {
            lpu->dma.dst = (lpu->dma.dst & 0xFFFFFFFF) |
                           ((dma_addr_t)val << 32);
        }
        break;
    case 0x90:
        dma_rw(lpu, true, &val, &lpu->dma.cnt, false);
        break;
    case 0x98:
        if (!(val & LPU_DMA_RUN)) {
            break;
        }
        /* LPU engine: fetch program from guest memory and execute */
        {
            dma_addr_t src = lpu_clamp_addr(lpu, lpu->dma.src);
            uint32_t cnt = (uint32_t)lpu->dma.cnt;
            uint8_t *prog = g_malloc(cnt);

            pci_dma_read(&lpu->pdev, src, prog, cnt);

            /* Skip 64-byte binary header (lpu_bin_header_t) --
             * instructions start at offset 64 */
            int lpu_rc = -1;
            if (cnt > LPU_INST_BYTES) {
                lpu_rc = lpu_engine_run(&lpu->lpu_engine,
                               prog + LPU_INST_BYTES,
                               cnt - LPU_INST_BYTES,
                               lpu_npu_dma_read, lpu_npu_dma_write, lpu);
            }
            g_free(prog);

            lpu->dma.cmd &= ~LPU_DMA_RUN;
            if (lpu_rc == 0) {
                lpu_raise_irq(lpu, DMA_IRQ);
            } else {
                qemu_log_mask(LOG_GUEST_ERROR,
                    "LPU: engine_run FAILED (rc=%d)\n", lpu_rc);
                lpu_raise_irq(lpu, DMA_IRQ | LPU_ERROR_IRQ);
            }
        }
        break;
    }
}

static const MemoryRegionOps lpu_mmio_ops = {
    .read = lpu_mmio_read,
    .write = lpu_mmio_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 8,
    },
    .impl = {
        .min_access_size = 4,
        .max_access_size = 8,
    },

};

static void pci_lpu_realize(PCIDevice *pdev, Error **errp)
{
    LpuState *lpu = LPU(pdev);
    uint8_t *pci_conf = pdev->config;

    pci_config_set_interrupt_pin(pci_conf, 1);

    if (msi_init(pdev, 0, 1, true, false, errp)) {
        return;
    }

    timer_init_ms(&lpu->dma_timer, QEMU_CLOCK_VIRTUAL, lpu_dma_timer, lpu);

    lpu_engine_init(&lpu->lpu_engine);

    memory_region_init_io(&lpu->mmio, OBJECT(lpu), &lpu_mmio_ops, lpu,
                    "lpu-mmio", 1 * MiB);
    pci_register_bar(pdev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &lpu->mmio);
}

static void pci_lpu_uninit(PCIDevice *pdev)
{
    LpuState *lpu = LPU(pdev);

    timer_del(&lpu->dma_timer);
    msi_uninit(pdev);
}

static void lpu_instance_init(Object *obj)
{
    LpuState *lpu = LPU(obj);

    lpu->dma_mask = UINT64_MAX;
    object_property_add_uint64_ptr(obj, "dma_mask",
                                   &lpu->dma_mask, OBJ_PROP_FLAG_READWRITE);
}

static void lpu_class_init(ObjectClass *class, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(class);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(class);

    k->realize = pci_lpu_realize;
    k->exit = pci_lpu_uninit;
    k->vendor_id = PCI_VENDOR_ID_QEMU;
    k->device_id = 0x11e9;
    k->revision = 0x10;
    k->class_id = PCI_CLASS_OTHERS;
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static const TypeInfo lpu_types[] = {
    {
        .name          = TYPE_PCI_LPU_DEVICE,
        .parent        = TYPE_PCI_DEVICE,
        .instance_size = sizeof(LpuState),
        .instance_init = lpu_instance_init,
        .class_init    = lpu_class_init,
        .interfaces    = (const InterfaceInfo[]) {
            { INTERFACE_CONVENTIONAL_PCI_DEVICE },
            { },
        },
    }
};

DEFINE_TYPES(lpu_types)
