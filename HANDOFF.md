# Handoff index

**NativeKWS** — keyword spotting designed to compete with large models while actually
running on a microcontroller. Two machines do the work, and each has its own self-contained
brief. Start with the one you are sitting at.

| you are on… | read | what lives there |
|---|---|---|
| the **GPU server** | **[`docs/HANDOFF_TRAINING.md`](docs/HANDOFF_TRAINING.md)** | dataset caches, `kws/`, `runs/`, training queue, the exports in `deploy/onchip/` |
| the **computer the board plugs into** | **[`docs/HANDOFF_DEVICE.md`](docs/HANDOFF_DEVICE.md)** | ESP-IDF, esp-ppq quantization, `deploy/esp32_kws/`, the serial evaluation driver |
| the **microphone phase** (deferred) | [`docs/HANDOFF_MIC.md`](docs/HANDOFF_MIC.md) | INMP441 wiring, `rig/`, the recording protocol |

Neither brief needs the other machine's code, and neither needs this file.

## Where things stand

**Accuracy — NativeKWS-H + NASS, 301 K parameters** (3 seeds, final epoch; `README.md` has
the caveats):

| V2-12 | V1-12 | V2-35 |
|---|---|---|
| **99.02 ± 0.05** | **98.13 ± 0.04** | **98.07 ± 0.09** |

V2-12 and V2-35 beat the best published from-scratch result (Keyword Mamba KWM-T-192,
5.2 M parameters) with **17× fewer parameters**.

**On silicon — BC-ResNet-2, 27,284 parameters, int8 on a plain ESP32 with no PSRAM:**

| full GSC v2-12 test set | float, same clips | cost of int8 | per 1-s window | activation arena |
|---|---|---|---|---|
| **97.06 %** (4,890 clips) | 98.02 % | 0.96 points | 951.6 ms | 98.64 KB |

The chip is **bit-exact with esp-ppq's host simulation** on 4,890/4,890 clips, so most
remaining quantization work needs no board (E32).

**Streaming is the open contribution.** Whole-window, BC-ResNet-8 asks for a 391 KB arena
against a 108 KB largest free block and cannot run at all. Streamed, the same network peaks
at **23.3 KB** — so every scale fits the chip you already have. The causal models that
streaming requires are trained and cost only **0.08–0.37 points** of accuracy (E33). The
per-frame step graph itself is **not built yet**, and the latency figures in
[`docs/ONCHIP.md`](docs/ONCHIP.md) are extrapolations, clearly labelled as such.

> An earlier version of this file said milestone 1 was the current task and that an
> **ESP32-S3 is required** for deployment. Both are out of date: milestones 1 and 2 are
> done on the base ESP32, and E31 showed the S3 was never required — it is per-tensor too,
> so it does not fix the quantization problem, and the base chip's binding memory limit is
> exactly what makes the streaming result worth having.

## The log

[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) is the chronological record — hypothesis,
command, result, decision — for both machines, in **one shared numbering** (E26, E29 and
E32 came from the device side; the next free number is **E34**). Negative results are in
there deliberately; several of the most useful entries are corrections of earlier claims.

Also: [`docs/RESULTS.md`](docs/RESULTS.md) (auto-generated tables),
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/ONCHIP.md`](docs/ONCHIP.md) (the
three on-chip milestones), [`docs/DATASET.md`](docs/DATASET.md) (the pre-registered
microphone protocol), [`docs/LITERATURE.md`](docs/LITERATURE.md).
