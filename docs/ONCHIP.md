# Milestone 1: run the network on the ESP32, with no microphone

**Goal.** Get a trained keyword-spotting network running on the ESP32, fed four
log-mel inputs baked into flash, and confirm it predicts the same classes the host
does. Then report how long one inference takes.

No microphone, no audio capture, no feature extraction on the device. Those are
separate problems and mixing them in makes failures impossible to attribute. This
milestone answers one question: **does our network run correctly on this chip.**

## What to deploy, and why it is not our best model

Deploy a **small BC-ResNet scale** — BC-ResNet-2 or BC-ResNet-3 — not BC-ResNet-8
and not NativeKWS-H. BC-ResNet-8 (98.69 %) was the first attempt and does not fit;
see "Which model actually fits" below.

NativeKWS-H is the better model (99.02 %) but it contains a *selective scan*, a
sequential recurrence, and **ESP-DL has no scan or cumsum operator** — verified from
Espressif's operator support table and recorded in `docs/LITERATURE.md` §6. It
cannot be exported as-is. BC-ResNet-8 is pure convolution, every operator is on the
whitelist, and it is already trained and committed here. It proves the whole
toolchain with none of that risk.

Two routes for NativeKWS afterwards, neither of them this milestone:
1. a custom ESP-DL kernel for the scan, or
2. statically unrolling the 101-step recurrence into Mul/Add pairs, which uses only
   whitelisted operators at the cost of a very large graph.

Choosing between those is a real research decision — note that all three published
Mamba-on-MCU papers dodge it by keeping the recurrence in float. Do not start it here.

## Already prepared for you, in `deploy/onchip/<model>/`

One directory per model: `bcresnet-1/`, `bcresnet-2/`, `bcresnet-3/`, `bcresnet-8/`.
Each holds the same four things, and **the files inside a directory belong together** —
mixing an ONNX with another directory's calibration or test vectors gives a silently
wrong result.

| file | what it is |
|---|---|
| `<model>.onnx` | the network, opset 13, fixed batch 1, input `logmel` (1,1,40,101), **BatchNorm already folded out** |
| `calib_128.npy` | 128 real log-mel features for int8 post-training quantization |
| `test_vectors.h` | 4 inputs (`yes`, `no`, `stop`, `_silence_`) + the float logits the host produced |
| `test_vectors.npz` | the same, for host-side checking |
| `export_info.json` | provenance: source checkpoint, accuracy, input convention |

Regenerate with `python scripts/export_onchip.py` (needs the dataset cache, so that
runs on the training machine).

**The graph is already ESP-DL-clean — do not "fix" it.** A plain PyTorch export
contains `BatchNormalization`, which ESP-DL has no operator for. The export script
removes all of it: onnxsim folds what sits behind a Conv, and the 12 that survive
(BC-ResNet's SubSpectralNorm normalises a *reshaped* tensor, so its BN is not
directly behind a Conv) become **depthwise `Conv` nodes with kernel 1**, carrying the
same affine in weights and bias.

Those 12 were first written as `Mul` + `Add`. The device rejected 5 of them on
ESP-DL's int8 `Mul` constraint `output_exponent - input0_exponent - input1_exponent
>= 0` (E26), so they became depthwise `Conv`, which did measurably help — at τ=8 the
`stop` margin doubled, 2.0 → 4.25 (E29).

**But the reason given for that fix was wrong, and it matters.** E27 claimed a
depthwise Conv "gets a per-output-channel scale of its own, so the constraint does
not arise". **On the ESP32 it does not.** ESP-DL quantizes *per tensor* on ESP32 and
ESP32-S3; per-channel Conv exists only on ESP32-P4/S31. So one power-of-two exponent
must cover a fold whose channel scales span three orders of magnitude, and the small
channels round to zero. E29 measured BC-ResNet-2's `bnfold_conv_0` at a **581×
spread, 39 of 80 channels zeroed, 81.9 % layerwise noise** — which cost that run a
test vector. The fix was right in kind and insufficient in degree.

**What actually addresses it: cross-layer equalization, applied in the export.** The
reshape in front of a fold sends Conv output channel `k` to fold channels
`k*S … k*S+S-1`, so multiplying the Conv's channel `k` by `g[k]` and dividing those
`S` fold channels by `g[k]` is an **exact identity**. `g` is the geometric mean over
the sub-bands, damped by a per-layer exponent `alpha` chosen from a small grid,
because moving the whole spread fixes the fold but inflates the Conv feeding it and
which trade wins differs by layer. A layer is rewritten only if some `alpha` strictly
reduces the destroyed-channel count over the *pair*, and only when the Conv's output
has no other consumer (otherwise rescaling it would corrupt a residual branch).

Channels left with ≤2 quantization levels, before → after equalization:

| model | destroyed channels | worst layer |
|---|---|---|
| BC-ResNet-1 | 87 → **40** | — |
| BC-ResNet-2 | 249 → **74** | `bnfold_conv_0` 72 → **1** |
| BC-ResNet-3 | 324 → **56** | `bnfold_conv_1` 78 → **4** |
| BC-ResNet-8 | 676 → **51** | `bnfold_conv_0` 123 → **3** |

Verified exact: all four graphs still match PyTorch to ≤1e-5 with unchanged argmax.
This is a screening metric, not an accuracy measurement — esp-ppq's layerwise noise
table is the real check, so re-quantize and compare before trusting it.

**Moving to an ESP32-S3 would not remove this problem.** From ESP-DL's quantization
guide: ESP32 and **ESP32-S3 are both per-tensor**; per-channel `Conv`/`GEMM` exists
only on **ESP32-P4 and ESP32-S31**. So the S3 buys roughly 7x speed and a PSRAM
option — it fixes the latency and memory walls — while the channel-scale spread stays
exactly as it is. That is also why the equalization above is worth having rather than
waiting for better silicon: per-tensor covers ESP32, ESP32-S3 and ESP32-C3, which is
most of the installed base.

Verified: operators are `Add, Conv, Flatten, GlobalAveragePool, Mul, ReduceMean,
Relu, Reshape, Sigmoid`, **all whitelisted** (the remaining `Mul`s are the
architecture's own SiLU, `x*sigmoid(x)`, which quantized without complaint); the
rewritten graph matches PyTorch to 1.1e-05.

## Which model actually fits

**BC-ResNet-8 does not fit on an ESP32 without PSRAM, and no quantization setting
changes that.** Its largest single activation is the stem output `[1,128,20,101]`,
which is 252.5 KB in int8. The ESP-DL arena asked for 391 KB; the chip had ~366 KB
free at boot but a largest *contiguous* block of only 108 KB at the point of
allocation. Weights were never the problem — they stayed in flash as intended.

The arena holds more than one tensor at a time. The τ=8 run measured a **391.38 KB
arena against a 252.5 KB largest tensor, so arena ≈ 1.55 × largest tensor** under
`MEMORY_MANAGER_GREEDY`. Applying that ratio against the 108 KB largest free block
observed after the model loads:

| model | params | largest activation | arena ≈ 1.55x | fits in 108 KB? |
|---|---|---|---|---|
| BC-ResNet-1 | 9,232 | 31.6 KB | ~49 KB | **yes, comfortably** |
| BC-ResNet-2 | 27,284 | 63.1 KB | ~98 KB | yes, but only 10 KB spare |
| BC-ResNet-3 | 54,168 | 94.7 KB | ~147 KB | **no** |
| BC-ResNet-4 | 89,884 | 126.2 KB | ~196 KB | no |
| BC-ResNet-6 | 187,812 | 189.4 KB | ~294 KB | no |
| BC-ResNet-8 | 321,068 | 252.5 KB | 391 KB measured | no — this is what failed |

**Deploy BC-ResNet-1 first** to get a green run, then try BC-ResNet-2. τ=3 and above
need either streaming execution or a board with PSRAM. (An earlier version of this
file said to start at τ=2 and that τ=3 was "tight"; that predated the measured 1.55x
arena ratio and was wrong.)

**Measured on device (E29), superseding the estimates above.** τ=2 runs: arena
**98.64 KB**, **949 ms** per inference, `model->test()` PASS so the chip is bit-exact
with esp-ppq, **3/4** test vectors correct (`no` → `_unknown_`, caused by the
`bnfold_conv_0` damage described above, not by anything on the chip). τ=3 needs a
**191.75 KB** arena and does not fit.

Two corrections to earlier guesses in this file:

* **The 1.55× arena ratio does not generalise.** Measured: τ=2 1.56×, τ=8 1.55×, but
  τ=3 **2.02×**. The greedy manager's packing depends on the graph, so the arena has
  to be measured per model, not extrapolated.
* **`Input cap=0x804 can not callocate with MALLOC_CAP_SIMD` is a symptom, not a
  configuration bug.** This file previously guessed it was a capability
  misconfiguration worth fixing. It is not: `0x804` is
  `MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT`, it is ESP-DL reporting the request that
  failed, and it appears only when the arena allocation fails (τ=3, τ=8) and not at
  τ=2. Ignore it and read the allocation size instead.
* The usable ceiling is **~108 KB contiguous**, not the 152 KB free at boot: ESP-DL
  allocates ~80–97 KB of other model structures first and splits the large block.

**All three small scales are trained and exported**, each in its own directory with
its own ONNX, calibration tensors and test vectors:

| directory | params | float accuracy (official V2-12) |
|---|---|---|
| `deploy/onchip/bcresnet-1/` | 9,232 | 96.91 % |
| `deploy/onchip/bcresnet-2/` | 27,284 | **98.02 %** |
| `deploy/onchip/bcresnet-3/` | 54,168 | 98.32 % |

(The accuracies here are measured in fp32 with no autocast, so they differ by up to
0.05 from the best-by-validation figures in `RESULTS.md`, which are evaluated in
bf16. Same checkpoints.)

Regenerate or export another run with:

```bash
python scripts/export_onchip.py --run runs/v2_12_bcresnet-2    # on the training machine
```

which prints the largest activation, so whether it fits is known before flashing.

### Deploying BC-ResNet-2

Note that `scripts/quantize_onchip.py` (on the `e26-esp32-bringup` branch) hard-codes
`deploy/onchip/bcresnet8.onnx`, a path that no longer exists — the τ=8 export moved to
`deploy/onchip/bcresnet-8/`, so the old default now fails loudly instead of silently
loading the wrong model. Give it a `--src` argument pointing at the directory
you want (its `test_vectors.npz` and `calib_128.npy` must come from the *same*
directory as the ONNX, or the calibration and the graph will not correspond):

```bash
python scripts/quantize_onchip.py --src deploy/onchip/bcresnet-2 \
    --out deploy/esp32_kws/main/models/bcresnet-2.espdl
```

Then rebuild with the new `test_vectors.h` from that directory and the new `.espdl`.

Note what this says about the project as a whole: NativeKWS-H's whole-window peak is
189 KB, so **the final model does not fit this chip whole-window either**. Its
streaming peak is ~24 KB. That gap is not an accident of this board — it is the
open problem recorded in `docs/LITERATURE.md` §11, that no MCU toolchain plans
streaming execution and every one of them is whole-window. Reaching it by hand here
is the contribution, not a workaround.

## Toolchain

**ESP-DL is an ESP-IDF component, not an Arduino library.** The Arduino IDE will not
build this. You need **ESP-IDF release/v5.3 or newer**.

ESP-DL does support the plain ESP32: quantize with the esp-ppq target `"c"`, and
build the project with target `esp32`. Be aware that ESP-DL's ESP32 operators are
plain C with no vector acceleration, so inference will be several times slower than
the ESP32-S3 numbers in Espressif's benchmarks. That is expected and is not a bug to
optimise away — slow is fine for this milestone.

## Steps

1. Install ESP-IDF ≥ 5.3 and export its environment. Confirm with `idf.py --version`.
2. `pip install esp-ppq`. Quantize the chosen `<model>.onnx` to int8 `.espdl`, following the
   ESP-DL quantization tutorial, with: target `"c"`, input shape `[1,1,40,101]`, and
   the calibration tensors from `calib_128.npy`. Read the tutorial for the current
   API rather than guessing it. Keep the quantization report it prints — if any layer
   falls back or any operator is unsupported, that report is the finding, so paste it.
3. Create an ESP-IDF project with the esp-dl component, `idf.py set-target esp32`.
4. Embed the `.espdl` model and `#include "test_vectors.h"`. For each of the 4 inputs,
   run the model and print: the predicted class index and name, the 12 output values,
   and the elapsed time (`esp_timer_get_time()`) for the inference alone.
5. Flash and run.

## Success criteria

* **All 4 predictions match `tv_expected_label`.** That is the pass/fail.
* The 12 on-chip outputs are close to `tv_host_logits` — int8 quantization will shift
  them, so judge by argmax and margin, not by equality.
* One latency figure in milliseconds, and the reported SRAM/flash usage from the build.

If predictions are wrong but the model loads, the most likely causes, in order: the
input is being fed in the wrong layout (expected is 40 mel bands x 101 frames,
mel-major), the input is being quantized with the wrong scale, or the calibration
step silently failed.

## Constraints

* 321 K int8 weights is ~321 KB. The ESP32 has ~320 KB usable DRAM, so **weights
  must stay in flash**, not be copied to SRAM. If you hit an allocation failure, that
  is the cause.
* Batch size is 1 and the input shape is fixed. Do not make it dynamic.
* ESP-DL's `Softmax` outputs float32 and breaks the int8 chain, so take the argmax
  over the raw logits; do not add a Softmax to the graph.
* **Do not modify** `deploy/onchip/*` or anything under `kws/`. Regenerate exports
  with the script instead.

## Report back

1. `idf.py --version` and the esp-ppq version
2. the full esp-ppq quantization report, including any unsupported-operator warnings
3. the build's size summary
4. the device output for all 4 test vectors, and the measured inference time
5. whatever failed, with the actual error text

Append what you did to `docs/EXPERIMENTS.md` as a new numbered entry, negative
results included.


---

# Milestone 2: test-set accuracy measured on the chip

Milestone 1 ran four clips. Four clips is not an accuracy measurement — one flip
moves it 25 points — so the headline number from E29, "3/4", says almost nothing
about how good the deployed model is. This milestone produces the real number, and
**it needs no new hardware**: 4,890 clips at ~0.95 s is about **77 minutes** on the
plain ESP32 you already have.

## How it works

The board has no dataset, so the features travel as a file. `app_eval.cpp` replaces
`app_main.cpp` and turns the device into a server: the host sends one 40x101 float32
log-mel over UART, the device quantizes it **with its own input exponent** — the same
path `app_main.cpp` already uses for `tv_input`, so no scale is duplicated on the
host — runs the model, and replies with the predicted class, the 12 int8 logits and
the inference time.

Everything is pre-built:

| file | what |
|---|---|
| `deploy/onchip/eval_firmware/app_eval.cpp` | device-side evaluation server |
| `deploy/onchip/eval_features_1200.npz` | 1,200-clip stratified subset, 8 MB, ~19 min |
| `deploy/onchip/eval_features.npz` | the full 4,890-clip test set, 32 MB, ~77 min |
| `scripts/eval_onchip.py` | host driver: streams clips, tallies, saves after every clip |
| `scripts/make_eval_features.py` | regenerates either file (deterministic, seed 0) |

Both feature files also carry the **host float32 prediction for every clip**, so the
report is a paired comparison — same model, same clips — and the difference is int8
quantization and nothing else. Host float32 accuracy: **98.02 %** on the full set,
97.33 % on the 1,200 subset.

## Running it

Build with `app_eval.cpp` in place of `app_main.cpp` (`SRCS app_eval.cpp` in
`main/CMakeLists.txt`, same `models/model.espdl` embed), flash, then **close the
serial monitor** and:

```bash
# 19-minute smoke run first -- confirms the plumbing before committing 77 minutes
python scripts/eval_onchip.py --features deploy/onchip/eval_features_1200.npz \
    --port COM5 --out onchip_eval_bcresnet-2_1200.json

# the real number
python scripts/eval_onchip.py --features deploy/onchip/eval_features.npz \
    --port COM5 --out onchip_eval_bcresnet-2.json
```

Results are written after every clip and `--resume` continues an interrupted run, so
a dropped USB connection costs minutes rather than the whole hour.

## What it prints

```
ON-CHIP int8 accuracy    : xx.xx%
host float32, same clips : 98.02%
cost of int8             : -x.xx points
chip disagrees with float: N of 4890 clips
mean inference           : ~949 ms
top confusions: ...
```

**That `cost of int8` line is the deliverable of this phase.** It is the number that
turns "the model runs on the chip" into a result, and it is the one figure the
published MCU KWS literature usually reports from host simulation rather than from
silicon.

## Verified in simulation

The host driver was tested against an emulated device on a virtual serial port that
replies with the host float prediction: accuracy came back identical to the host
float accuracy with **0 disagreements over 120 clips**, which confirms the indices,
framing, parsing and tallying all line up. `--resume` was tested by stopping at 60
clips and continuing to 120. Not yet verified against the real firmware.


---

# Milestone 3: streaming execution — the reason no ESP32-S3 is needed

Whole-window inference recomputes the entire 101-frame activation map for every
decision. Streaming keeps a small ring buffer per temporal layer and computes only
the new time column. The structure of BC-ResNet makes this unusually cheap: its
temporal convolution `f1` runs on the **frequency-averaged** signal
(`x.mean(dim=2)`), so the history a layer must retain is `C x 2*dilation` values, not
`C x F x 2*dilation`.

Computed from the graph (`causal=True`, history = `(k_t - 1) * dilation` input columns
per temporal conv, peak = history + a double-buffered working column):

| model | params | non-causal | **causal** | whole-window | streaming | reduction | fits 108 KB? |
|---|---|---|---|---|---|---|---|
| BC-ResNet-1 | 9,232 | 96.81 % | 96.44 % | 31.6 KB | **3.0 KB** | 10x | both |
| BC-ResNet-2 | 27,284 | 98.02 % | **97.67 %** | 63.1 KB | **5.9 KB** | 11x | both |
| BC-ResNet-3 | 54,168 | 98.36 % | 98.18 % | 94.7 KB | **8.8 KB** | 11x | streaming only |
| BC-ResNet-4 | 89,884 | — | — | 126.2 KB | **11.7 KB** | 11x | streaming only |
| BC-ResNet-6 | 187,812 | — | — | 189.4 KB | **17.5 KB** | 11x | streaming only |
| BC-ResNet-8 | 321,068 | 98.67 % | 98.59 % | 252.5 KB (391 KB arena) | **23.3 KB** | 11x | streaming only |

The causal column is the accuracy a streaming deployment actually gets, since streaming
cannot use look-ahead (E33). Whole-window memory is unchanged by causality - only the
padding moves.

**Streaming puts BC-ResNet-8 on a plain ESP32, which cannot hold it whole-window at
any quantization setting.** That is a stronger demonstration than running it on an
ESP32-S3 with PSRAM, because here the constraint actually binds: 391 KB requested
against a 108 KB largest free block, versus 23.3 KB.

Latency is a *separate* constraint and streaming resolves it at different scales.
From the measured whole-window throughput (τ=2: 7.32 M MACs in 949 ms = **7.7
MMAC/s** with ESP-DL's plain-C kernels), per-frame cost if streamed:

| model | MACs/frame | estimated ms/frame | real time (10 ms budget)? |
|---|---|---|---|
| BC-ResNet-1 | 24.6 k | ~3.2 ms | yes, with room for the front-end |
| BC-ResNet-2 | 72.5 k | ~9.4 ms | borderline — no headroom for log-mel |
| BC-ResNet-3 | 143.8 k | ~18.7 ms | no |
| BC-ResNet-8 | 850.7 k | ~110 ms | no — fits in memory, not in time |

**Treat that column as an estimate, not a result.** It extrapolates a whole-window
throughput that includes the 16 % Transpose tax and the quantize/requantize overhead,
and streaming will have *worse* per-MAC efficiency: tiny per-call tensors mean poorer
data locality and more fixed overhead per invocation. It needs measuring.

## Prerequisite, now in place

`BCResNet` and `BCBlock` take `causal=True`, which puts left-only padding on the stem
(5-wide in time), on every `f1` temporal conv, and on the head's 5-wide temporal
depthwise conv — the last place look-ahead hides. Verified:

* parameter counts are **identical** to the non-causal models (only padding changes);
* the non-causal path is untouched — existing checkpoints load and reproduce their
  stored logits to 2.9e-06;
* **zero look-ahead**: randomising frames ≥60 leaves every feature column before 60
  bit-identical, where the non-causal model leaks 12 frames (120 ms) backwards.

**The accuracy cost of losing look-ahead is measured, not assumed** (E33). Causal
BC-ResNet-1/2/3/8 trained on v2_12 with the official SGD recipe, identical to their
non-causal counterparts:

| model | non-causal | causal | cost | last EMA, non-causal -> causal |
|---|---|---|---|---|
| BC-ResNet-1 | 96.81 | 96.44 | **-0.37** | 96.99 -> 96.56 (-0.43) |
| BC-ResNet-2 | 98.02 | 97.67 | **-0.35** | 98.02 -> 97.67 (-0.35) |
| BC-ResNet-3 | 98.36 | 98.18 | **-0.18** | 98.40 -> 98.12 (-0.29) |
| BC-ResNet-8 | 98.67 | 98.59 | **-0.08** | 98.83 -> 98.63 (-0.20) |

Streaming therefore costs **0.08-0.37 points**, and the cost shrinks as the model grows -
larger models have enough capacity to make up for the missing 120 ms of future context.
So the streaming deployment target is causal BC-ResNet-2 at **97.67 %**, and the int8
measurement of that model on the board is the remaining number.
