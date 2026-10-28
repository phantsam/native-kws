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
