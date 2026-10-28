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

## Already prepared for you, in `deploy/onchip/`

| file | what it is |
|---|---|
| `bcresnet8.onnx` | the network, opset 13, fixed batch 1, input `logmel` (1,1,40,101), **BatchNorm already folded out** |
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

Those 12 were first written as `Mul` + `Add`, which is mathematically identical and
also whitelisted — and it was wrong. ESP-DL's int8 `Mul` requires
`output_exponent - input0_exponent - input1_exponent >= 0`, and the small
SubSpectralNorm scales violate it, so the deployment check rejected 5 nodes
(`bnfold_mul_0/1/2/10/11`) and the on-chip result would have been silently wrong
while esp-ppq's PC simulation still looked correct. A depthwise `Conv` gets a
per-output-channel scale of its own, so the constraint does not arise. Reported from
the device by the first bring-up run; see `docs/EXPERIMENTS.md` E26.

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

Largest single activation by scale, int8, for the fixed 101-frame window:

| model | params | largest activation | fits in ~108 KB contiguous? |
|---|---|---|---|
| BC-ResNet-1 | 9,232 | 31.6 KB | yes, comfortably |
| BC-ResNet-2 | 27,284 | 63.1 KB | yes |
| BC-ResNet-3 | 54,168 | 94.7 KB | yes, tight |
| BC-ResNet-4 | 89,884 | 126.2 KB | no |
| BC-ResNet-6 | 187,812 | 189.4 KB | no |
| BC-ResNet-8 | 321,068 | 252.5 KB | no — this is what failed |

The arena must hold more than one tensor at a time, so treat these as lower bounds
and prefer a scale with headroom. **Deploy BC-ResNet-2 first, then BC-ResNet-3.**

Export any of them with:

```bash
python scripts/export_onchip.py --run runs/v2_12_bcresnet-2    # on the training machine
```

which writes `deploy/onchip/bcresnet-2/` and prints the largest activation so the fit
is known before anything is flashed.

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
2. `pip install esp-ppq`. Quantize `bcresnet8.onnx` to int8 `.espdl`, following the
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
