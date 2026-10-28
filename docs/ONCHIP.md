# Milestone 1: run the network on the ESP32, with no microphone

**Goal.** Get a trained keyword-spotting network running on the ESP32, fed four
log-mel inputs baked into flash, and confirm it predicts the same classes the host
does. Then report how long one inference takes.

No microphone, no audio capture, no feature extraction on the device. Those are
separate problems and mixing them in makes failures impossible to attribute. This
milestone answers one question: **does our network run correctly on this chip.**

## What to deploy, and why it is not our best model

Deploy **BC-ResNet-8** (321,068 parameters, 98.69 % on the official V2-12 test set),
not NativeKWS-H.

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
| `bcresnet8.onnx` | the network, opset 13, fixed batch 1, input `logmel` (1,1,40,101) |
| `calib_128.npy` | 128 real log-mel features for int8 post-training quantization |
| `test_vectors.h` | 4 inputs (`yes`, `no`, `stop`, `_silence_`) + the float logits the host produced |
| `test_vectors.npz` | the same, for host-side checking |
| `export_info.json` | provenance: source checkpoint, accuracy, input convention |

Regenerate with `python scripts/export_onchip.py` (needs the dataset cache, so that
runs on the training machine).

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
