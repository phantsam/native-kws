# Handoff: the device side — ESP32 deployment (as of 2026-11-24)

For whoever continues the on-chip work on the **Windows laptop the boards plug into**.
The training side (dataset, GPUs, `kws/`, `runs/`, exports) lives on a separate server
and has its own handoff. Read this file, then [`ONCHIP.md`](ONCHIP.md), then
E26–E33 in [`EXPERIMENTS.md`](EXPERIMENTS.md). You do not need the training code.

---

## 1. Where things stand

* **BC-ResNet-2 (27,284 parameters) runs int8 on a plain ESP32 with no PSRAM:
  97.06 % on the full GSC v2-12 test set (4,890 clips), measured on the chip**, against
  98.02 % in float on the same clips — **0.96 points** of int8 cost, 102 clips where
  chip and float disagree, **951.6 ms** per 1-s window, 98.64 KB activation arena,
  weights in flash (E32).
* **The chip is bit-exact with esp-ppq's host simulation:** identical int8 logits on
  4,890 / 4,890 clips. `scripts/sim_onchip_int8.py` therefore predicts on-chip accuracy
  in ~4 minutes; the board run takes ~92. Screen on the host, confirm on the board.
* **Almost all of the int8 cost is one word.** `go` drops 98.3 % → 88.1 % and accounts
  for 41 of the 47 clips lost net (top confusions `go → down` ×16, `go → _unknown_` ×15,
  `go → no` ×7). It is input-dependent — int8 lowers the `go` logit by 5.47 on `go` clips
  but 0.78 elsewhere — so some layer's quantization destroys the evidence for *go*. Not
  yet localized (§5.1).
* Milestones in `ONCHIP.md`: **1 done** for τ=2 (4/4 embedded vectors after the
  cross-layer equalization of E30), **2 done** (E32), **3 (streaming) not started on the
  device**. τ=1 was never deployed. τ=3 and τ=8 do not fit whole-window on this board.
* **A second board has arrived** (described as the newer ESP32 with a camera). It has not
  been plugged in or identified yet — §6.

## 2. Where things are on this laptop

| what | where |
|---|---|
| repo of record | `D:\esp\native-kws`, branch `e26-esp32-bringup` (remote `github.com/phantsam/native-kws`) |
| **not** the repo | `C:\Users\Rakshit Joshi\Downloads\native-kws-main\native-kws-main` is an old unzipped snapshot (the desktop session may open there). `C:\Users\Rakshit Joshi` is itself a git repo with an unrelated remote — never commit there. |
| ESP-IDF | v5.4.2 in `D:\esp\esp-idf`, tools in `D:\esp\.espressif` (`IDF_TOOLS_PATH`) |
| Python for quantization and eval | `D:\esp\ppq-venv`: Python 3.11.6, esp-ppq 1.3.13, torch 2.15.0+cpu, onnx 1.19.1, onnxruntime 1.31.1, numpy 2.4.6, pyserial 3.5 |
| ESP-DL | 3.3.14, managed component, pinned by `deploy/esp32_kws/dependencies.lock` |
| board 1 | ESP32-D0WDQ6 rev v1.0, 4 MB flash, **no PSRAM**, CP2102 bridge (USB serial `0001`), **COM5** |
| raw logs from the bring-up | `D:\esp\*.log` (not in git) |

## 3. Rules learned the hard way

1. **ESP-IDF must run from PowerShell.** From Git Bash it stops with
   `MSys/Mingw is not supported`.
   ```powershell
   $env:IDF_TOOLS_PATH = 'D:\esp\.espressif'; . D:\esp\esp-idf\export.ps1
   cd D:\esp\native-kws\deploy\esp32_kws
   idf.py build                                # app_main.cpp: the 4 embedded vectors
   idf.py -B build_eval -DKWS_APP=eval build   # app_eval.cpp: the milestone-2 server
   ```
2. **Flashing needs a human on board 1.** Auto-reset into download mode fails every time
   (`Wrong boot mode detected (0x13)`). Ask the owner to *hold BOOT, tap EN, release
   BOOT*, then from the build directory:
   ```powershell
   python -m esptool --chip esp32 -p COM5 -b 115200 --before no_reset --after hard_reset write_flash "@flash_args"
   ```
   460800 baud dropped mid-write; use a direct laptop USB port, not a hub. Resetting
   into the app afterwards works fine (RTS → EN is wired; DTR → IO0 is the flaky part).
3. **Opening COM5 resets the board**, even with DTR/RTS cleared first. Anything sent in
   the first ~0.8 s is lost. `eval_onchip.py` now waits for the firmware's `ready` line.
4. **The CP2102 cannot carry 2 Mbaud.** Nothing intelligible comes back. 921600 works.
5. **Use `scripts/serial_capture.py` to read the board**, not `idf.py monitor`, which
   needs an interactive terminal.
6. **esp-ppq rewrites the ONNX it is given, in place.** `quantize_onchip.py` hands it a
   temporary copy. Never point esp-ppq at `deploy/onchip/` directly.
7. **`deploy/onchip/*` and `kws/*` belong to the training server.** Do not edit them;
   exports are regenerated there with `scripts/export_onchip.py`. When the prepared
   eval firmware needed changes, a copy went into `deploy/esp32_kws/main/` instead.
8. The `.espdl` output is not byte-reproducible (auto-generated `PPQ_Variable_N` names)
   but is semantically identical run to run.
9. Commit or push only when the owner asks. Keep docs at LF line endings.

## 4. How the deployment is built

| file | what it does |
|---|---|
| `scripts/quantize_onchip.py` | `--src deploy/onchip/<model> --out deploy/esp32_kws/main/models/model.espdl [--target c\|esp32s3\|esp32p4] [--int16-ops ...]`. esp-ppq PTQ, w8a8, calibrated on the export's 128 tensors; embeds the "yes" vector so `model->test()` checks bit-exactness on the chip; writes `ppq_host_check.json` with host int8 results for the 4 vectors. The output is always `model.espdl`, so the embedded symbol stays `_binary_model_espdl_start`. |
| `scripts/sim_onchip_int8.py` | Same quantization into a temp file, then esp-ppq's int8 simulation over `eval_features.npz`; prints accuracy, cost, disagreements, confusions. `--chip <json>` compares a board run logit by logit. |
| `scripts/eval_onchip.py` | Host driver for the milestone-2 server: streams clips, saves after every clip, `--resume`. On board 1 pass `--baud 921600`. |
| `scripts/serial_capture.py` | Reset the board and capture N seconds of output to a file. |
| `deploy/esp32_kws/` | ESP-IDF project. The model stays in flash `.rodata` (`param_copy=false`); `main/CMakeLists.txt` includes `../../onchip/bcresnet-2` for `test_vectors.h`; `-DKWS_APP=eval` swaps in the eval server. `sdkconfig.defaults`: ESP32, 4 MB flash, 240 MHz, `-O2`, 16 KB main stack, idle-task watchdog off. |
| `main/app_main.cpp` | Milestone 1: heap log, `model->test()`, the 4 embedded vectors with chip vs host logits and margins, `profile_memory()` and `profile_module()`. |
| `main/app_eval.cpp` | Milestone 2 server, this board's version of `deploy/onchip/eval_firmware/app_eval.cpp`: same protocol (`0xA5A5A5A5`, index, 4040 float32 in; `R <idx> <pred> <us> <12 logits>` out), but **921600 baud**, and the model is loaded **before** the feature and UART buffers are allocated. The prepared file's static 16 KB buffer left an 88 KB largest block for a 98.64 KB arena, so it cannot run τ=2 on any PSRAM-less ESP32. |

**Reproduce E32 end to end** (Python commands from `D:\esp\native-kws` with
`D:\esp\ppq-venv\Scripts\python.exe`, builds from PowerShell as in §3):
```
python scripts/quantize_onchip.py --src deploy/onchip/bcresnet-2
python scripts/sim_onchip_int8.py --src deploy/onchip/bcresnet-2      # 97.06 %, ~4 min
idf.py build ; flash ; python scripts/serial_capture.py --port COM5 --seconds 15 --out out.txt   # 4/4
idf.py -B build_eval -DKWS_APP=eval build ; flash
python scripts/eval_onchip.py --features deploy/onchip/eval_features_1200.npz --port COM5 --baud 921600 --out smoke.json
python scripts/eval_onchip.py --features deploy/onchip/eval_features.npz --port COM5 --baud 921600 --out full.json
```

## 5. Results and what is known about the chip

| entry | model | outcome |
|---|---|---|
| E26 | τ=8, old Mul-form BN fold | does not fit: arena 391.38 KB vs 108 KB; 5 `bnfold_mul` deployment-rule errors |
| E29 | τ=2, depthwise-Conv fold | runs, bit-exact, 949 ms, **3/4** (`no` → `_unknown_`): ESP32 quantizes **per tensor**, so `bnfold_conv_0`'s 581× channel-scale spread zeroed 39/80 channels (81.9 % noise). τ=3 does not fit (191.75 KB arena). |
| E32 | τ=2, cross-layer equalized fold (E30) | `bnfold_conv_0` noise 0.354 %, **4/4**, **97.06 % on 4,890 clips**, 951.6 ms |

* **Latency (τ=2, 954 ms profiled):** Conv 682 ms (the stem alone 194 ms), **Transpose
  151 ms (61 ops, 15.8 %)** that esp-ppq inserts around every SubSpectralNorm `Reshape`,
  ReduceMean 48, RequantizeLinear 29, Add 24, Relu 16, Swish 3.
* **Memory (board 1):** 366 KB internal free at boot, 152 KB largest block. ESP-DL
  allocates ~97 KB of its own structures before the arena, which leaves **~108 KB
  contiguous** for the arena. Arena ÷ largest activation is 1.56× (τ=2), 2.02× (τ=3),
  1.55× (τ=8): it depends on the graph, so measure it rather than extrapolate.
* `Input cap=0x804 can not callocate with MALLOC_CAP_SIMD` is ESP-DL reporting a failed
  arena allocation, not a separate capability problem (E29).
* Images: app_main 906,750 B (41 % of the 1.5 MB partition free), app_eval 816,162 B.

### Open threads, roughly in order

1. **Localize the `go` loss — host only.** The simulation is exact, so no board is
   needed. Re-run the int8 simulation on the 402 `go` clips with one layer at a time
   kept at higher precision (`--int16-ops <onnx node name>` is already wired; esp-ppq's
   dispatching table can also keep an op in FP32 for diagnosis) and see which layer
   restores the `go` logit. Suspects: `bnfold_conv_11` (the one layer E30's equalization
   made worse, 0.46 % → 1.56 %) and the 74 channels E30 still counts as destroyed. The
   owner's standing rule for quantization problems: **report what you find before
   working around it**; a fix may belong in the export on the training server.
2. **Causal τ=2 int8 (E33).** The streaming target is causal BC-ResNet-2 (97.67 % float).
   Needs an export from the training server; then `sim_onchip_int8.py`, then the board.
3. **Milestone 3, streaming on the device.** Not started. `ONCHIP.md` has the memory
   analysis (τ=2 streams in 5.9 KB, τ=8 in 23.3 KB) and per-frame latency *estimates*.
   ESP-DL runs whole graphs, so streaming needs a per-frame step graph with the layer
   histories as explicit state inputs/outputs; that design has not been done.
4. **The Transpose tax** (151 ms, 15.8 %) comes from SubSpectralNorm's reshape. Fixing it
   is a model-design change on the training side.
5. **τ=1** was never deployed (`ONCHIP.md` asked for it first; τ=2 was deployed by
   instruction).
6. **Live microphone** (`HANDOFF.md` §5–7, INMP441 on I2S) is deferred and has never run
   on hardware.

## 6. The new board

The owner reports a newer ESP32 with a camera. **Identify it before anything else**:
```
python -m esptool -p COMx chip_id
python -m esptool -p COMx flash_id
```
Record the chip and revision, flash size, and whether esptool's feature list shows
embedded PSRAM; check Device Manager for how USB is wired (`USB JTAG/serial debug unit`,
VID 303A = the chip's native USB; otherwise a CP210x/CH34x bridge). Camera boards of this
kind are most often ESP32-S3 with 8 MB PSRAM (ESP32-S3-EYE, XIAO ESP32S3 Sense,
ESP32-S3-CAM clones), and some have an on-board microphone. An ESP32-P4 board is the
other possibility. Do not assume which.

**What changes with the chip:**

| | ESP32 (board 1) | ESP32-S3 | ESP32-P4 |
|---|---|---|---|
| Conv quantization | per tensor | **per tensor** (E30) — the equalized export is still needed | per channel — the fold spread stops mattering |
| esp-ppq target | `c` | `esp32s3` | `esp32p4` |
| expected speed | 951 ms measured | much faster: SIMD kernels; Espressif's ESP-NN benchmark is ~7× on a similar model (`HANDOFF.md` §2) — **not measured for this graph** | not estimated |
| PSRAM | none | usually 8 MB → τ=3 and τ=8 can run whole-window, arena in PSRAM | usually 32 MB |

**What the firmware needs** (all untested):
* A target-specific build: `idf.py -B build_s3 set-target esp32s3`, with its own
  sdkconfig defaults (PSRAM on, correct flash size). The current `sdkconfig.defaults` is
  written for board 1.
* Re-quantize **for that target** (`--target esp32s3`) and run `sim_onchip_int8.py` with
  the same target. **Re-establish bit-exactness with `model->test()` and a board run; do
  not assume it carries over** — the S3 uses different kernels.
* If USB goes to the chip's native USB-Serial/JTAG, `app_eval.cpp`'s UART0 transport will
  not reach the host. It would need the `usb_serial_jtag` driver, and then the baud-rate
  limit and the reset-on-open behaviour both change.
* The camera is irrelevant to keyword spotting; leave it unused. An on-board microphone
  matters for the live-mic phase, but it will likely be on different pins and possibly
  PDM, so `rig/firmware/i2s_stream` will not apply as is.

**Suggested order:**
1. Identify the board.
2. τ=2 with `app_main.cpp`: 4/4 and the latency.
3. Milestone 2 on the new chip, simulation first: the speed headline, plus the int8 accuracy for that target.
4. τ=3 and τ=8 whole-window from PSRAM: measured silicon accuracy for scales board 1 cannot hold.

Keep board 1 for the streaming milestone: E31's argument stands, since its memory limit
is what makes the streaming result meaningful. The new board is a second platform, not
a replacement.
