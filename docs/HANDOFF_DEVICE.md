# Handoff: the device side — ESP32 deployment (as of 2026-11-24)

For whoever continues the on-chip work **on a new machine, starting cold**: nothing
installed, no access to earlier conversations. Everything you need is in this repository
except the toolchains, which §2 installs. The training side (dataset, GPUs, `kws/`,
`runs/`, model exports) lives on a separate server and has its own handoff,
[`HANDOFF_TRAINING.md`](HANDOFF_TRAINING.md), if it has been written yet.

Read this file, then [`ONCHIP.md`](ONCHIP.md), then E26–E33 in
[`EXPERIMENTS.md`](EXPERIMENTS.md). You do not need the training code.

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
  yet localized (§6, item 1).
* Milestones in `ONCHIP.md`: **1 done** for τ=2 (4/4 embedded vectors after the
  cross-layer equalization of E30), **2 done** (E32), **3 (streaming) not started on the
  device**. τ=1 was never deployed. τ=3 and τ=8 do not fit whole-window on a PSRAM-less
  ESP32.
* **Two boards.** *Board 1*, a plain ESP32 dev board, is the one every result above was
  measured on. *Board 2* arrived later: described by the owner as the newer ESP32 with a
  camera, **not yet plugged in or identified** (§7).

## 2. Set up a new machine

### 2.1 The repository

```bash
git clone https://github.com/phantsam/native-kws.git
cd native-kws
git checkout main      # if main does not contain docs/HANDOFF_DEVICE.md yet, use e26-esp32-bringup
```

The repository is public, so cloning needs no login; **pushing needs the owner's GitHub
sign-in** — let them do it, never type credentials yourself. It contains everything the
device side uses: the model exports (`deploy/onchip/<model>/`), the evaluation features
(`deploy/onchip/eval_features*.npz`, 32 MB + 8 MB), the ESP-IDF project
(`deploy/esp32_kws/`), the quantized τ=2 model that was measured
(`deploy/esp32_kws/main/models/model.espdl`), and every result file. Git LFS is not used.

### 2.2 Python (quantization, simulation, host drivers)

Python **3.11** (what every result was produced with). From the repo root:

```bash
python3.11 -m venv .venv-device                # Windows: py -3.11 -m venv .venv-device
. .venv-device/bin/activate                    # Windows: .venv-device\Scripts\activate
pip install -r deploy/esp32_kws/requirements-host.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**Check it before touching hardware.** This should reproduce E32's host numbers exactly —
`esp-ppq int8 (simulated) : 97.06%`, and identical logits to the chip on 4,890 / 4,890
clips. If it does not, the toolchain differs from the one the results came from; stop and
find out why before trusting anything downstream.

```bash
python scripts/sim_onchip_int8.py --src deploy/onchip/bcresnet-2 \
    --features deploy/onchip/eval_features.npz \
    --chip deploy/esp32_kws/onchip_eval_bcresnet-2.json
```

### 2.3 ESP-IDF v5.4.2

Install the toolchains for **both** chips you may meet (`esp32,esp32s3`; add `esp32p4` if
§7 identifies board 2 as a P4). **Keep every path free of spaces** — ESP-IDF breaks on
them, and on Windows the default tools location `%USERPROFILE%\.espressif` contains one
whenever the user name does.

*Windows* — use **PowerShell or cmd, not Git Bash** (ESP-IDF refuses MSYS:
`MSys/Mingw is not supported`):
```powershell
git clone -b v5.4.2 --recursive https://github.com/espressif/esp-idf.git C:\esp\esp-idf
$env:IDF_TOOLS_PATH = 'C:\esp\.espressif'
C:\esp\esp-idf\install.ps1 esp32,esp32s3
# then in every new shell:
$env:IDF_TOOLS_PATH = 'C:\esp\.espressif'; . C:\esp\esp-idf\export.ps1
```

*Linux:*
```bash
sudo apt install git wget flex bison gperf python3 python3-venv cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
git clone -b v5.4.2 --recursive https://github.com/espressif/esp-idf.git ~/esp/esp-idf
~/esp/esp-idf/install.sh esp32,esp32s3
. ~/esp/esp-idf/export.sh                      # in every new shell
sudo usermod -aG dialout $USER                 # serial access; log out and back in
```

*macOS:* as Linux, with the prerequisites from Homebrew (Espressif's Get Started guide
lists them).

ESP-DL is not installed by hand: the component manager downloads it on the first build,
pinned by `deploy/esp32_kws/dependencies.lock` to **esp-dl 3.3.14** (internet needed once).
Check the install with a build, which needs no board:
```
cd deploy/esp32_kws
idf.py build                                   # expect: esp32_kws.bin binary size 0xdd670 bytes ... (41%) free
```

### 2.4 USB and the serial port

* A board with a **USB-UART bridge** (board 1 has a CP2102) shows up as `COMx` on Windows
  or `/dev/ttyUSB*` on Linux. Windows may need the Silicon Labs CP210x driver (or WCH's for a
  CH34x bridge). A board wired to the chip's **native USB** (common on ESP32-S3 boards)
  shows up as `USB JTAG/serial debug unit` (VID 303A) on Windows or `/dev/ttyACM*` on
  Linux, and needs no driver.
* List ports with `python -m serial.tools.list_ports -v`. Below, `<PORT>` means that port.
* Only one program can hold a port. Close any serial monitor or Arduino IDE before running
  the scripts here.

## 3. Rules learned the hard way

1. **Board 1 needs a human to flash it.** Auto-reset into download mode failed every time
   (`Wrong boot mode detected (0x13)`). Ask the owner to *hold BOOT, tap EN, release
   BOOT*, then from the build directory, in the ESP-IDF shell:
   ```
   python -m esptool --chip esp32 -p <PORT> -b 115200 --before no_reset --after hard_reset write_flash "@flash_args"
   ```
   460800 baud dropped mid-write; use a direct USB port on the computer, not a hub.
   Resetting into the app afterwards works (RTS → EN is wired; DTR → IO0 is the flaky part).
   Other boards may auto-reset fine — try `idf.py -p <PORT> flash` first.
2. **Opening board 1's port resets it**, even with DTR/RTS cleared first. Anything sent in
   the first ~0.8 s is lost. `eval_onchip.py` waits for the firmware's `ready` line.
3. **Board 1's CP2102 cannot carry 2 Mbaud.** Nothing intelligible comes back; 921600
   works. CP2102N and CH34x bridges handle 2 Mbaud; native USB has no baud limit.
4. **Read the board with `scripts/serial_capture.py`**, not `idf.py monitor`, which needs
   an interactive terminal an agent does not have.
5. **esp-ppq rewrites the ONNX it is given, in place.** `quantize_onchip.py` hands it a
   temporary copy. Never point esp-ppq at `deploy/onchip/` directly.
6. **`deploy/onchip/*` and `kws/*` belong to the training server.** Do not edit them;
   exports are regenerated there with `scripts/export_onchip.py`. When the prepared eval
   firmware needed changes, a copy went into `deploy/esp32_kws/main/` instead.
7. The `.espdl` output is not byte-reproducible (auto-generated `PPQ_Variable_N` names)
   but is semantically identical run to run.
8. **`deploy/esp32_kws/sdkconfig` and `sdkconfig.defaults` are board 1's** (they set
   `CONFIG_IDF_TARGET="esp32"`, 4 MB flash, no PSRAM). Do not `idf.py set-target` in place
   for another chip — that rewrites them. See §7.
9. Commit or push only when the owner asks. Keep docs at LF line endings. The experiment
   log's numbering is shared with the training side: take the next free `E` number.

## 4. How the deployment is built

| file | what it does |
|---|---|
| `scripts/quantize_onchip.py` | `--src deploy/onchip/<model> --out deploy/esp32_kws/main/models/model.espdl [--target c\|esp32s3\|esp32p4] [--int16-ops ...]`. esp-ppq PTQ, w8a8, calibrated on the export's 128 tensors; embeds the "yes" vector so `model->test()` checks bit-exactness on the chip; writes `ppq_host_check.json` with host int8 results for the 4 vectors. The output is always `model.espdl`, so the embedded symbol stays `_binary_model_espdl_start`. Target `c` is the plain ESP32. |
| `scripts/sim_onchip_int8.py` | Same quantization into a temp file, then esp-ppq's int8 simulation over `eval_features.npz`; prints accuracy, cost, disagreements, confusions. `--chip <json>` compares a board run logit by logit. |
| `scripts/eval_onchip.py` | Host driver for the milestone-2 server: streams clips, saves after every clip, `--resume`. With board 1 pass `--baud 921600`. |
| `scripts/serial_capture.py` | Reset the board and capture N seconds of output to a file. |
| `deploy/esp32_kws/` | ESP-IDF project. The model stays in flash `.rodata` (`param_copy=false`); `main/CMakeLists.txt` includes `../../onchip/bcresnet-2` for `test_vectors.h`; `-DKWS_APP=eval` swaps in the eval server. `sdkconfig.defaults`: ESP32, 4 MB flash, 240 MHz, `-O2`, 16 KB main stack, idle-task watchdog off. |
| `main/app_main.cpp` | Milestone 1: heap log, `model->test()`, the 4 embedded vectors with chip vs host logits and margins, `profile_memory()` and `profile_module()`. |
| `main/app_eval.cpp` | Milestone 2 server, this project's version of `deploy/onchip/eval_firmware/app_eval.cpp`: same protocol (`0xA5A5A5A5`, index, 4040 float32 in; `R <idx> <pred> <us> <12 logits>` out), but **921600 baud on UART0**, and the model is loaded **before** the feature and UART buffers are allocated. The prepared file's static 16 KB buffer left an 88 KB largest block for a 98.64 KB arena, so it cannot run τ=2 on any PSRAM-less ESP32. |
| `deploy/esp32_kws/requirements-host.txt` | The exact Python versions (§2.2). |

**Reproduce E32 end to end on board 1** (Python from the repo root with the venv active;
`idf.py` from `deploy/esp32_kws` in the ESP-IDF shell; flashing as in §3.1):
```
python scripts/quantize_onchip.py --src deploy/onchip/bcresnet-2
python scripts/sim_onchip_int8.py --src deploy/onchip/bcresnet-2           # 97.06 %, ~4 min
idf.py build                                                               # then flash from build/ (§3.1)
python scripts/serial_capture.py --port <PORT> --seconds 15 --out out.txt  # RESULT: 4/4
idf.py -B build_eval -DKWS_APP=eval build                                  # then flash from build_eval/
python scripts/eval_onchip.py --features deploy/onchip/eval_features_1200.npz --port <PORT> --baud 921600 --out smoke.json   # ~23 min
python scripts/eval_onchip.py --features deploy/onchip/eval_features.npz --port <PORT> --baud 921600 --out full.json         # ~92 min
```
Run the long evaluations in the background and do not open the port meanwhile.

## 5. Results, and what is known about the chip

| entry | model | outcome |
|---|---|---|
| E26 | τ=8, old Mul-form BN fold | does not fit: arena 391.38 KB vs 108 KB; 5 `bnfold_mul` deployment-rule errors |
| E29 | τ=2, depthwise-Conv fold | runs, bit-exact, 949 ms, **3/4** (`no` → `_unknown_`): ESP32 quantizes **per tensor**, so `bnfold_conv_0`'s 581× channel-scale spread zeroed 39/80 channels (81.9 % noise). τ=3 does not fit (191.75 KB arena). |
| E32 | τ=2, cross-layer equalized fold (E30) | `bnfold_conv_0` noise 0.354 %, **4/4**, **97.06 % on 4,890 clips**, 951.6 ms |

* **Latency (τ=2, 954 ms profiled):** Conv 682 ms (the stem alone 194 ms), **Transpose
  151 ms (61 ops, 15.8 %)** that esp-ppq inserts around every SubSpectralNorm `Reshape`,
  ReduceMean 48, RequantizeLinear 29, Add 24, Relu 16, Swish 3.
* **Memory (board 1, ESP32-D0WDQ6, no PSRAM):** 366 KB internal free at boot, 152 KB
  largest block. ESP-DL allocates ~97 KB of its own structures before the arena, which
  leaves **~108 KB contiguous** for it. Arena ÷ largest activation is 1.56× (τ=2), 2.02×
  (τ=3), 1.55× (τ=8): it depends on the graph, so measure it rather than extrapolate.
* `Input cap=0x804 can not callocate with MALLOC_CAP_SIMD` is ESP-DL reporting a failed
  arena allocation, not a separate capability problem (E29).
* Images: app_main 906,750 B (41 % of the 1.5 MB partition free), app_eval 816,162 B.

## 6. Open threads, roughly in order

1. **Localize the `go` loss — host only.** The simulation is exact, so no board is
   needed. Re-run the int8 simulation on the 402 `go` clips with one layer at a time kept
   at higher precision (`--int16-ops <onnx node name>` is already wired; esp-ppq's
   dispatching table can also keep an op in FP32 for diagnosis) and see which layer
   restores the `go` logit. Suspects: `bnfold_conv_11` (the one layer E30's equalization
   made worse, 0.46 % → 1.56 %) and the 74 channels E30 still counts as destroyed. The
   owner's standing rule for quantization problems: **report what you find before
   working around it**; the fix may belong in the export on the training server.
2. **Causal τ=2 int8 (E33).** The streaming target is causal BC-ResNet-2 (97.67 % float).
   Needs an export from the training server; then `sim_onchip_int8.py`, then the board.
3. **Milestone 3, streaming on the device.** Not started. `ONCHIP.md` has the memory
   analysis (τ=2 streams in 5.9 KB, τ=8 in 23.3 KB) and per-frame latency *estimates*.
   ESP-DL runs whole graphs, so streaming needs a per-frame step graph with the layer
   histories as explicit state inputs and outputs; that design has not been done.
4. **The Transpose tax** (151 ms, 15.8 %) comes from SubSpectralNorm's reshape. Fixing it
   is a model-design change on the training side.
5. **τ=1** was never deployed (`ONCHIP.md` asked for it first; τ=2 was deployed by
   instruction).
6. **Live microphone** (INMP441 on I2S; the root `HANDOFF.md`, which the training side
   may move to `docs/HANDOFF_MIC.md`) is deferred and has never run on hardware.

## 7. Board 2 — identify it first

The owner reports a newer ESP32 with a camera. **Identify it before building anything**
(ESP-IDF shell, board plugged in):
```
python -m esptool -p <PORT> chip_id
python -m esptool -p <PORT> flash_id
```
Record the chip and revision, the flash size, and whether esptool's feature list shows
embedded PSRAM; note how its USB is wired (§2.4) and what is printed on the board. Camera
boards of this kind are most often an ESP32-S3 with 8 MB PSRAM (ESP32-S3-EYE, XIAO
ESP32S3 Sense, ESP32-S3-CAM clones), and some have an on-board microphone. An ESP32-P4
board is the other possibility. **Do not assume which — report it to the owner first.**

**What changes with the chip:**

| | ESP32 (board 1) | ESP32-S3 | ESP32-P4 |
|---|---|---|---|
| Conv quantization | per tensor | **per tensor** (E30) — the equalized export is still needed | per channel — the fold spread stops mattering |
| esp-ppq target | `c` | `esp32s3` | `esp32p4` |
| expected speed | 951 ms measured | much faster: SIMD kernels; Espressif's ESP-NN benchmark is ~7× on a similar model — **not measured for this graph** | not estimated |
| PSRAM | none | usually 8 MB → τ=3 and τ=8 can run whole-window, arena in PSRAM | usually 32 MB |

**What the firmware needs** (none of this has been tried):
* **Its own configuration, leaving board 1's alone** (§3.8). One way: a
  `sdkconfig.defaults.esp32s3` without the `CONFIG_IDF_TARGET="esp32"` line, with PSRAM
  enabled and the right flash size, built as
  `idf.py -B build_s3 -DIDF_TARGET=esp32s3 -DSDKCONFIG=sdkconfig.esp32s3 -DSDKCONFIG_DEFAULTS=sdkconfig.defaults.esp32s3 build`.
  Check ESP-IDF's docs for how `SDKCONFIG_DEFAULTS` combines files before relying on it.
* **Re-quantize for that target** (`--target esp32s3`) and simulate with the same target.
  **Re-establish bit-exactness with `model->test()` and a board run — do not assume it
  carries over**; the S3 uses different kernels.
* If USB goes to the chip's native USB-Serial/JTAG, `app_eval.cpp`'s UART0 transport will
  not reach the host. It would need the `usb_serial_jtag` driver, and then the baud limit
  and the reset-on-open behaviour both change.
* The camera is irrelevant to keyword spotting; leave it unused. An on-board microphone
  matters for the live-mic phase, but it is likely on different pins and possibly PDM, so
  the existing I2S recording firmware (`rig/firmware/i2s_stream`) will not apply as is.

**Suggested order:**
1. Identify the board.
2. τ=2 with `app_main.cpp`: 4/4 and the latency.
3. Milestone 2 on the new chip, simulation first: the speed headline, plus the int8
   accuracy for that target.
4. τ=3 and τ=8 whole-window from PSRAM: measured silicon accuracy for scales board 1
   cannot hold.

Keep board 1 for the streaming milestone: E31's argument stands, since its memory limit
is what makes the streaming result meaningful. Board 2 is a second platform, not a
replacement.

## 8. The machine the work was done on (reference only)

E26–E33 ran on a Windows 11 laptop: ESP-IDF in `D:\esp\esp-idf` with tools in
`D:\esp\.espressif`, the Python environment in `D:\esp\ppq-venv` (same versions as
`requirements-host.txt`, plus esptool 5.6.0), the repository at `D:\esp\native-kws`, and
board 1 on `COM5`. Raw build and flash logs stayed there and are not in git; nothing in
them is needed — every result is in the repository.
