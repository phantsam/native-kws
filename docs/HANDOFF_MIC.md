# Handoff: the INMP441 microphone phase

**Status: deferred, and nothing here has ever run on hardware.** The on-chip deployment
came first and is now measured on silicon (E26–E33, [`ONCHIP.md`](ONCHIP.md)). This file is
the plan for the microphone phase, moved out of the root `HANDOFF.md` unchanged in
substance so it is ready when that phase starts.

Read it together with [`DATASET.md`](DATASET.md), which **pre-registers** the recording
protocol — the condition grid, the four sets, and the validity rules that make the recorded
audio a test set rather than something to tune against.

---

## 1. Why this phase exists

Every accuracy number in this project is measured on a decade-old public dataset recorded
through unknown microphones. The point of this phase is to find out what happens when the
same model hears **this** microphone in **this** room, which nobody has measured, and which
is probably a larger effect than the fractions of a point separating us from the published
state of the art.

The scientifically important part is **set A, the replay set**: the official test clips
played through a speaker into this microphone, giving a *paired*, per-clip measurement of
what the acoustic path alone costs. Set B is live speakers.

## 2. Hardware

A plain **ESP32** (Xtensa LX6) and an **INMP441** I2S MEMS microphone on a breadboard,
powered over USB.

| INMP441 pin | connect to | note |
|---|---|---|
| VDD | **3V3** | 3.3 V part. 5 V may destroy it. |
| GND | GND | |
| L/R | **GND** | selects the left I2S slot, which the firmware reads |
| SCK | GPIO 26 | bit clock |
| WS | GPIO 25 | word select |
| SD | GPIO 33 | data out of the mic, into the ESP32 |

Check this wiring physically before flashing. Three of the five failure modes in §6 are
wiring.

**The plain ESP32 is the right board for this.** It streams audio to the host over USB and
the host runs the model, so the chip's inference speed is irrelevant here — `rig/listen.py`
does 32 ms per window on a laptop CPU, comfortably inside real time. (The root `HANDOFF.md`
used to claim "an ESP32-S3 is required for the deployment phase". E31 corrected that for
deployment, and it was never true for recording: the I2S peripheral is identical.
On-device real-time inference is a separate question, answered in `ONCHIP.md` — 952 ms per
window whole-window on the base ESP32, which is why milestone 3 is streaming execution.)

## 3. Set up the machine with the board

```bash
cd native-kws
python3 -m venv .venv
.venv/bin/pip install pyserial numpy soundfile
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

CPU torch is deliberate: a far smaller download, and inference takes 32 ms per window. No
GPU and no dataset are needed on this machine. The trained checkpoint is committed at
`deploy/v2_12_native-H-nass-kdens/` (`args.json` + `best.pt`) — that is the only model you
need.

## 4. Flash the firmware

Sketch: `rig/firmware/i2s_stream/i2s_stream.ino`. It streams raw 32-bit I2S samples in
256-sample blocks, each tagged with a magic word and a sequence number, so the host detects
a dropped block instead of silently shifting every later sample in time.

**`arduino-cli` is strongly preferred over the IDE GUI** — deterministic, readable errors,
no screenshots:

```bash
arduino-cli config init
arduino-cli config add board_manager.additional_urls \
    https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli core list                 # MUST show 3.x -- see below
arduino-cli board list                # find the port
arduino-cli compile --fqbn esp32:esp32:esp32 rig/firmware/i2s_stream
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 rig/firmware/i2s_stream
```

If you must use the IDE GUI: Board = **ESP32 Dev Module**, correct port, Upload.

### The three things that will actually go wrong

1. **arduino-esp32 core 2.x will not compile this.** The sketch uses the ESP-IDF 5.x I2S
   driver (`driver/i2s_std.h`), which does not exist in core 2.x. The error is
   `driver/i2s_std.h: No such file or directory`. Fix it by installing core **3.x**, not by
   rewriting the sketch for the legacy `driver/i2s.h` API — if you think you need to, stop
   and report back instead.
2. **The Arduino IDE Serial Monitor holds the port exclusively.** Close it before running
   any Python script here, or you get `could not open port` / `multiple access on port`.
   This will bite you at least once. (The same lesson recurred on the deployment side:
   `HANDOFF_DEVICE.md` §3 uses `scripts/serial_capture.py` instead of `idf.py monitor`.)
3. **Many clone boards need the BOOT button held** while the uploader prints
   `Connecting....`. Release it once it starts writing. On the board used for deployment,
   this turned out to be mandatory every time — *hold BOOT, tap EN, release BOOT*.

## 5. Verify the microphone — do this before anything else

```bash
.venv/bin/python -m rig.listen deploy/v2_12_native-H-nass-kdens --port /dev/ttyUSB0
```

Speak `yes`, `no`, `stop`, `go`, `left`, `right`, `up`, `down`, `on`, `off`. Those ten words
plus `_unknown_` and `_silence_` are the entire vocabulary; anything else is correctly
reported as unknown, which is not a bug.

**Success looks like:** a level meter sitting around −45 dBFS in silence, rising to roughly
−30…−6 dBFS while you speak, and `>>> YES` lines when you say a keyword.

This one command exercises wiring, I2S slot and bit alignment, gain, DC offset, the feature
front-end and the model. It is by far the fastest way to find a wiring mistake, and it takes
about ten seconds.

Expect lower accuracy than 99 %. Speech Commands is predominantly American-accented
English, so a different accent will cost real accuracy. **That gap is a finding worth
measuring, not a bug to tune away.**

## 6. If it does not work

| symptom | most likely cause |
|---|---|
| `no data from the serial port` | firmware not flashed, wrong port, or Serial Monitor still open |
| meter flat and very low (< −60 dBFS), never moves | SD not on GPIO 33, or mic not powered — check 3V3 |
| meter reads exactly zero, perfectly flat | reading the wrong I2S slot: confirm **L/R is tied to GND** |
| loud constant noise, no reaction to speech | wrong slot or wrong bit alignment; report the `capture.py` sidecar |
| meter pinned near 0 dBFS | clipping; this mic has no gain control, so move the source back |
| `blocks lost` greater than 0 | the USB-UART bridge cannot sustain 2 Mbaud (see below) |
| fires keywords while the room is silent | gain or DC problem; try `--agc`, and report the sidecar JSON |

**On dropped blocks:** the stream needs 64 kB/s. 2 Mbaud gives ~200 kB/s, and CP2102N and
CH340 bridges handle it, but an older CP2102 caps lower. **This is now a measured fact, not
a hypothesis:** the deployment board's CP2102 (USB serial `0001`) could not carry 2 Mbaud at
all — nothing intelligible came back — and the device-side protocol was moved to 921600
(E32). Expect to need the same here.

Note that *lowering* the baud rate makes dropped blocks worse, not better — the real fix is
sending 16-bit instead of 32-bit samples, which is a change to both the firmware and
`rig/stream.py`. **Do not improvise that format change.** Report the drop count and let it
be implemented and tested properly, because a half-correct framing change silently corrupts
every recording.

## 7. Then collect data

Once §5 works, in this order:

```bash
# 10 s smoke recording -- check the sidecar JSON before anything longer
.venv/bin/python rig/capture.py rig/out/smoke.wav --seconds 10 --note "first hardware take"

# set B: live speakers
.venv/bin/python rig/record_live.py --speaker s01 --distance 1.0 --reps 10
```

Set A needs `stimulus_*.wav` files built on the training server (they need the Speech
Commands cache, which is not in this repo). Ask for them; do not try to regenerate them on
the board machine. Alignment, `make_set.py` and `eval_real.py` also run on the training
server — record here, copy the WAVs over.

## 8. Rules

* **Do not modify** `rig/stream.py`, `rig/align.py`, `rig/eval_real.py`, or anything under
  `kws/`. They are verified against tests recorded in `EXPERIMENTS.md` E24–E25 (bit-exact
  stream decoding, drop handling that preserves sample timing, alignment to a median of
  1 sample). Changing them invalidates those results. Fixing the firmware and adding new
  scripts is fine.
* **Never tune anything against recorded audio.** `DATASET.md` pre-registers this: the
  recorded set is a test set, and a model, augmentation or channel simulator fitted to it
  invalidates every number it produces.
* **No audio in git.** `rig/out/` is gitignored; keep it that way.
* **Set B records identifiable human voices.** Get each person's agreement, use anonymous
  ids (`s01`, `s02`), never names, and do not publish any of it.
* Append what you did to `EXPERIMENTS.md` as a new numbered entry, **including what
  failed** — the numbering is shared with the other machine, so check the last entry first.

## 9. What is already verified, and what is not

**Verified in software** (E24, E25): the host serial path decodes bit-exactly; injected
block drops are zero-filled in the correct positions with later blocks still bit-exact;
injected bus garbage resynchronises; `capture.py` writes a bit-exact WAV; `listen.py` fed
six real keyword clips detected all six in order with no duplicates; the replay aligner
recovers a 230 ppm clock error and lands clips to a median of 1 sample; the live trimmer
centres a word to a median of 10 ms.

**Never run on hardware: all of it.** Specifically unverified are the I2S bit alignment and
slot selection, the real throughput of this USB bridge, and whether the sketch compiles
against the installed core. Assume the first compile fails.

## 10. Report back

Paste these so the problem can be diagnosed from the other side:

1. `arduino-cli core list` and `arduino-cli board list`
2. the full compile/upload output, success or failure
3. about 10 s of `rig.listen` output, including the meter line and any detections
4. the sidecar JSON from the smoke recording (`rig/out/smoke.json`)
