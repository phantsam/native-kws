# Handoff: bring up the INMP441 microphone on the ESP32

You are picking this up on the machine the **ESP32 is physically plugged into**. The
model, the training pipeline and the recording tools already exist and are tested in
software. Nothing has ever run on the hardware. Your job is the first hardware
bring-up, and then data collection.

Read this file, then `rig/README.md`, then `docs/DATASET.md` §Procedure. Do not start
by reading the training code; you do not need it.

---

## 1. What this project is, in four sentences

A keyword-spotting neural network (301 K parameters) that reaches 99.02 % on Google
Speech Commands V2-12, beating the best published from-scratch result with 17x fewer
parameters. Every one of those numbers is measured on a decade-old public dataset
recorded through unknown microphones. The point of this phase is to find out what
happens when the same model hears **this** microphone in **this** room, which nobody
has measured, and which is probably a larger effect than the fractions of a point
separating us from the published state of the art.

Full results and caveats: `README.md`. Experiment log: `docs/EXPERIMENTS.md`
(entries E24 and E25 cover the recording rig).

## 2. Hardware

A plain **ESP32** (Xtensa LX6, *not* an ESP32-S3) and an **INMP441** I2S MEMS
microphone on a breadboard, powered over USB.

| INMP441 pin | connect to | note |
|---|---|---|
| VDD | **3V3** | 3.3 V part. 5 V may destroy it. |
| GND | GND | |
| L/R | **GND** | selects the left I2S slot, which the firmware reads |
| SCK | GPIO 26 | bit clock |
| WS | GPIO 25 | word select |
| SD | GPIO 33 | data out of the mic, into the ESP32 |

Check this wiring physically before flashing. Three of the five failure modes in
§6 are wiring.

**Known for later, not now:** this model will *not* run in real time on a plain
ESP32. Espressif's own ESP-NN benchmark runs the same int8 model in 380 ms on ESP32
versus 54 ms on ESP32-S3 at the same 240 MHz, because the S3 has vector instructions
and hand-written assembly kernels while the ESP32 gets generic C. Our budget needs
63 MMAC/s, already 20–60 % of one S3 core. **An ESP32-S3 is required for the
deployment phase.** For recording, the plain ESP32 is completely adequate — the I2S
peripheral is the same. Do not let anyone "fix" this by shrinking the model.

## 3. Set up the machine

```bash
git clone <this repo>            # you are probably already in it
cd native-kws
python3 -m venv .venv
.venv/bin/pip install pyserial numpy soundfile
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

CPU torch is deliberate: it is a far smaller download and inference takes 32 ms per
window, which is well inside real time. No GPU is needed on this machine.

A trained checkpoint is committed at `deploy/v2_12_native-H-nass-kdens/`
(`args.json` + `best.pt`). That is the only model you need.

## 4. Flash the firmware

Sketch: `rig/firmware/i2s_stream/i2s_stream.ino`. It streams raw 32-bit I2S samples
in 256-sample blocks, each tagged with a magic word and a sequence number so the
host detects a dropped block instead of silently shifting every later sample in time.

**`arduino-cli` is strongly preferred over the IDE GUI** — it is deterministic,
its errors are readable, and it does not need screenshots:

```bash
arduino-cli config init
arduino-cli config add board_manager.additional_urls \
    https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli core list                 # MUST show 3.x -- see the warning below
arduino-cli board list                # find the port
arduino-cli compile --fqbn esp32:esp32:esp32 rig/firmware/i2s_stream
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 rig/firmware/i2s_stream
```

If you must use the IDE GUI: Board = **ESP32 Dev Module**, correct port, Upload.

### The three things that will actually go wrong

1. **arduino-esp32 core 2.x will not compile this.** The sketch uses the ESP-IDF 5.x
   I2S driver (`driver/i2s_std.h`), which does not exist in core 2.x. The error is
   `driver/i2s_std.h: No such file or directory`. Fix by installing core **3.x**
   (3.3.x at the time of writing), not by rewriting the sketch for the legacy
   `driver/i2s.h` API — if you think you need to, stop and report back instead.
2. **The Arduino IDE Serial Monitor holds the port exclusively.** Close it before
   running any Python script here, or you get `could not open port` /
   `multiple access on port`. This will bite you at least once.
3. **Many clone boards need the BOOT button held** while the uploader prints
   `Connecting....`. Release it once it starts writing.

## 5. Verify the microphone — do this before anything else

```bash
.venv/bin/python -m rig.listen deploy/v2_12_native-H-nass-kdens --port /dev/ttyUSB0
```

Speak `yes`, `no`, `stop`, `go`, `left`, `right`, `up`, `down`, `on`, `off`. Those
ten words plus `_unknown_` and `_silence_` are the entire vocabulary; anything else
is correctly reported as unknown, which is not a bug.

**Success looks like:** a level meter that sits around −45 dBFS in silence and rises
to roughly −30…−6 dBFS while you speak, and `>>> YES` lines when you say a keyword.

This one command exercises wiring, I2S slot and bit alignment, gain, DC offset, the
feature front-end and the model. It is by far the fastest way to find a wiring
mistake, and it takes about ten seconds.

Expect lower accuracy than 99 %. Speech Commands is predominantly American-accented
English, so a different accent will cost real accuracy. That gap is a finding worth
measuring, not a bug to tune away.

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

**On dropped blocks:** the stream needs 64 kB/s. 2 Mbaud gives ~200 kB/s, and CP2102N
and CH340 bridges handle it, but an older CP2102 caps at 1 Mbaud (~100 kB/s). Note
that *lowering* the baud rate makes this worse, not better — the fix is to send 16-bit
instead of 32-bit samples, which is a change to both the firmware and `rig/stream.py`.
**Do not improvise that format change.** Report the drop count and let it be
implemented and tested properly, because a half-correct framing change silently
corrupts every recording.

## 7. Then collect data

Once §5 works, in this order:

```bash
# 10 s smoke recording -- check the sidecar JSON before anything longer
.venv/bin/python rig/capture.py rig/out/smoke.wav --seconds 10 --note "first hardware take"

# set B: live speakers
.venv/bin/python rig/record_live.py --speaker s01 --distance 1.0 --reps 10
```

Set A, the replay set, is the scientifically important one — it plays the official
test clips through a speaker into this microphone, giving a paired, per-clip
measurement of what the acoustic path costs. It needs `stimulus_*.wav` files that are
built on the training machine (they need the Speech Commands cache, which is not in
this repo). Ask for them; do not try to regenerate them here.

Alignment, `make_set.py` and `eval_real.py` run on the training machine. Record here,
copy the WAVs over.

## 8. Rules

* **Do not modify** `rig/stream.py`, `rig/align.py`, `rig/eval_real.py`, or anything
  under `kws/`. They are verified against tests recorded in `docs/EXPERIMENTS.md`
  E24–E25 (bit-exact stream decoding, drop handling that preserves sample timing,
  alignment to a median of 1 sample). Changing them invalidates those results. Fixing
  the firmware and adding new scripts is fine.
* **Never tune anything against recorded audio.** `docs/DATASET.md` pre-registers
  this: the recorded set is a test set, and a model, augmentation or channel
  simulator fitted to it invalidates every number it produces.
* **No audio in git.** `rig/out/` is gitignored; keep it that way.
* **Set B records identifiable human voices.** Get each person's agreement, use
  anonymous ids (`s01`, `s02`), never names, and do not publish any of it.
* Append what you did to `docs/EXPERIMENTS.md` as a new numbered entry, including
  what failed. The log's value is that it records negative results too.

## 9. What is already verified, and what is not

**Verified in software** (E24, E25): the host serial path decodes bit-exactly;
injected block drops are zero-filled in the correct positions with later blocks still
bit-exact; injected bus garbage resynchronises; `capture.py` writes a bit-exact WAV;
`listen.py` fed six real keyword clips detected all six in order with no duplicates;
the replay aligner recovers a 230 ppm clock error and lands clips to a median of
1 sample; the live trimmer centres a word to a median of 10 ms.

**Never run on hardware:** all of it. Specifically unverified are the I2S bit
alignment and slot selection, the real throughput of this USB bridge, and whether
the sketch compiles against the installed core. Assume the first compile fails.

## 10. Report back

Paste these so the problem can be diagnosed from the other side:

1. `arduino-cli core list` and `arduino-cli board list`
2. the full compile/upload output, success or failure
3. about 10 s of `rig.listen` output, including the meter line and any detections
4. the sidecar JSON from the smoke recording (`rig/out/smoke.json`)
