# Recording rig: INMP441 -> ESP32 -> host

Builds a real-microphone test set for the model in this repository. The protocol,
the conditions to record and the validity rules are in
[`../docs/DATASET.md`](../docs/DATASET.md) -- read that first; this file is wiring
and commands.

## Wiring

| INMP441 | ESP32 | ESP32-S3 |
|---|---|---|
| VDD | 3V3 | 3V3 |
| GND | GND | GND |
| L/R | GND (selects the left slot) | GND |
| SCK | GPIO 26 | GPIO 4 |
| WS | GPIO 25 | GPIO 5 |
| SD | GPIO 33 | GPIO 6 |

The INMP441 is a 3.3 V part: do not feed it 5 V. The sketch picks the pin set
automatically from the build target.

## Firmware

`firmware/i2s_stream/i2s_stream.ino`, Arduino-ESP32 core 3.x (ESP-IDF 5.x),
board "ESP32 Dev Module". It streams 256-sample blocks of raw int32 I2S data with a
magic word and sequence number, so a dropped block is detected instead of silently
shifting every later clip in time.

64 kB/s goes over the plain ESP32's USB-UART bridge at 2 Mbaud (~200 kB/s), leaving
about 3x headroom. On an ESP32-S3 the same sketch uses native USB and the rate is
ignored. The port is typically `/dev/ttyUSB0` on ESP32 and `/dev/ttyACM0` on S3.

## Scripts

| | |
|---|---|
| `capture.py` | serial stream -> 32-bit WAV + sidecar JSON (level, DC, dropped blocks) |
| `build_stimulus.py` | official test clips -> playback WAV + manifest, with resync sweeps |
| `align.py` | recording + manifest -> labelled 1 s clips with measured per-clip SNR |
| `eval_real.py` | paired cached-vs-recorded accuracy for trained runs |

`capture.py` needs `pyserial` (`.venv/bin/pip install pyserial`); the others use the
packages already in the environment.

## First run

```bash
.venv/bin/pip install pyserial
python rig/build_stimulus.py --out rig/stimulus --task v2_12 --limit 200   # 5 min dry run
python rig/capture.py rig/out/test__stimulus_00.wav --seconds 320 --note "dry run"
python rig/align.py rig/out/test__stimulus_00.wav --out cache/real_test.pt
```

Check, in order: the sidecar JSON's peak level is between -30 and -3 dBFS and
`dropped_blocks` is 0; `align.py` reports a clock drift under ~1000 ppm and a
residual under ~50 ms. Then record the full conditions from `docs/DATASET.md`.
