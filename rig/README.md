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

| | runs on | |
|---|---|---|
| `stream.py` | laptop | framed block reader: resync, sequence checking, zero-filled drops |
| `listen.py` | laptop | live keyword spotting -- speak and watch the model fire |
| `capture.py` | laptop | serial stream -> 32-bit WAV + sidecar JSON (level, DC, drops) |
| `record_live.py` | laptop | prompted recording of real speakers (set B) |
| `build_stimulus.py` | server | official test clips -> playback WAV + manifest, with resync sweeps |
| `align.py` | server | recording + manifest -> labelled 1 s clips with per-clip SNR |
| `make_set.py` | server | live sessions -> cache file |
| `eval_real.py` | server | paired cached-vs-recorded accuracy for trained runs |

**The board is plugged into your laptop, not the GPU server**, so the work splits:
record locally, copy the WAVs over, align and evaluate on the server.
`stream.py`, `capture.py` and `record_live.py` need only `pyserial`, `numpy` and
`soundfile`. `listen.py` also needs `torch` and a checkpoint, but runs fine on CPU
(~32 ms per window with `--threads 4`; more threads is *slower*, not faster).

## First run

Start here -- it finds wiring mistakes in about ten seconds:

```bash
pip install pyserial
python -m rig.listen runs/v2_12_native-H-nass-kdens --port /dev/ttyUSB0
```

Say `yes`, `no`, `stop`, `go`. The meter should sit between about -30 and -6 dBFS
while you speak. Nothing moving at all means the wiring or the slot is wrong (check
SD on GPIO 33 and L/R grounded); a meter pinned at 0 dBFS means it is clipping.

Then the replay dry run:

```bash
python rig/build_stimulus.py --out rig/stimulus --task v2_12 --limit 200   # 5 min
python rig/capture.py rig/out/test__stimulus_00.wav --seconds 320 --note "dry run"
python rig/align.py rig/out/test__stimulus_00.wav --out cache/real_test.pt
```

Check, in order: the sidecar JSON's peak level is between -30 and -3 dBFS and
`dropped_blocks` is 0; `align.py` reports a clock drift under ~1000 ppm and a
residual under ~50 ms. Then record the full conditions from
[`../docs/DATASET.md`](../docs/DATASET.md).
