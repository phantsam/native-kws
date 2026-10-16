# A real-microphone test set for the INMP441

Status: **tooling built and verified in simulation; no audio recorded yet** (2026-10-16).

## Why

Every number in `RESULTS.md` is measured on Google Speech Commands: clips recorded
years ago, through unknown consumer microphones, by crowdworkers. The model we
intend to deploy will instead hear an INMP441 MEMS microphone, clocked by an ESP32
I2S peripheral, in one room, at a distance, with that room's noise. Nothing in the
literature measures that gap for this microphone, because the gap is specific to the
hardware and nobody publishes hardware-specific test sets.

A half point on Speech Commands is within the winner's curse we already documented
(E19). The size of the microphone channel's cost is not: it is an unmeasured
quantity that plausibly dwarfs it.

## The design that makes this a measurement rather than a demo

The obvious thing is to record people saying the keywords. That gives a number, but
the number confounds three things at once: new speakers, new room, new microphone.
A drop of 6 points could be any of them.

So the primary set **replays the official Speech Commands test clips through a
speaker into the INMP441**. The utterances, speakers, labels and class balance are
bit-identical to the cached test set. The only thing that changes is the acoustic
path. Scoring the same model on the cached clip and on its recording is therefore a
*paired* measurement of the channel alone, clip by clip, and `rig/eval_real.py`
reports exactly that: `cached`, `recorded`, and which individual clips broke.

A caveat that must stay attached to every number from this set: the path is
**speaker + room + microphone**, not the microphone alone. The loudspeaker's own
response is inside the measurement. The sweep recording (set D) is what lets us
estimate and, if needed, deconvolve it.

## What gets recorded

**Set A — replay (primary, paired).** The 4,074-clip official V2-12 test set,
shuffled with a fixed seed so class never correlates with position in the
recording, played back under each condition below. ~102 min per condition.

| id | distance | noise source | what it isolates |
|---|---|---|---|
| `quiet_1m` | 1 m | quiet room | the headline channel cost |
| `quiet_3m` | 3 m | quiet room | far-field level loss and reverberation |
| `fan_1m` | 1 m | fan / AC | stationary broadband noise |
| `babble_1m` | 1 m | TV or radio speech | non-stationary, speech-like: the hard case |

Optional once the core four are done: `quiet_0.5m`, `offaxis_1m` (mic rotated 90°,
which tests the port and board shadow), `kitchen_1m`.

**Set B — live speakers.** Real people saying the 10 keywords, 10 repetitions each,
at 1 m and 3 m. This is the realism check that set A cannot give: Lombard effect,
real vocal effort, real head movement. It is *not* a benchmark — with a handful of
speakers it has no statistical power for accuracy claims, and it must be reported as
a qualitative check with the speaker count stated.

`rig/record_live.py` prompts one word at a time in a fixed-seed random order (so the
speaker cannot settle into a rhythm, and no single word lines up with drift in their
voice), records a 2 s window, centres a 1 s window on the word by short-time energy,
and rejects takes that were silent, clipped or hit a dropped block so they can be
repeated immediately. It records 5 s of room noise first, which sets the SNR
reference for every clip in the session.

Two things to expect, both real and both worth measuring rather than hiding:

* **Only the ten keywords exist.** The model knows `down go left no off on right stop
  up yes`, plus `_unknown_` and `_silence_`. Anything else is correctly classed as
  unknown; that is not a bug.
* **Speech Commands is predominantly American-accented English.** A speaker with a
  different accent should expect lower accuracy than the 99.02 % in `RESULTS.md`.
  The size of that gap is itself a finding, and it is one of the few things a small
  live set *can* honestly show, because an accent effect is large where a half-point
  benchmark difference is not.

**Privacy.** Set B records identifiable human voices. Get each speaker's agreement
first, label sessions with anonymous ids (`s01`, `s02`) and never a name, and keep
the audio out of the repository — `rig/out/` is in `.gitignore` and should stay
there. Publishing any of it is a separate decision that needs each speaker's consent.

**Set C — noise alone.** 30 s before every condition block, with nothing playing.
Two uses: it sets the true noise floor for the SNR measurement, and it becomes a
new background-noise bank for training augmentation. The GSC `_background_noise_`
set is six files that every paper in the field trains on; room noise recorded
through the actual deployment microphone is not.

**Set D — calibration.** A 20 s logarithmic sweep, 50 Hz to 8 kHz, at 1 m.
Gives the speaker+room+mic transfer function, which supports the deconvolution
above and a channel simulator for augmentation.

## Procedure

Keep the microphone and speaker fixed for a whole block; move them only between
conditions, and write down what changed in `--note`.

```bash
# once: build the stimulus (writes stimulus_NN.wav + manifest.json)
python rig/build_stimulus.py --out rig/stimulus --task v2_12

# per condition, with the firmware flashed and playback ready
python rig/capture.py rig/out/quiet_1m__noise.wav  --seconds 30  --note "quiet room, nothing playing"
python rig/capture.py rig/out/quiet_1m__stimulus_00.wav --seconds 1250 --note "1 m on-axis, quiet"
#   start capture first, then start playback within the 30 s lead-in

# cut into labelled clips (reports clock drift and alignment residual)
python rig/align.py rig/out/quiet_1m__stimulus_00.wav --out cache/real_quiet_1m.pt \
    --note "1 m on-axis, quiet"

# the payoff: paired cached-vs-recorded accuracy for any trained run
python -m rig.eval_real cache/real_quiet_1m.pt runs/v2_12_native-H-nass-kdens \
    runs/v2_12_bcresnet-8
```

Live speakers (set B), on the machine the board is plugged into:

```bash
python rig/record_live.py --speaker s01 --distance 1.0 --reps 10
# ...copy the session folder to the training machine, then
python rig/make_set.py rig/out/live/s01_1m --out cache/real_live.pt
python -m rig.eval_real cache/real_live.pt runs/v2_12_native-H-nass-kdens
```

### Before any of it: does the chain work at all?

```bash
python -m rig.listen runs/v2_12_native-H-nass-kdens --port /dev/ttyUSB0
```

Speak the keywords and watch it fire. This runs the PyTorch model on the host, so it
says nothing about on-device latency, but it exercises everything else — wiring, I2S
slot and bit alignment, gain, DC offset, front-end, model — in about ten seconds, and
it is by far the fastest way to find a miswired SD pin or an ungrounded L/R. The
level meter should sit between about −30 and −6 dBFS while you speak.

`capture.py` writes a sidecar JSON with peak level, DC offset and dropped-block
count, and warns on clipping or a dead microphone. Check it before moving anything:
a take with dropped blocks is still usable (gaps are zero-filled so the timeline
stays exact) but its SNR numbers near the gap are not.

## Validity rules, fixed in advance

These are pre-registered the same way the clean-win rule in E19 was, so that the
recorded set cannot be quietly turned into a tuning set.

1. **No model selection on set A.** It is a test set. Any augmentation, channel
   simulator or retraining tuned against it invalidates every number it produces.
   Tune on simulated channels or on a held-out condition, never on the reported one.
2. **Report every condition recorded**, including takes that look bad, with the
   drop count and measured SNR beside each.
3. **Report the speaker count for set B** in the same sentence as any set B number.
4. **One room is one room.** Results describe this room, this mic unit and this
   loudspeaker. A second room, or a second INMP441 unit, is what would turn this
   into a generalisation claim, and until then the claim is not made.
5. The gain applied in `align.py` matches the recorded set's overall RMS to the
   cached set's, so the comparison is not a trivial loudness difference. The gain is
   stored in the output's `meta` and must be reported.

## Verification status

The alignment and SNR path was tested against a synthetic recording of the stimulus
with +230 ppm clock drift, a 137 ms start delay, spectral colouring, additive noise
and a DC offset:

* clock ratio recovered as **+230 ppm**
* per-clip alignment error **median +1 sample**, 5th–95th percentile **−1…+2 samples**
* waveform correlation with the original clip **median 0.949**
* labels and clip indices preserved for all clips
* SNR estimator tracks known 10 dB steps in input SNR

The host-side serial path was tested against a virtual serial port fed by an
emulated firmware:

* clean stream decoded **bit-exactly** (zero error over 51,200 samples)
* three injected block drops were detected, zero-filled in the correct positions,
  and a block 40 later was **still bit-exact** — a dropout cannot silently shift
  the labels
* injected bus garbage was resynchronised without losing the stream
* `capture.py` wrote a bit-exact WAV with a correct sidecar
* `listen.py`, fed six real keyword clips, detected all six in the right order with
  no duplicates
* the set B trimmer centres the word to **median 10 ms** (p90 60 ms) with >98 % of
  the word's energy retained, for words placed randomly in a 2 s window

Not yet verified against hardware: I2S bit alignment and slot selection, real UART
throughput on a physical USB bridge, and whether the sweep survives a small
loudspeaker well enough to anchor every segment.

## What the data is for

1. **The headline measurement:** how much accuracy the microphone channel costs,
   paired, per clip. This is the number the paper is built on and nobody has it.
2. **A channel simulator** fitted to set D plus set C, so training augmentation can
   match the deployment path instead of guessing.
3. **Whether NASS earns its place in the real world.** The speech-presence gate is
   trained on synthetic SNR targets from mixed GSC noise. Set C noise is the first
   honest test of whether the gate fires correctly on noise it was never shown.
4. **A noise bank** that is not the same six files as everyone else's.

Related: [`EXPERIMENTS.md`](EXPERIMENTS.md) for the run log, [`LITERATURE.md`](LITERATURE.md)
§4 for the robustness prior art this set is meant to go beyond.
