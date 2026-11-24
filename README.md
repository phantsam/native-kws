# NativeKWS

Keyword spotting model designed to be competitive with large SOTA models while staying deployable on a plain ESP32 + INMP441 microphone.

**Final model: NativeKWS-H + NASS (301 K parameters)**
* BC-ResNet-depth convolutional front-end → two bidirectional Mamba (selective state-space) blocks.
* **Noise-Aware Selective Scan (NASS):** a tiny speech-presence gate, trained with free frame-level SNR labels, scales the SSM step size Δ so that noise frames are not written into memory.
* Trained with knowledge distillation from a 3-model teacher (BC-ResNet-8 + NativeKWS-L + NativeKWS-M) trained on the same data. Inference uses the 301 K student only.

Also explored (see `docs/`): BC-SSM (selective SSM inside the BC-ResNet broadcast path); no clean-accuracy gain.

## Results (Google Speech Commands, test accuracy %, mean ± std over 3 seeds, final epoch)

| Benchmark | NativeKWS-H + NASS + KD (301 K) | BC-ResNet-8, same pipeline (321 K) | Keyword Mamba KWM-T-192, paper (5.2 M) |
|---|---|---|---|
| V2-12 | **99.02 ± 0.05** | 98.66 ± 0.14 | 98.91 |
| V1-12 | 98.13 ± 0.04 | 98.03 ± 0.04 | 98.05 |
| V2-35 | **98.07 ± 0.09** | 97.90 ± 0.05 | 97.89 |

V2-12 and V2-35 beat the best published from-scratch result (Keyword Mamba) by more than one standard deviation, with 17x fewer parameters. Caveats (details in `docs/EXPERIMENTS.md`, E6, E19, E21): distillation is not used by the published baselines; on V1-12 the result depends on the test-set protocol (97.95 on the testing_list protocol used by Keyword Mamba); the noise-robustness gains are not consistent at −5 dB SNR.

## On the chip, measured

**BC-ResNet-2 (27,284 parameters) runs int8 on a plain ESP32 with no PSRAM and scores
97.06 % on the full 4,890-clip V2-12 test set**, against 98.02 % in float on the same
clips — 0.96 points of int8 cost, 951.6 ms per 1-second window, a 98.64 KB activation
arena, weights in flash (`docs/EXPERIMENTS.md` E32). The chip is bit-exact with esp-ppq's
host simulation on all 4,890 clips.

The 301 K final model does **not** fit this chip whole-window at any quantization setting;
streaming execution reduces the peak from 252.5 KB to 23.3 KB, which is the open
engineering contribution (`docs/ONCHIP.md`). The causal models streaming requires cost
0.08–0.37 points (E33).

[`HANDOFF.md`](HANDOFF.md) indexes the two self-contained briefs: the GPU server
([`docs/HANDOFF_TRAINING.md`](docs/HANDOFF_TRAINING.md)) and the computer the board plugs
into ([`docs/HANDOFF_DEVICE.md`](docs/HANDOFF_DEVICE.md)). A trained checkpoint is
committed at `deploy/v2_12_native-H-nass-kdens/`, so the device machine needs no GPU and
no dataset.

## Documentation
* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): all models, equations, design rationale, novelty assessment
* [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): chronological experiment log (hypothesis, command, result, decision)
* [`docs/RESULTS.md`](docs/RESULTS.md): auto-generated results tables (`python scripts/collect_results.py`)
* [`docs/LITERATURE.md`](docs/LITERATURE.md): literature notes behind the design decisions, with provenance tags and an explicit scope caveat (not a systematic review)
* [`docs/DATASET.md`](docs/DATASET.md): protocol for the real-microphone (INMP441) test set — replay design, condition grid, pre-registered validity rules
* [`HANDOFF.md`](HANDOFF.md): index of the two machine-specific handoffs, plus the current headline numbers
* [`docs/HANDOFF_TRAINING.md`](docs/HANDOFF_TRAINING.md): the GPU server — environment, dataset caches, job queue, which checkpoint backs which number, the export pipeline
* [`docs/HANDOFF_DEVICE.md`](docs/HANDOFF_DEVICE.md): the computer the ESP32 plugs into — ESP-IDF, quantization, flashing, the serial evaluation driver
* [`docs/HANDOFF_MIC.md`](docs/HANDOFF_MIC.md): the deferred INMP441 microphone phase — wiring, flashing, verification
* [`docs/ONCHIP.md`](docs/ONCHIP.md): the three on-chip milestones — run the network (done), measure test-set accuracy (done), streaming execution (open)

## Real-microphone test set

`rig/` records the official test clips back through an INMP441 on an ESP32, so the
accuracy cost of the deployment microphone can be measured *paired*, clip by clip,
against the cached originals. It also records live speakers (`record_live.py`) and
runs the model on the live microphone stream (`listen.py`).

```bash
python -m rig.listen runs/v2_12_native-H-nass-kdens --port /dev/ttyUSB0   # speak at it
```

Verified in simulation and against a virtual serial port (alignment median +1 sample
under 230 ppm drift; bit-exact decoding; all six streamed keywords detected in
order); **no audio recorded from hardware yet**. Deferred for now. See [`docs/HANDOFF_MIC.md`](docs/HANDOFF_MIC.md),
[`docs/DATASET.md`](docs/DATASET.md) and [`rig/README.md`](rig/README.md).

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu128
# data: Google Speech Commands v0.01 / v0.02 + official test sets under data/
.venv/bin/python scripts/prepare_data.py
```
`requirements-train.txt` pins the versions every number here was produced with. Full
bootstrap for a fresh server: [`docs/HANDOFF_TRAINING.md`](docs/HANDOFF_TRAINING.md) §10.

## Train
```bash
# final model: NativeKWS-H + NASS, distilled from BC-ResNet-8 + NativeKWS-L + NativeKWS-M (train those first)
python -m kws.train --task v2_12 --size H --noise_aware --teacher runs/v2_12_bcresnet-8 runs/v2_12_native-L runs/v2_12_native-M

# BC-SSM (explored): SSM inside the broadcast path, + NASS, BC-ResNet recipe
python -m kws.train --task v2_12 --model bcssm-7 --noise_aware --opt sgd --lr 0.1 --wd 1e-3 --bs 100 --smoothing 0 --compile
#   ablations: drop --noise_aware | add --causal (streaming) | add --gate_weight 0 (unsupervised gate)

# BC-ResNet-8 reference with its official recipe (pipeline validation)
python -m kws.train --task v2_12 --model bcresnet-8 --opt sgd --lr 0.1 --wd 1e-3 --bs 100 --smoothing 0

# NativeKWS sizes S / M / L / H without distillation (--causal drops the backward Mamba branch only)
python -m kws.train --task v2_12 --size L
```
Tasks: `v1_12`, `v2_12` (official 12-class test set, as BC-ResNet) and `v2_35` (testing_list, as KWT / Keyword Mamba).

## Evaluate noise robustness
```bash
python -m kws.eval_noise runs/v2_12_native-M runs/v2_12_bcresnet-8
```

## Long runs
`scripts/jobqueue.py jobs/main.txt --slots 0:6 1:6` runs a priority queue of training jobs (waits for teachers, checks free GPU memory, resumes interrupted runs). `scripts/finalize_when_done.sh` evaluates noise robustness and both test protocols as runs finish. Model checkpoints (`*.pt`) are not in the repository.
