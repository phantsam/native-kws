# NativeKWS

Keyword spotting model designed to be competitive with large SOTA models while staying deployable on an ESP32-S3 + INMP441 microphone.

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
| V2-35 | 98.01 (1 of 3 seeds so far) | 97.90 ± 0.05 | 97.89 |

Caveats (details in `docs/EXPERIMENTS.md`, E6, E19, E21): distillation is not used by the published baselines; on V1-12 the result depends on the test-set protocol (97.95 on the testing_list protocol used by Keyword Mamba); the noise-robustness gains are not consistent at −5 dB SNR.

## Documentation
* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): all models, equations, design rationale, novelty assessment
* [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): chronological experiment log (hypothesis, command, result, decision)
* [`docs/RESULTS.md`](docs/RESULTS.md): auto-generated results tables (`python scripts/collect_results.py`)

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install numpy soundfile tqdm tensorboard pyyaml
# data: Google Speech Commands v0.01 / v0.02 + official test sets under data/
.venv/bin/python scripts/prepare_data.py
```

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
