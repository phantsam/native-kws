# Experiment log

Chronological lab notebook: what was run, why, the exact command, the outcome, and the decision taken. The live numbers table is `docs/RESULTS.md` (regenerate with `python scripts/collect_results.py`). Architecture details: `docs/ARCHITECTURE.md`.

---

## Environment and protocol

| Item | Value |
|---|---|
| Hardware | 2× NVIDIA H100 NVL, **shared with other users' jobs** (epoch times vary 2–3× with contention) |
| Software | Python 3.12, PyTorch 2.11 + CUDA 12.8, torchaudio 2.11 |
| Data | Google Speech Commands v0.01 / v0.02 + official 12-class test sets, decoded to int16 caches (`scripts/prepare_data.py`) |
| v2 split sizes | 105,829 clips / 35 words; train 84,843 · valid 9,981 · test 11,005 (matches the published lists); official 12-class test set 4,890 |
| v1 split sizes | 64,721 clips / 30 words; train 51,088 · valid 6,798 · test 6,835; official 12-class test set 3,081 |

### Task protocols (`kws/data.py`)
* **v2_12 / v1_12** (BC-ResNet protocol): 10 keywords + `_unknown_` + `_silence_`. Train: all keyword clips of the train split + 3,077 (v2) / 1,854 (v1) `_unknown_` clips **re-sampled every epoch** from the unknown training pool + the same number of all-zero `_silence_` clips. Valid: keyword clips + 371 / 258 fixed random unknown + as many noise-only silence clips. **Test: the official 12-class test set.**
* **v2_35**: all 35 words, standard `validation_list` / `testing_list`.

### Augmentation (all runs, BC-ResNet recipe; `Augment`, `spec_augment`)
Time shift U(−100, 100) ms with zero padding; GSC background noise added with prob. 0.8 at amplitude U(0, 0.1) (silence: always, U(0, 1)); SpecAugment 2 × frequency masks (width ≤ 7) + 2 × time masks (width ≤ 20).

### Reporting rules
* Every run evaluates valid and test every 5 epochs, for both raw and EMA (0.999) weights.
* **Reported number = test accuracy of the checkpoint with the best validation accuracy** ("best" column). The final-epoch numbers are also logged. The test set is never used for selection.
* Single seed (0) so far. The test set has 4,890 clips (1 clip = 0.02 %), so differences below ~0.15 % should be confirmed with more seeds before being claimed.

### Recipes
| Name | Used by | Settings |
|---|---|---|
| **SGD (BC-ResNet official)** | BC-ResNet-8, all BC-SSM runs | SGD momentum 0.9, lr 0.1, wd 1e-3 (not on 1-D params), bs 100, 200 epochs, 5 warm-up epochs + cosine, no label smoothing |
| **AdamW** | NativeKWS runs | AdamW β=(0.9, 0.98), lr 3e-3, wd 0.05, bs 256, 200 epochs, 5 warm-up + cosine, label smoothing 0.1, drop-path 0.1 |

Deviations from the official BC-ResNet code, applied to all runs: gradient-norm clipping at 5, EMA weights also evaluated, weight decay not applied to 1-D parameters, per-epoch unknown re-sampling.

---

## 2026-09-14

### E0: Pipeline set-up and validation
* Built caches; split counts match the published numbers exactly (see above).
* Ported BC-ResNet; parameter counts match the paper exactly for τ = 1, 2, 3, 8.
* Smoke test of NativeKWS-M (2 epochs): 31 % → 76 % test, pipeline OK.

### E1: BC-ResNet-8 reproduction (`v2_12_bcresnet-8`)
* **Why:** validate that our data protocol and training loop reproduce the published SOTA before trusting any comparison.
* **Command:** `python -m kws.train --task v2_12 --model bcresnet-8 --opt sgd --lr 0.1 --wd 1e-3 --bs 100 --smoothing 0 --name v2_12_bcresnet-8`
* **Result:** best-by-valid **98.67 %** (epoch 160, EMA); final epoch 98.79 % raw / **98.83 %** EMA. Paper: 98.7 %.
* **Decision:** pipeline validated; BC-ResNet-8 is the reference baseline and the KD teacher.

### E2: NativeKWS first design (`v2_12_native-M`, `-M-causal`, `-L`)
* **Hypothesis:** thin BC front-end + bidirectional Mamba back-end beats BC-ResNet-8 at similar size.
* **Commands:** `python -m kws.train --task v2_12 --size M` (`--causal`; `--size L`)
* **Results:** M (278 K) best **98.47 %**; M-causal (239 K) best **98.61 %** (final 98.45 %); L (615 K) best **98.57 %** (final 98.59 %).
* **Findings:**
  1. Below BC-ResNet-8 (98.67 / 98.83 %) despite similar or larger size; doubling the size (L) does not help → **hypothesis rejected**.
  2. Causal ≈ bidirectional → streaming costs essentially nothing, which is encouraging for deployment.
  3. Diagnosis: most parameters in the Mamba blocks; the conv front-end (6 blocks, 32–64 ch) is much weaker than BC-ResNet-8's (12 blocks, 64–160 ch).
* **Decision:** rebalance (E3) and test distillation (E3b); treat stacked conv→Mamba as a baseline, not the contribution.

### E3: Rebalanced hybrid NativeKWS-H (`v2_12_native-H`) and distillation (`v2_12_native-H-kd`)
* **Hypothesis:** a BC-ResNet-depth backbone + light 2-layer Mamba (296 K) closes the gap; KD from BC-ResNet-8 adds more.
* **Commands:**
  `python -m kws.train --task v2_12 --size H --name v2_12_native-H`
  `python -m kws.train --task v2_12 --size H --teacher runs/v2_12_bcresnet-8 --name v2_12_native-H-kd` (α=0.7, T=2)
* **Results (finished 2026-09-15 ~01:30 / 01:50):**

| Run | Params | Best valid | Test @ best valid | Test final raw / EMA | Noise 0 dB / −5 dB |
|---|---|---|---|---|---|
| NativeKWS-H | 296 K | 98.29 | 98.59 | 98.77 / 98.73 | 81.27 / 49.75 |
| NativeKWS-H + KD (BC-ResNet-8) | 296 K | **98.45** | **98.85** | **98.92** / 98.88 | 77.46 / 37.57 |
| *BC-ResNet-8 (E1, reference)* | 321 K | 98.36 | 98.67 | 98.79 / 98.83 | 74.50 / 42.47 |

* **Findings:** (1) rebalancing works: H without KD matches BC-ResNet-8 on the paper rule (98.77 vs 98.79) and is much more noise-robust (+6.8 points at 0 dB, +7.3 at −5 dB); (2) KD adds +0.16 valid and +0.26 test @ best-valid, and the student beats its own teacher; (3) KD reduces robustness at low SNR (the teacher's clean-speech bias is inherited). Single seed, so not yet a claim; seeds and ensemble KD (S1) follow.

### E3c: NativeKWS-M + KD (`v2_12_native-M-kd`): STOPPED
* Started with E3 to isolate the effect of KD on the first design.
* **Stopped at epoch 33** when another user's job filled GPU 1 memory; the M design is superseded, so this was the least informative run to keep. Marker: `runs/v2_12_native-M-kd/STOPPED`.

### E4: Ideas 1 + 2, controlled ablation (BC-SSM / NASS)
* **Design:** see ARCHITECTURE.md §4–5. All four runs use the **exact BC-ResNet-8 SGD recipe**, so differences come from the architecture alone. τ = 7 keeps parameters close to BC-ResNet-8 (342–349 K vs 321 K).
* **Pre-flight checks:** `BCSSMNet(ssm_stages=())` has exactly BC-ResNet-8's 321,068 params; causal mode has zero future leakage; speech-presence targets are 0 % on silence clips and ≈ 39 % on keyword clips; 2-epoch smoke test stable under SGD lr 0.1 (valid 82.7 %, gate frame accuracy 88 % after 2 epochs).
* **Speed:** eager 0.277 s/step vs BC-ResNet-8 0.045 s/step. `torch.compile` brings it to 0.135 s/step (≈ 2×), so all E4 runs use `--compile`.
* **Runs** (common flags `--task v2_12 --model bcssm-7 --opt sgd --lr 0.1 --wd 1e-3 --bs 100 --smoothing 0 --compile`):

| Run | Extra flags | Question it answers |
|---|---|---|
| `v2_12_bcssm-7` | – | Idea 1 alone: does an SSM in the broadcast path beat the temporal conv? (vs E1) |
| `v2_12_bcssm-7-nass` | `--noise_aware` | Ideas 1+2: does supervised noise-aware Δ help, on clean and noisy test sets? |
| `v2_12_bcssm-7-nass-causal` | `--noise_aware --causal` | Streaming version: cost of causality |
| `v2_12_bcssm-7-nass-unsup` | `--noise_aware --gate_weight 0` | Is the free SNR supervision what helps, or just the extra gate? |

* **Status:** launched 23:20 on GPU 0. Expected ~3–5 h each under contention.

### E5: Noise-robustness baseline (`kws.eval_noise`, best-by-valid checkpoints)
* **Protocol:** each official test clip (except `_silence_`) gets a random 1 s GSC background-noise segment mixed in at a fixed SNR computed over the whole clip; seed 0, identical noise across models.
* **Results (test accuracy %):**

| Model | clean | 20 dB | 10 dB | 5 dB | 0 dB | −5 dB |
|---|---|---|---|---|---|---|
| BC-ResNet-8 | 98.67 | 97.77 | 95.32 | 89.75 | 74.50 | 42.47 |
| NativeKWS-M | 98.47 | 97.71 | 95.05 | 89.96 | 69.92 | 32.02 |
| NativeKWS-M causal | 98.61 | 97.65 | 95.83 | 91.98 | 74.93 | 38.26 |
| NativeKWS-L | 98.55 | 97.91 | 94.72 | 89.41 | 73.13 | 35.48 |

* **Findings:** all models collapse at low SNR (−24 to −29 points at 0 dB, −56 to −66 points at −5 dB), so there is large headroom for NASS. Training only sees noise at amplitude ≤ 0.1 (mostly > 10 dB SNR), so 0 / −5 dB is out-of-distribution for every model. BC-ResNet-8 degrades most gracefully at −5 dB among these.
* **Automation:** `scripts/finalize_when_done.sh 1 <runs…>` (running, log `runs/logs/finalize.log`) evaluates noise robustness on GPU 1 as each E3/E4 run finishes and regenerates `docs/RESULTS.md`.

### E6: Benchmark-protocol audit (are our numbers comparable to the SOTA papers?)
* **Why:** papers report "GSC v2-12 accuracy" but build the 12-class test set differently. Checked the official code of each paper (2026-09-15).

| Paper | Test set construction (V2-12) | Model evaluated | Notes |
|---|---|---|---|
| **BC-ResNet** (official repo) | Separately released `speech_commands_test_set_v0.02` (4,890 clips) | Final epoch | Same as our `data.test` |
| **KWT** (`kws_streaming`, `split_data=1`) | `testing_list` keywords + 10 % silence + 10 % unknown (seeded random) | Final | 40 MFCC from 80 mel, speed resampling 0.15, 23,438 steps × bs 512 |
| **Keyword Mamba** (`TorchKWS`, NeMo-style script) | `testing_list` keywords + 10 % unknown + 10 % silence (background slices × U(0,1)) | Final epoch (`best_model` is a reference to the live model, not a copy) | Test loader uses `drop_last=True`, dropping up to 127 clips |

* **Check 1: audio overlap.** All 4,074 keyword clips of the official test set are byte-identical to the `testing_list` keyword clips, and all 408 official unknown clips come from the `testing_list` split. **The protocols share the exact same keyword test audio and class balance (4,074 + 408 + 408)**; they differ only in which 408 unknown clips are drawn and how silence is synthesised.
* **Check 2: measured effect.** New `kws/eval_protocols.py` evaluates each checkpoint on both. Difference is ≤ 0.09 points:

| Run | best.pt official / list | last.pt official / list |
|---|---|---|
| BC-ResNet-8 | 98.67 / 98.65 | 98.79 / 98.88 |
| NativeKWS-M | 98.47 / 98.47 | 98.43 / 98.43 |
| NativeKWS-M causal | 98.61 / 98.65 | 98.47 / 98.51 |
| NativeKWS-L | 98.55 / 98.55 | 98.59 / 98.61 |

* **Conclusion:** V2-12 numbers across BC-ResNet, KWT and Keyword Mamba are comparable to within ~0.1 points. The metric (top-1 accuracy over all test clips) is identical. V2-35 uses the plain `testing_list` (11,005 clips) in all papers. **Remaining differences:** (1) model selection: the papers report the final epoch, we additionally report best-by-validation (more conservative); (2) papers mostly report single runs; (3) training recipes (features, augmentation, epochs) differ, which is legitimately part of each method.
* **Reporting decision:** main tables give *final-epoch raw weights* (same rule as the papers) **and** best-by-validation; SOTA numbers are quoted from the papers with this caveat.

## 2026-09-15

### Goal set by the user: seeds, V1-12 and V2-35, and beat SOTA cleanly
**Rules adopted so the result is clean:**
1. All tweaks and model choices are made on the **validation** split only.
2. The test sets are evaluated for the record but **never used to pick a configuration**.
3. The final frozen configuration is run with **3 seeds on each benchmark**; we report mean ± std, under both the paper rule (final epoch) and best-by-validation.
4. A claim of "beats SOTA" requires the **3-seed mean** to exceed the best published number of the same category (trained from scratch on GSC, no external pre-training) by more than one standard deviation.

**Targets** (best published numbers, from-scratch models):

| Benchmark | Best published | Params | Next best |
|---|---|---|---|
| V1-12 | Keyword Mamba KWM-T-192: 98.05 | 5.2 M | BC-ResNet-8: 98.0 (321 K) |
| V2-12 | Keyword Mamba KWM-T-192: 98.91 | 5.2 M | BC-ResNet-8: 98.7 (321 K) |
| V2-35 | Keyword Mamba KWM-T-192: 97.89 | 5.2 M | KWT-3: 97.69 (5.4 M) |

(Out of scope for the claim: models with large-scale external pre-training, e.g. AST with ImageNet + AudioSet pre-training, 87 M params, reports 98.11 on V2-35.)

### E7: Bug fixes before multi-task runs
* **35-class silence bug:** `Augment` treated label 0 as `_silence_` (always noise at U(0,1) amplitude). In V2-35, label 0 is the word *backward*. Fixed with `has_silence=False` for 35-class tasks. **No completed run is affected** (all were 12-class).
* **Seed coupling:** `--seed` also changed the data seed, i.e. the random unknown/silence composition of the validation set. The data seed is now fixed at 0, so every run uses the identical validation set. Seed-0 runs are unaffected.
* Run names get a `-s<seed>` suffix for seeds ≠ 0.

### E8: Baseline wave on the new benchmarks and seeds (GPU 1, launched 00:10)
BC-ResNet-8, official recipe: `v1_12_bcresnet-8`, `v2_35_bcresnet-8`, `v2_12_bcresnet-8-s1`, `v2_12_bcresnet-8-s2`. They serve as (a) in-house SOTA baselines on every benchmark, (b) seed variance for V2-12, and (c) distillation teachers per task.

### E9: Faster SSM scan attempt: rejected
* Tried a loop-free exact scan: split `bx` into ±parts and evaluate `h_t = Σ_s exp(L_t − L_s) bx_s` with `logcumsumexp`. Numerically exact (forward rel. error 2e-5, gradients match).
* Speed with `torch.compile` (GPU heavily contended): chunked scan **0.304 s/step**, logcumsumexp scan 0.444 s/step, because logcumsumexp's backward is expensive. **Kept the chunked scan.**

### E10: Ensemble teacher for distillation (validation only)
Softmax-averaged `last_ema` checkpoints on the **validation** split (test not looked at):

| Teacher | Valid acc |
|---|---|
| BC-ResNet-8 | 98.29 |
| NativeKWS-L / M / M-causal | 98.09 / 98.18 / 97.91 |
| BC-ResNet-8 + NativeKWS-M | 98.47 |
| **BC-ResNet-8 + NativeKWS-L + NativeKWS-M** | **98.52** |
| all four | 98.54 |

Diverse architectures ensemble well (+0.23 over the best single model). The 3-model ensemble is chosen as the teacher: nearly as good as four, and one fewer forward pass per step.

### E11: Screening runs and job queue
* `scripts/jobqueue.py jobs/main.txt --slots 0:5 1:6`: priority queue, launches when GPU slots free, re-reads the jobs file each poll, log `runs/logs/queue.log`.
* Queued (priority order):
  * **S1** `v2_12_native-H-kdens`: NativeKWS-H + 3-model ensemble KD (compare with the single-teacher `v2_12_native-H-kd`)
  * **S2** `v2_12_bcssm-7-nass-kdens`: BC-SSM-7 + NASS + ensemble KD (SGD recipe)
  * BC-ResNet-8 seeds 1, 2 on V1-12 and V2-35
* **Selection rule for the final model:** highest best-validation accuracy among {BC-SSM ablations, S1, S2}; ties (< 0.1) go to the variant with the novel components (BC-SSM + NASS).

### E12: Results of the BC-SSM / NASS ablation (E4) and first seeds
**BC-ResNet-8 seed variance (same recipe):**

| Benchmark | Test final epoch (s0 / s1 / s2) → mean ± std | Test @ best valid → mean ± std | Paper |
|---|---|---|---|
| V2-12 | 98.79 / 98.51 / 98.67 → **98.66 ± 0.14** | 98.67 / 98.63 / 98.81 → 98.70 ± 0.09 | 98.7 |
| V1-12 | 97.99 / 98.05 / 98.05 → **98.03 ± 0.03** | 98.02 / 97.99 / 98.05 → 98.02 ± 0.03 | 98.0 |
| V2-35 | 97.86 (s0; s1, s2 running) | 97.84 | – |

Our BC-ResNet-8 reproduces the paper on both V1-12 and V2-12.

**Ablation (V2-12, single seed, all with the BC-ResNet-8 recipe):**

| Model | Params | Best valid | Test @ bv | Test final | 0 dB | −5 dB |
|---|---|---|---|---|---|---|
| BC-ResNet-8 (3-seed mean) | 321 K | 98.31 | 98.70 | 98.66 | 74.50* | 42.47* |
| BC-SSM-7 (idea 1) | 342 K | 98.22 | 98.65 | 98.81 | 74.97 | 40.29 |
| BC-SSM-7 + NASS (ideas 1+2) | 349 K | 98.20 | 98.69 | 98.85 | **76.40** | **43.91** |
| BC-SSM-7 + NASS, unsupervised gate | 349 K | 98.18 | 98.79 | 98.85 | 72.86 | 39.00 |
| BC-SSM-7 + NASS, causal | 301 K | 98.22 | 98.49 | 98.40 | 72.47 | 37.42 |

\* seed 0 only.

**Interpretation (honest):**
1. **Idea 1 (BC-SSM) does not improve clean accuracy.** Validation 98.18–98.22 vs 98.27–98.36 for BC-ResNet-8 seeds; test differences are within seed noise (std 0.14). Whole-utterance context per block gives no measurable gain on 1 s clips, where BC-ResNet's dilated stack already covers the word.
2. **Idea 2 (NASS) improves noise robustness, and the supervision is what does it.** Against the identical architecture with an unsupervised gate: +3.5 points at 0 dB, +4.9 at −5 dB, +1.5 at 5 dB, same clean accuracy. Against BC-SSM without the gate: +1.4 / +3.6. The gate learns frame-level speech presence at 92.5 % agreement with the SNR targets; the unsupervised gate's agreement is 29 %, i.e. it learns something unrelated. Single seed; needs seeds before claiming.
3. **Causal BC-SSM costs ≈ 0.2–0.4 points clean** and some robustness.
4. **For raw accuracy, the best family on validation is NativeKWS-H + KD** (98.45 single teacher; 98.47 at epoch 105 with the 3-model ensemble, still running).

**Decision:** SOTA-accuracy track continues with NativeKWS-H + KD (selected on validation). NASS is kept as the robustness contribution. Next step there is NASS in the H model (not in this wave).

### E13: Stronger teacher T4 and wave 3 (launched 05:30)
Validation accuracy of candidate ensemble teachers (softmax average of `last_ema`):

| Teacher | Valid |
|---|---|
| BC-ResNet-8 × 3 seeds | 98.47 |
| **BC-ResNet-8 × 3 + NativeKWS-H (T4)** | **98.63** |
| BC-ResNet-8 × 3 + H + L | 98.49 |
| BC-ResNet-8 × 3 + H + L + M | 98.56 |
| BC-ResNet-8 × 3 + BC-SSM × 3 | 98.43 |
| all 9 non-KD models | 98.54 |
| S1 teacher (BC-ResNet-8 + L + M) | 98.52 |

Queued: `v2_12_native-H-kdT4` (200 ep), `v2_12_native-H-kdT4-e300` (300 ep: KD students often benefit from longer schedules), and `v1_12_native-H`, `v2_35_native-H` (baselines + 4th T4 member for those benchmarks).

### E14: Interim results (2026-09-15 07:40) and correction
**Correction: "causal" NativeKWS is not fully causal.** In `NativeKWS` the `--causal` flag only removes the backward Mamba branch; the conv front-end (stem 5×5, BC-block temporal convs, collapse conv) keeps centred padding, giving about 17 frames (~170 ms) of look-ahead. `v2_12_native-M-causal` is therefore a *limited-look-ahead* model, not a streaming one. BC-SSM `--causal` *is* fully causal (verified, E4). Must be fixed before any NativeKWS streaming claim.

**Finished since E12:**

| Run | Params | Best valid | Test @ bv | Test final | 0 dB | −5 dB |
|---|---|---|---|---|---|---|
| BC-ResNet-8, V2-35 (s0/s1/s2) | 327 K | 97.96 / 97.93 / 97.96 | 97.84 / 97.86 / 97.89 | 97.86 / 97.89 / 97.96 → **97.90 ± 0.05** | 83.6–83.8 | 63.8–64.0 |
| NativeKWS-H, V1-12 | 296 K | 97.35 | 97.96 | 97.92 | 83.09 | 51.22 |
| BC-SSM-7 + NASS + KD (3-model ens.), V2-12 | 349 K | 98.29 | 98.85 | 98.83 | 78.16 | 43.46 |

**Running (partial, V2-12):** NativeKWS-H + KD (3-model ens.) epoch 177/200: best valid **98.61**, test @ bv 98.96, latest test 98.90. T4-teacher runs at epoch 65.

**Observations:**
* On V2-35 our BC-ResNet-8 (3-seed mean 97.90) already equals Keyword Mamba's published 97.89, so V2-35 "SOTA" is essentially a BC-ResNet-8 tie; beating it cleanly needs ≥ ~98.0.
* KD also helps BC-SSM + NASS (+0.09 valid) *and* its low-SNR robustness improves further (78.2 at 0 dB), unlike NativeKWS-H where KD hurt robustness.
* NativeKWS-H without KD does not beat BC-ResNet-8 on V1-12 (97.92 vs 98.03 ± 0.03).

### E15: NASS in NativeKWS-H (queued 07:50)
`NativeKWS(noise_aware=True)`: the same `SpeechPresenceGate` (log-mel input) drives Δ in both Mamba blocks (both directions). +5,601 params (301,173 total). Old checkpoints load unchanged.
* `v2_12_native-H-nass`: vs `v2_12_native-H` (no KD)
* `v2_12_native-H-nass-kdens`: vs `v2_12_native-H-kdens` (same 3-model teacher)

Rationale: the best accuracy model (H + KD) has no novel component, while NASS is the component with a measured effect. If H + NASS + KD matches H + KD on validation and keeps NASS's robustness gain, it becomes the final model.

### E16: Lost runs after the session ended, and a fix
* **What happened:** at 09:01 the Claude Code login-session scope ended; systemd killed every process in it. `setsid`/`nohup` does not escape the session cgroup. Killed mid-training (no resume support at the time): `v2_12_native-H-kdT4` (ep 112/200), `v2_12_native-H-kdT4-e300` (112/300), `v2_35_native-H` (70/200), `v2_12_native-H-nass` (128/200), `v2_12_native-H-nass-kdens` (73/200). The queue and auto-evaluation also stopped. About 6 h of GPU time lost; noticed at 14:48.
* **Partial validation, same schedule (valid acc, max raw/EMA):** kdT4 ep110 98.29 vs kdens ep110 98.34 (T4 teacher not better so far); H-nass ep125 98.11 vs H ep125 98.25; H-nass-kdens ep70 98.29 vs kdens ep70 98.22. All within noise, so no decision possible; archived in `runs/<run>/aborted-20260915-0901/`.
* **Fixes:**
  1. `kws/train.py --resume`: full training state (model, EMA, optimiser, step, best, RNG) saved atomically to `state.pt` every `eval_every` epochs; on resume the log is truncated to the checkpoint epoch; a fresh start deletes stale logs. Tested (4 → 6 epochs, continuous log).
  2. `scripts/jobqueue.py` always passes `--resume`; restarting the queue continues interrupted runs.
  3. Queue and auto-evaluation now run as **user systemd units** (`systemd-run --user --unit=kws-queue` / `kws-finalize`, `KillMode=process`), outside the session scope. Caveat: lingering is off for the user, so if *all* of the user's login sessions end, the user manager stops too.
* **Restarted (14:53), priority order:** `v2_12_native-H-nass-kdens`, `v2_12_native-H-kdT4`, `v2_35_native-H`, `v2_12_native-H-nass`. **Not restarted:** `v2_12_native-H-kdT4-e300` (no gain at ep110; `STOPPED` marker).
* Operations: `systemctl --user status kws-queue`, `journalctl --user -u kws-queue`, `tail runs/logs/queue.log`.

### E17: Wave-3 / E15 results and the frozen final configuration
**V2-12, single seed (seed 0). Selection metric = validation. Test and noise columns are for the record only.**

| Run | Params | Best valid (epoch) | Mean valid ep 150–200 | Test @ bv | Test final | 0 dB | −5 dB | Gate agree |
|---|---|---|---|---|---|---|---|---|
| NativeKWS-H | 296 K | 98.29 (90) | 98.14 | 98.59 | 98.77 | 81.27 | 49.75 | – |
| NativeKWS-H + NASS | 301 K | 98.27 (145) | 98.09 | 98.94 | 98.79 | 80.35 | 43.60 | 93.7 % |
| H + KD (BC-ResNet-8) | 296 K | 98.45 (130) | 98.36 | 98.85 | 98.92 | 77.46 | 37.57 | – |
| H + KD (T3 = BC-ResNet-8 + L + M) | 296 K | 98.61 (140) | 98.55 | 98.96 | 98.88 | 74.44 | 34.34 | – |
| H + KD (T4 = BC-ResNet-8 ×3 + H) | 296 K | 98.40 (175) | 98.31 | 98.96 | 98.98 | 76.38 | 36.36 | – |
| **H + NASS + KD (T3)** | **301 K** | **98.65 (110)** | **98.58** | 98.90 | **98.98** | 78.83 | 39.92 | 93.6 % |
| BC-SSM-7 + NASS + KD (T3) | 349 K | 98.29 (175) | 98.22 | 98.85 | 98.83 | 78.16 | 43.46 | 91.6 % |
| *BC-ResNet-8, 3 seeds* | 321 K | 98.31 ± 0.05 | 98.19 | 98.70 ± 0.09 | 98.66 ± 0.14 | 74.50* | 42.47* | – |

**V2-35 / V1-12, single seed, no KD:** NativeKWS-H V2-35: best valid 97.93, test 97.88 / 97.99 final (BC-ResNet-8 3-seed 97.90 ± 0.05). V1-12: valid 97.35, test 97.96 / 97.92 (BC-ResNet-8 98.03 ± 0.03).

**Findings**
1. **Teacher choice:** T3 beats T4 as a teacher on validation (98.61 vs 98.40 best; 98.55 vs 98.31 late mean), although T4 is the better ensemble on its own (E13). A diverse teacher (conv + two SSM models) beats a stronger but more homogeneous one (three BC-ResNet seeds). T3 is used from now on.
2. **NASS inside the best model:** with KD, adding NASS gives the highest validation of all runs (+0.04 best, +0.03 late mean vs H + KD T3; within noise) **and** better low-SNR robustness (+4.4 at 0 dB, +5.6 at −5 dB). Without KD, NASS does not change validation (98.27 vs 98.29) and slightly lowers robustness. The robustness effect of NASS in the H model is therefore not yet consistent; seeds are needed.
3. **KD consistently trades low-SNR robustness for clean accuracy** (every KD student < its non-KD counterpart at −5 dB). Candidate future fix: distil only on clean/high-SNR frames, or noise-augmented teacher inference.

**Decision (frozen 2026-09-15 18:55): FINAL = NativeKWS-H + NASS, distilled from T3 = {BC-ResNet-8 s0, NativeKWS-L, NativeKWS-M} trained on the same benchmark.** Selected purely on validation (highest best and late-mean validation); it also contains the novel component.

### E18: Final wave (launched 18:55; queue + finalize as systemd units)
* Teacher members for the other benchmarks (same recipes as V2-12): `v1_12_native-L`, `v1_12_native-M`, `v2_35_native-L`, `v2_35_native-M`.
* Final students, 3 seeds each: `v2_12_native-H-nass-kdens{,-s1,-s2}` (seed 0 already done), `v1_12_native-H-nass-kdens{,-s1,-s2}`, `v2_35_native-H-nass-kdens{,-s1,-s2}`. V1-12/V2-35 students start automatically once their teachers finish (`jobqueue.py` now waits for teachers' `last_ema.pt`).
* Auto-evaluation (noise + protocol) for all of these and for the BC-ResNet-8 seeds that lacked noise evaluation.
* Honest limitation: the configuration was selected on V2-12 validation only and then applied unchanged to V1-12 and V2-35 (no per-benchmark tuning).

### E19: Leakage and overfitting audit (2026-09-15 19:35)
**Data leakage checks (from the raw caches):**

| Check | V1 | V2 |
|---|---|---|
| Speakers train / valid / test | 1,503 / 189 / 189 | 2,112 / 256 / 250 |
| Speaker overlap train∩valid, train∩test, valid∩test | 0 / 0 / 0 | 0 / 0 / 0 |
| Byte-identical waveforms shared across splits | 0 | 0 |
| Official 12-class test clips identical to a train / valid clip | 0 / 0 | 0 / 0 |

**Selection hygiene:**
* Final configuration and teacher chosen on validation only (E13, E17). Checkpoint = best validation; the paper rule (final epoch) is reported alongside.
* Teachers are trained on the train split only; teacher ensembles were chosen on validation.
* Risk that remains: test numbers were *visible* during development (logged every 5 epochs), and early design moves (M → H) were discussed with test and validation side by side. Mitigations: (1) the final choice is validation-only; (2) seeds 1–2 are fresh runs; (3) **V1-12 and V2-35 were never used for any decision**, so they are genuinely held-out benchmarks for the frozen configuration.
* Validation-winner's curse: the final config is the maximum of ~15 configurations on a 4,445-clip validation set, so its validation score (98.65) is optimistically biased; test and seeds are the unbiased estimates.
* Validation < test for every model (e.g. BC-ResNet-8 98.31 vs 98.66), because validation silence is synthetic noise at up to full scale. The gap is similar across models, so it is not a sign of test overfitting.
* Statistical resolution: binomial standard error of a single 99 %-accuracy measurement on 4,890 clips ≈ 0.14 points; differences under ~0.3 between single runs are not significant.

**Caveats to state in any write-up:**
* Noise-robustness test uses the same six GSC `_background_noise_` recordings as training augmentation (different random segments and much higher levels): matched-noise robustness, **not unseen noise**. An unseen-noise test (e.g. MUSAN/DEMAND) is still needed.
* Our model is trained with KD from a 3-model teacher (1.2 M params in total at training time); most published baselines are not distilled. Inference cost is the 301 K student only. Non-KD numbers are reported too.
* Published baselines use their own recipes and protocols (E6); only BC-ResNet-8 was re-run in our pipeline (it matches or exceeds its paper).

### E20: Status 20:20 and queue hardening
* **Progress:** V2-12 final seeds s1 / s2 at epoch 53 / 51 (≈ 93 s/epoch → ~4 h left); V1-12 teachers L 63/200, M 104/200; V2-35 teachers L 16/200 (293 s/epoch → ~15 h), M 27/200 (179 s/epoch → ~9 h). V1-12 and V2-35 students wait for their teachers. V2-35 is the bottleneck: expected ~1–1.5 days.
* **GPU 0 memory at 91 / 96 GB** (our three jobs + another user's 15 GB), so launching the V1-12 students there could run out of memory. `jobqueue.py` now launches only if `nvidia-smi` reports ≥ `--min_free_gb` (32) free, and reserves that amount per launch within a poll.
* **Queue bug fixed:** a restarted queue ignored jobs started by an earlier queue and exited ("only blocked jobs remain"). Exit conditions now consider every running job named in the jobs file. Running as unit `kws-queue3`. Trainers were never interrupted (`KillMode=process`).

## 2026-09-16

### E21: Final results: V2-12 and V1-12 complete (3 seeds), V2-35 partial (1/3)
Final model: NativeKWS-H + NASS (301 K), KD from T3 of the same benchmark. Test accuracy %, mean ± sample std over seeds 0, 1, 2. "Official" = released 12-class test set; "list" = testing_list-built test set (KWT / Keyword Mamba protocol, E6).

| Benchmark | Rule | **Ours (301 K)** | BC-ResNet-8, our pipeline (321 K) | Keyword Mamba KWM-T-192 (5.2 M, paper, 3-run mean) |
|---|---|---|---|---|
| V2-12 | final epoch, official | **99.02 ± 0.05** (98.98 / 99.00 / 99.08) | 98.66 ± 0.14 | 98.91 |
| V2-12 | final epoch, list | **99.05 ± 0.08** | 98.76 ± 0.13 | (their protocol) |
| V2-12 | best-valid ckpt, official | 98.96 ± 0.06 | 98.70 ± 0.10 | – |
| V1-12 | final epoch, official | **98.13 ± 0.04** (98.09 / 98.15 / 98.15) | 98.03 ± 0.04 | 98.05 |
| V1-12 | final epoch, list | 97.95 ± 0.02 | 97.77 ± 0.07 | (their protocol) |
| V1-12 | best-valid ckpt, official | 98.11 ± 0.08 | 98.02 ± 0.03 | – |
| V2-35 | final epoch (s1 only) | 98.01 (n = 1) | 97.90 ± 0.05 | 97.89 |
| V2-35 | best-valid ckpt (s1 only) | 98.08 (n = 1) | 97.86 ± 0.03 | – |

**Verdict against the pre-registered rule (E7, rule 4: 3-seed mean > best published + 1 std):**
* **V2-12: passes on the papers' reporting rule** (99.02 > 98.91 + 0.05), and also on the testing_list protocol they actually used (99.05 ± 0.08). On our stricter best-valid-checkpoint rule it is borderline (98.96 vs 98.91 + 0.06 = 98.97): **not** a clean pass.
* **V1-12: not clean.** Official-test final epoch passes (98.13 > 98.05 + 0.04), but on the testing_list protocol that Keyword Mamba used our model scores 97.95, below their 98.05. On V1 the two protocols differ by ~0.2 points (larger than on V2, ≤ 0.1), so the cross-paper comparison is ambiguous. Best-valid rule also fails (98.11 < 98.13).
* **V2-35: pending** (seeds 0 and 2 at epoch 145/200).
* Against BC-ResNet-8 trained in the **same pipeline**, the final model is better on every benchmark and rule: +0.36 (V2-12), +0.10 (V1-12), +0.11 (V2-35, n = 1) on final-epoch test.

**Noise robustness (mean over seeds, matched-noise test, E19 caveat):** V2-12 0 dB 77.1 vs BC-ResNet-8 73.3 (+3.8), −5 dB 37.9 vs 38.5 (−0.6); V1-12 0 dB 79.1 vs 78.3, −5 dB 40.9 vs 45.3 (−4.4); V2-35 (n = 1) 0 dB 84.8 vs 83.7, −5 dB 66.4 vs 63.9. **Robustness gains are not consistent** at the lowest SNR: KD's clean-speech bias (E17) offsets NASS.

## 2026-09-29

### E22: V2-35 complete (3 seeds): second clean pass
| Benchmark | Rule | Ours (301 K) | BC-ResNet-8, our pipeline | KWM-T-192 (5.2 M, paper) | Verdict vs the pre-registered rule |
|---|---|---|---|---|---|
| V2-35 | final epoch | **98.07 ± 0.09** (98.03 / 98.01 / 98.18) | 97.90 ± 0.05 | 97.89 | **passes** (+0.18 > 1 std) |
| V2-35 | best-valid ckpt | **98.07 ± 0.02** | 97.86 ± 0.03 | – | passes |

Summary over the three benchmarks: **V2-12 and V2-35 are clean passes** on the papers' reporting rule; **V1-12 remains ambiguous** (protocol-dependent, E21). Noise robustness (V2-35, mean of 3 seeds): 0 dB **85.10** vs BC-ResNet-8 83.69; −5 dB **66.26** vs 63.90 — on this benchmark the gain holds at both low SNRs, unlike V2-12 and V1-12 (E21).

### E23: Literature notes written up (`docs/LITERATURE.md`)
Everything consulted during the project, with provenance tags ([V] source opened, [S] search summary, [M] background knowledge — unverified) and a scope caveat: this is **not** a systematic review. Key consequences recorded there:
* **Ensemble KD into a tiny KWS student is prior art** (curriculum learning + weighted ensemble distillation for noise-robust SF-KWS, IEEE Access 2023) — our distillation setup is not novel and must cite it.
* **Learned frame skipping is prior art** (Skip RNN, ICLR 2018, also used for KWS) — a skipping contribution must rest on the SNR supervision, the exactness of skipping in a selective SSM, and measured device energy.
* **Mamba on MCUs already exists** (MambaLite-Micro 2025; FEMBA on GAP9 2026; BabyMamba-HAR 2026) — "first Mamba on a microcontroller" is unavailable; notably all of them keep the recurrent path in float because it resists quantization, without analysing why.
* Richest related-work entry point: the June 2025 SF-KWS survey (~250 papers).
* Six gaps that look open, including a contractivity-based error bound for integer selective scans, decay-aware checkpointing for intermittent power, streaming execution planning in MCU toolchains, and unseen-noise evaluation hygiene.

### E24: Real-microphone test set — rig built and verified in simulation (2026-10-16)
* **Hypothesis:** the accuracy cost of the actual deployment microphone (INMP441 → ESP32 I2S) is an unmeasured quantity that is probably larger than the fractions of a point separating us from published results, and no public benchmark contains it.
* **Design.** Rather than record new speech (which confounds new speakers, new room and new microphone), the primary set **replays the official V2-12 test clips through a speaker into the INMP441**. Utterances, speakers, labels and class balance are bit-identical to the cached set, so cached-vs-recorded accuracy is a *paired*, per-clip measurement of the acoustic path alone. Protocol, condition grid and pre-registered validity rules: `docs/DATASET.md`.
* **Built.** `rig/firmware/i2s_stream` (framed int32 I2S stream with sequence numbers so dropped blocks are detected, not silently absorbed), `rig/capture.py`, `rig/build_stimulus.py` (fixed-seed shuffle; periodic sweeps for resync), `rig/align.py` (sweep tracking + per-segment interpolation + per-clip SNR from the leading gap), `rig/eval_real.py` (paired scoring, broken/fixed clip counts, accuracy by measured SNR).
* **Verified** against a synthetic recording with +230 ppm clock drift, 137 ms delay, spectral colouring, additive noise and DC offset: clock recovered **+230 ppm**; per-clip alignment **median +1 sample** (p5–p95 −1…+2); waveform correlation to the original **median 0.949**; labels preserved; SNR estimator tracks 10 dB input steps.
* **Two bugs found and fixed while testing.** (a) The first sweep tracker anchored on the last sweep by searching the end of the file, but the last sweep is followed by 50 more clips — it estimated +197,000 ppm. Replaced with a sequential tracker that re-estimates the clock from the sweeps already found. (b) The cached test set is ordered by class and `_silence_` sorts first, so a short `--limit` take was 100 % silence and the stimulus would have played all of one class at a time, correlating class with any slow drift in room conditions. Fixed with a fixed-seed shuffle.
* **Hardware correction.** The board is a plain **ESP32**, not an ESP32-S3. For recording this is immaterial (same I2S peripheral; the stream moves to the USB-UART bridge at 2 Mbaud, ~3× headroom over the 64 kB/s needed). For *deployment* it matters: Espressif's ESP-NN benchmark runs the same int8 Person Detection model in **380 ms on ESP32 vs 54 ms on ESP32-S3** at the same 240 MHz, because the S3's LX7 has vector instructions with hand-written assembly kernels while the ESP32 "gets generic optimisations". Our 63 MMAC/s real-time budget was already 20–60 % of one S3 core, so the original ESP32 cannot run this model live. All chip budgets in `ARCHITECTURE.md` assume the S3 and remain written for the S3.
* **Decision:** proceed with data collection on the ESP32 (it is a sufficient recorder); an ESP32-S3 is required before the deployment phase. **Nothing recorded yet** — no results from this set exist.

### E25: Live microphone path — record real voices, and listen in real time (2026-10-16)
* **Why:** the replay rig (E24) measures the acoustic channel rigorously but never hears a live person, and nothing in the project had yet run the model on a microphone at all.
* **Built.** `rig/stream.py` (shared framed-block reader), `rig/record_live.py` (prompted set-B recording: fixed-seed random word order, 5 s room-noise reference, energy-centred 1 s trim, on-the-spot rejection of silent/clipped/dropped takes), `rig/make_set.py` (sessions → cache file, one gain per session so within-session level differences survive), `rig/listen.py` (live keyword spotting on the host: posterior smoothing + release-threshold hysteresis). `capture.py` refactored onto `stream.py`.
* **Verified against a virtual serial port** driven by an emulated firmware:
  * clean stream decoded **bit-exactly** (0 error over 51,200 samples);
  * three injected block drops detected, zero-filled in the right positions, and block 160 **still bit-exact** — a dropout cannot silently shift labels;
  * injected bus garbage resynchronised without losing the stream;
  * `capture.py` wrote a bit-exact WAV with a correct sidecar;
  * `listen.py` fed six real keyword clips detected **all six, in order, no duplicates** (yes 0.97, no 0.71, left 0.77, stop 0.85, go 0.99, up 0.80);
  * set-B trimmer centres the word to **median 10 ms** (p90 60 ms), >98 % of word energy retained, for words placed randomly in a 2 s window.
* **Three bugs found by these tests.**
  1. **Frame misalignment (was also in the committed E24 code).** `resync()` consumed a block's 8-byte header and then the main loop read a whole frame, leaving the reader permanently offset by 8 bytes. `resync()` now returns a complete block so the reader always ends on a frame boundary.
  2. **Detection fired twice per word.** A keyword stays in the 1 s sliding window after it ends, so a fixed refractory timer has to guess how long. Replaced with hysteresis: re-arm only when the fired class falls below `--release` (0.4).
  3. **8× slower than real time on CPU** — torch defaulted to all cores on this many-core host and thrashed. `--threads` (default 4) gives 32 ms per window; `listen.py` now also warns if it cannot keep up.
* **Expectations recorded in `docs/DATASET.md`:** only the ten keywords exist (anything else is correctly `_unknown_`), and GSC is predominantly American-accented English, so an accent gap is expected — and is one of the few effects a small live set can honestly demonstrate, being large where a half-point benchmark difference is not. Privacy rules for set B added (consent, anonymous ids, audio stays out of the repo).
* **Still unverified:** everything that needs the physical board — I2S bit alignment and slot selection, real UART throughput, and whether a small loudspeaker reproduces the sweep well enough to anchor each segment.

### E26: Milestone 1 (`docs/ONCHIP.md`) — BC-ResNet-8 int8 on the plain ESP32: does not fit (2026-10-28)
* **Goal:** run `deploy/onchip/bcresnet8.onnx` on the ESP32 over the 4 baked-in log-mels, match `tv_expected_label`, measure latency.
* **Toolchain:** ESP-IDF **v5.4.2**, esp-ppq **1.3.13** (torch 2.15.0+cpu, onnx 1.19.1), esp-dl component **3.3.14**. Board: **ESP32-D0WDQ6 rev 1.0**, 240 MHz, 4 MB flash, **no PSRAM**. Everything installed under `D:\esp` (C: had 9 GB free). ESP-IDF refuses to install or run from Git Bash/MSYS; use PowerShell.
* **Built:** `scripts/quantize_onchip.py` (PTQ, target `"c"`, w8a8, 128 calibration tensors, "yes" vector embedded as the `.espdl` test input; also runs the quantized graph on the host for all 4 vectors) and `deploy/esp32_kws/` (ESP-IDF project; model in flash `.rodata`, `param_copy=false`; quantizes `tv_input` with the input exponent, argmax over raw int8 logits). Report: `deploy/esp32_kws/quantize_report.txt`; device log: `deploy/esp32_kws/device_output.txt`.
* **Quantization (host, esp-ppq simulation):** 156/156 ops quantized, no unsupported operators. All 4 vectors keep the correct argmax, but margins collapse and logits drift (max |Δ| 5.2–13.4): yes 6.75, no 3.0, stop 2.0, silence 6.0 (float margins ≈18.6, 8.7, 9.8, 12.3). Graph-wise noise:signal reaches **58 %** at `body.11` although every layer alone is ≤ 0.11 % — error accumulates through depth, not in any single layer.
* **esp-ppq deployment error (negative result):** `bnfold_mul_0, 1, 2, 10, 11` (the SubSpectralNorm BN rewritten as `Mul`) violate ESP-DL's int8 Mul constraint `output_exp − in0_exp − in1_exp ≥ 0`, so ESP-DL would compute them wrongly on chip even though PPQ's simulation doesn't show it. Tried dispatching those 5 to w16a16: the error **moved to `bnfold_mul_9`** and the host simulation then **misclassified `stop` as `_silence_`** (margins 1.5/1.25/0.75/5.0). Worse — not adopted. The proper fix is in the export (fold the SSN affine into the adjacent conv where the reshape permits, or quantization-aware handling), not in PPQ settings.
* **Build:** OK. App 0x122090 B (1.19 MB; 23 % of the 1.5 MB factory partition free); static DRAM 16.6 KB (.data 8.8 KB, .bss 7.8 KB); `.espdl` 462 KB in flash.
* **On device — FAILED, and the cause is not the one the doc anticipated.** Weights stayed in flash as planned; the **activation arena** did not fit: `dl::ModelContext: Failed to alloc 391.38KB internal RAM, largest available internal RAM block size 108.00KB` → `MemoryManagerGreedy: root_alloc failed`. Free internal heap at boot is 366 KB total but the largest contiguous block is 155 KB. Shape inference shows why: the stem output `[1,128,20,101]` is **252.5 KB in int8 by itself**, and `body.2` produces 189 KB tensors. No inference ran, so **no predictions and no latency figure exist.** (First firmware dereferenced the unallocated tensors and Guru-Meditation crash-looped; now guarded.)
* **Conclusion:** BC-ResNet-8 at full 40×101 resolution cannot execute under ESP-DL on a PSRAM-less ESP32 — independent of quantization. Options, each a decision rather than a fix: (a) an ESP32/ESP32-S3 module **with PSRAM** (consistent with E24's conclusion that deployment needs an S3); (b) a smaller model (BC-ResNet-1/2: the stem width scales with τ, so τ=1 gives a ~32 KB stem tensor); (c) streaming/chunked execution along time.
* **Also noted:** esptool's automatic reset into download mode on this board is intermittent ("Wrong boot mode detected (0x13)"); retries succeed.

### E27: Two fixes from the first device bring-up (2026-10-28)
The first ESP32 bring-up (E26, run on the machine with the board) got as far as a flashed, quantized BC-ResNet-8 and then failed to run. Both causes were on the training/export side, so both are fixed here.

**Fix 1 — the BatchNorm rewrite was the wrong primitive.** E24's export wrote the 12 surviving SubSpectralNorm BatchNorms as `Mul` + `Add`. Mathematically identical, whitelisted, and verified against PyTorch to 1.1e-05 — and wrong on device. ESP-DL's int8 `Mul` requires `output_exponent - input0_exponent - input1_exponent >= 0`; the small SubSpectralNorm scales violate it, and the deployment check rejected 5 nodes (`bnfold_mul_0/1/2/10/11`). The dangerous part is that **esp-ppq's PC simulation showed no problem** — all 4 test vectors still predicted correctly there — so this would have shipped as a silent numerical fault.
* Now rewritten as **depthwise `Conv` with kernel 1** (`group == in_channels`, the only grouping ESP-DL supports besides 1), carrying the affine in weights and bias. A Conv gets a per-output-channel quantization scale of its own, so the exponent constraint does not arise. Verified: 12 nodes converted, operator set all-whitelisted, logits match PyTorch to 1.1e-05, predictions unchanged.
* The surviving `Mul` nodes are the architecture's own SiLU (`x*sigmoid(x)`) and were not flagged by the device check. If they ever are, ESP-DL has a `Swish` operator to fuse them into.
* **Testable prediction for the next device run:** E26 measured accumulated quantization noise reaching 58 % in the last block and the `stop` margin collapsing from 9.8 (float) to 2.0 (int8), and found that forcing those 5 `Mul`s to int16 made it *worse* (error moved to `bnfold_mul_9`, `stop` then misclassified as `_silence_`). That pattern is what a badly-scaled primitive looks like, so the depthwise-Conv version should show a noticeably larger margin. If it does not, the noise has another source and the SiLU `Mul`s are the next suspect.

**Fix 2 — BC-ResNet-8 cannot fit this chip, and the reason generalises.** The failure was the activation arena, not the weights (which stayed in flash as designed): the stem output `[1,128,20,101]` is 252.5 KB in int8 alone, ESP-DL asked for a 391 KB arena, and the board (ESP32-D0WDQ6, no PSRAM) had ~366 KB free but a largest *contiguous* block of 108 KB. Largest single activation by scale, computed on the training machine:

| model | params | largest activation (int8) | fits ~108 KB contiguous |
|---|---|---|---|
| BC-ResNet-1 | 9,232 | 31.6 KB | yes, comfortably |
| BC-ResNet-2 | 27,284 | 63.1 KB | yes |
| BC-ResNet-3 | 54,168 | 94.7 KB | yes, tight |
| BC-ResNet-4 | 89,884 | 126.2 KB | no |
| BC-ResNet-8 | 321,068 | 252.5 KB | **no — this is what failed** |

* **Started training `bcresnet-1`, `bcresnet-2`, `bcresnet-3` on v2_12** with the official BC-ResNet SGD recipe, so the scales are directly comparable to our τ=8 reference. This was already item 4 of "Planned next"; the device just made it urgent. Deploy τ=2 first, then τ=3.
* `scripts/export_onchip.py` now takes `--run`, writes to `deploy/onchip/<model>/`, and prints the largest activation, so whether a model fits is known before anything is flashed.
* **The important consequence:** NativeKWS-H's whole-window peak is 189 KB, so *the final model does not fit this chip whole-window either*, at any quantization setting. Its streaming peak is ~24 KB. The gap between those two numbers is exactly the open problem recorded in `docs/LITERATURE.md` §11 — every MCU toolchain surveyed is whole-window and none plans streaming execution. On this board that stops being a literature observation and becomes the thing that has to be built.

**Not fixed, noted:** the board was initially invisible on USB (no CP210x/CH340/FTDI enumeration at all) and later worked; cause unknown. ESP-IDF v5.4.2 will not install from Git Bash (MSYS unsupported) — use PowerShell. Automatic flash-mode entry intermittently fails with `Wrong boot mode (0x13)`; retrying always worked, so the BOOT button was never needed.

### E28: Small BC-ResNet scales trained for the ESP32 — and they beat the paper (2026-10-28)
* **Why:** E26/E27 established that BC-ResNet-8 cannot fit a PSRAM-less ESP32 (arena 391 KB vs a 108 KB largest free block). Smaller scales were item 4 of "Planned next"; the device made them the critical path.
* **Trained** `bcresnet-1/2/3` on v2_12 with the official BC-ResNet SGD recipe (lr 0.1, wd 1e-3, bs 100, no label smoothing), identical to the τ=8 reference, so the scales are directly comparable.

| model | params | ours, best-by-valid | ours, last EMA | BC-ResNet paper | Δ vs paper | largest activation | arena ≈1.55x | fits 108 KB |
|---|---|---|---|---|---|---|---|---|
| BC-ResNet-1 | 9,232 | 96.81 | 96.99 | 96.6 | +0.21 | 31.6 KB | ~49 KB | **yes** |
| BC-ResNet-2 | 27,284 | **98.02** | 98.02 | 97.4 | **+0.62** | 63.1 KB | ~98 KB | yes, ~10 KB spare |
| BC-ResNet-3 | 54,168 | **98.36** | 98.40 | 97.6 | **+0.76** | 94.7 KB | ~147 KB | no |
| BC-ResNet-8 | 321,068 | 98.67 | 98.83 | 98.7 | −0.03 | 252.5 KB | 391 KB measured | no |

* **The interesting result is the shape of the Δ column.** Our pipeline reproduces the paper almost exactly at τ=8 (−0.03) but beats it by **0.62–0.76 at τ=2 and τ=3**. The optimiser recipe is the paper's, so the gain comes from the rest of the pipeline (SpecAugment settings, EMA, per-epoch `_unknown_` resampling, time shift) and it helps small models far more than large ones. Two consequences: the published small-scale BC-ResNet numbers understate the architecture, and the understatement is concentrated exactly in the parameter regime where MCU deployment actually lives. This is worth a paragraph in the paper — accuracy-per-parameter curves in this literature are usually quoted from the original papers without re-running the small scales.
* **Exported** all three to `deploy/onchip/bcresnet-{1,2,3}/` (ONNX + 128 calibration tensors + 4 test vectors + host logits). Every one: 12 BatchNorms rewritten as depthwise Conv, **no operator off the ESP-DL whitelist**.
* **Deployment target is BC-ResNet-2**: 98.02 % at 27 K parameters — 12x smaller than τ=8 for 0.65 points, and it fits. τ=3 would add +0.34 but needs streaming execution or PSRAM.
* **Still untested on hardware.** No on-chip prediction or latency figure exists yet for any scale.

### E30: Cross-layer equalization for the SubSpectralNorm folds (2026-10-28)
Follows E29, which was recorded on the `e26-esp32-bringup` branch.

* **E27's reasoning was wrong and E29 caught it.** E27 justified rewriting the SubSpectralNorm BatchNorms as depthwise Conv on the grounds that "a Conv gets a per-output-channel quantization scale of its own, so the exponent constraint does not arise". **ESP-DL quantizes per tensor on ESP32 and ESP32-S3**; per-channel Conv exists only on P4/S31. The rewrite still helped (τ=8 `stop` margin 2.0 → 4.25, E29), so the conclusion held while the premise did not — the depthwise Conv removed the `Mul` exponent *rule* violation but left the underlying problem, which is the channel-scale spread itself.
* **The actual problem.** With one power-of-two exponent per tensor, a fold whose channel scales span three orders of magnitude loses its small channels entirely. E29 measured τ=2 `bnfold_conv_0` at 581× spread, 39/80 channels zeroed, 81.9 % layerwise noise, and `no` misclassified as `_unknown_`.
* **Fix: exact cross-layer equalization in the export.** The reshape in front of a fold maps Conv output channel `k` to fold channels `k*S … k*S+S-1`, so scaling the Conv's channel `k` (weights *and* bias) by `g[k]` while dividing those `S` fold channels by `g[k]` is an **exact identity**. The unusual part is that the fold is per-(channel, sub-band) while the Conv is only per-channel, so `g` can carry only the across-channel part of the spread; the within-sub-band residual stays.
* **Per-layer damping, selected automatically.** `g` is the geometric mean over sub-bands raised to `alpha`, with `alpha` chosen per layer from {0.25, 0.5, 0.75, 1.0} to minimise the number of destroyed channels across the *pair*. This matters: moving the full spread fixes the fold but inflates the Conv feeding it, and a blanket `alpha = 1` made τ=1 `bnfold_conv_10` **worse** (spread 2792× → 3300×, zeroed 6 → 8). A layer is rewritten only if some `alpha` strictly helps, and only if the Conv's output has no second consumer — otherwise rescaling it would corrupt a residual branch. Three layers in τ=1 and τ=3 are left untouched on that basis.
* **Result.** Channels left with ≤2 quantization levels under a single per-tensor scale:

| model | destroyed channels | worst layer | float logits vs PyTorch |
|---|---|---|---|
| BC-ResNet-1 | 87 → **40** | — | 3.8e-06, argmax unchanged |
| BC-ResNet-2 | 249 → **74** | `bnfold_conv_0` 72 → **1** | 2.4e-06, argmax unchanged |
| BC-ResNet-3 | 324 → **56** | `bnfold_conv_1` 78 → **4** | 3.8e-06, argmax unchanged |
| BC-ResNet-8 | 676 → **51** | `bnfold_conv_0` 123 → **3** | 9.5e-06, argmax unchanged |

* **Caveat, stated plainly:** destroyed-channel count is a *screening* metric chosen because it predicts the observed failure, not a measurement of accuracy. Whether τ=2 now gets 4/4 has to be settled by esp-ppq's layerwise noise table and a device run. It is also possible that some of the zeroed channels were dead anyway — the τ=2 Conv feeding `bnfold_conv_0` has a 7.9e10 weight spread with one channel at essentially zero, and E29 found that layer harmless, so spread alone does not imply damage.
* **Not addressed here:** the 61 `Transpose` ops costing 151 ms (15.8 % of τ=2's runtime) that esp-ppq inserts around every SubSpectralNorm `Reshape` for the NHWC/NCHW change. Those come from the same reshape structure that causes the quantization problem, so a reformulation of SubSpectralNorm that avoids the reshape entirely would address both at once. That is a model-design change, not an export change.

### E31: Causal BC-ResNet, and what streaming would buy on the base ESP32 (2026-10-28)
* **Context:** no ESP32-S3 is available, so everything has to work on the plain ESP32. Re-examining the plan under that constraint shows the S3 was never required — and that the base chip is the *better* platform for the streaming claim, because its limits actually bind.
* **Causal plumbing added.** `BCResNet(causal=True)` and `BCBlock(causal=True)` put left-only time padding on the stem (5 wide in time), every `f1` temporal conv, and the head's 5-wide temporal depthwise conv. Verified: parameter counts identical to non-causal (9,232 / 27,284 / 321,068 — only padding differs); the non-causal path is untouched, with existing checkpoints loading and reproducing stored logits to 2.9e-06; and **zero look-ahead** — randomising frames ≥60 leaves every feature column before 60 bit-identical, where the non-causal model leaks back 12 frames (120 ms).
* **Streaming memory, computed from the graph.** BC-ResNet's temporal conv `f1` operates on the frequency-averaged signal (`x.mean(dim=2)`), so a layer's retained history is `C x 2*dilation` values rather than `C x F x 2*dilation`. Peak = total history + a double-buffered working column:

| model | whole-window | streaming | reduction | fits 108 KB |
|---|---|---|---|---|
| BC-ResNet-1 | 31.6 KB | **3.0 KB** | 10x | both |
| BC-ResNet-2 | 63.1 KB | **5.9 KB** | 11x | both |
| BC-ResNet-3 | 94.7 KB | **8.8 KB** | 11x | streaming only |
| BC-ResNet-8 | 252.5 KB (391 KB arena measured) | **23.3 KB** | 11x | streaming only |

* **The headline this gives us:** streaming puts **BC-ResNet-8 (98.67 %) on a plain ESP32 that cannot hold it whole-window at any quantization setting** — 391 KB requested against a 108 KB largest free block, versus 23.3 KB streamed. That is a stronger result than demonstrating it on an S3 with PSRAM, where nothing would have been binding.
* **Latency is a separate constraint and streaming does not resolve it at every scale.** From τ=2's measured whole-window throughput (7.32 M MACs in 949 ms = 7.7 MMAC/s): streamed per-frame cost is ~3.2 ms for τ=1, ~9.4 ms for τ=2, ~18.7 ms for τ=3, ~110 ms for τ=8, against a 10 ms frame budget. So τ=1 streams comfortably in real time, τ=2 is borderline with no headroom for the log-mel front-end, and τ=8 fits in memory but not in time. **These are estimates:** they extrapolate a throughput that includes the 16 % Transpose tax, and streaming will be *less* efficient per MAC because tiny per-call tensors give worse locality and more fixed overhead. Measuring this is the next device task.
* **Training started:** causal `bcresnet-1/2/3/8` on v2_12, official SGD recipe, so the accuracy cost of losing look-ahead is measured rather than assumed.
* **Correction to earlier advice.** E29's and `ONCHIP.md`'s suggestion that an ESP32-S3 is "required for the deployment phase" was wrong as stated. The S3 buys ~7x speed and a PSRAM option; it does not fix per-tensor quantization (ESP32 and S3 are both per-tensor; per-channel Conv exists only on P4/S31), and it is not needed for the streaming contribution, which the base chip demonstrates more convincingly.

---

### E29: BC-ResNet-2 runs on the ESP32 — 949 ms, bit-exact with esp-ppq, 3/4 correct; τ=3 does not fit (2026-10-28)
* **Setup.** Merged `origin/main` (759dd85) into `e26-esp32-bringup` — no conflicts. `scripts/quantize_onchip.py` now takes `--src <export dir>` (ONNX, `calib_128.npy`, `test_vectors.npz` all from that one directory) and writes `model.espdl`, so the embedded symbol is always `_binary_model_espdl_start`; `main/CMakeLists.txt` includes `../../onchip/bcresnet-2`. Same toolchain as E26 (ESP-IDF v5.4.2, esp-ppq 1.3.13, esp-dl 3.3.14, ESP32-D0WDQ6, no PSRAM). Deployed τ=2 as instructed, although `docs/ONCHIP.md` says τ=1 first; τ=1 was not run.
* **The deployment check still fires — on `bnfold_conv_0`, a Conv.** Same rule as E26: `output_exponent - input0_exponent - input1_exponent >= 0`. 144/144 ops quantized, no unsupported operators, no other errors. **Cause: on target `"c"` every op is per-tensor.** ESP-DL's quantization docs: ESP32 and ESP32-S3 use per-tensor quantization; per-channel Conv exists only on ESP32-P4/S31. E27's premise that "a depthwise Conv gets a per-output-channel scale of its own" does not hold on this chip. Measured: τ=2's `bnfold_conv_0` has a **581× spread** between its largest and smallest channel scale, so with one power-of-two weight exponent **39 of 80 channels quantize to zero** and 75 of 80 keep ≤ 2 LSB. Layerwise noise:signal **81.9 %** for that layer alone; every other layer ≤ 0.52 %. Spreads elsewhere: τ=3 `bnfold_conv_0` 28× (nothing zeroed), τ=3 `bnfold_conv_1` 4,761×, τ=8 `bnfold_conv_2` 7,008×.
* **A flag is not the same as damage.** τ=3 flags `bnfold_conv_1`, τ=8 (depthwise form) flags `bnfold_conv_0` and `bnfold_conv_2`, yet their layerwise noise is only 1.9–2.8 % and their predictions are fine. Only τ=2's `bnfold_conv_0` is both flagged and destructive. Read the layerwise table, not just the ERROR lines.
* **Device, τ=2 — it runs.** `model->test()` **PASS** (max_error 0.0005): the chip is bit-exact with esp-ppq, so every host-simulated number here is what the chip computes. Input `logmel` int8 `[1,40,101,1]` exp −3; output int8, 12 values, exp −3.

  | vector | expected | chip | int8 margin | float margin | time |
  |---|---|---|---|---|---|
  | yes | 11 | 11 ✓ | 4.875 | 15.66 | 949.15 ms |
  | no | 5 | **1 `_unknown_` ✗** | `no` ranks 4th, 2.125 below the winner | 7.51 | 949.22 ms |
  | stop | 9 | 9 ✓ | 0.375 | 8.08 | 949.25 ms |
  | `_silence_` | 0 | 0 ✓ | 0.25 | 7.70 | 949.26 ms |

  **3/4 — fails the milestone's pass criterion.** The chip logits equal the host's int8 logits, so this is a quantization failure, not a deployment one.
* **Latency: 949 ms per inference** (profiled sum 951.0 ms over 211 modules): Conv 680 ms (71.5 %, of which the stem conv alone is 194 ms), **Transpose 151 ms (15.8 %, 61 ops that esp-ppq inserts around every SubSpectralNorm `Reshape` for the NHWC/NCHW layout change)**, ReduceMean 48 ms, RequantizeLinear 28 ms, Add 24 ms, Relu 16 ms, Swish 3 ms.
* **Memory, τ=2:** arena (`variable`) **98.64 KB** internal; `others` 96.51 KB internal; total 195.15 KB internal; model 175.64 KB in flash (36.91 KB parameters, `param_copy=false`). Heap: boot 366 KB free / 152 KB largest block → after load 161 KB free / 76 KB largest; minimum ever free 88.7 KB. Build: app image 905,758 B (41 % of the 1.5 MB partition free), `.rodata` 321 KB, static DRAM 16.6 KB, IRAM 50.8 KB.
* **The SIMD message is a symptom of size, not a cap misconfiguration.** `Input cap=0x804 can not callocate with MALLOC_CAP_SIMD` does **not** appear at τ=2 and **does** appear at τ=3 and τ=8, in both cases immediately before the arena allocation fails. `0x804` is `MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT`: it is ESP-DL reporting the failed request, not a separate problem. This contradicts the hypothesis in `docs/ONCHIP.md`.
* **Device, τ=3 — does not fit, and the 1.55× rule is wrong.** `Failed to alloc 191.75KB internal RAM, largest available internal RAM block size 108.00KB`. Arena / largest activation: τ=2 98.64/63.1 = **1.56×**, τ=8 391.38/252.5 = **1.55×**, τ=3 191.75/94.7 = **2.02×**. The greedy memory manager's packing depends on the graph, so the arena has to be measured, not extrapolated. The τ=3 host simulation is otherwise the healthiest of all (4/4; margins 14.25 / 5.75 / 13.00 / 1.25 vs float 16.87 / 7.03 / 13.93 / 7.93). Build: image 937,326 B, 39 % free.
* **Usable ceiling ≈ 108 KB contiguous, not 152 KB.** The same 108 KB figure appears in both failed runs (τ=3, τ=8): ESP-DL allocates its other model structures (~80–97 KB) before the arena and splits the 152 KB block that is free at boot.
* **Did the depthwise-Conv fix work? At τ=8, yes.** Quantizing the new τ=8 graph on the host (bit-exact with the chip, as above): margins 14.0 / 5.5 / **4.25** / 7.0 vs E26's Mul form 6.75 / 3.0 / **2.0** / 6.0 — `stop` doubles, as E27 predicted. **The SiLU `Mul`s are not the suspect:** esp-ppq fuses them into ESP-DL `Swish` ops (12 ops, 2.8 ms), so no SiLU `Mul` reaches the chip. τ=2's collapse is explained entirely by `bnfold_conv_0`'s per-tensor weight quantization.
* **Not tried, by instruction (report, do not work around):** cross-layer equalization in esp-ppq, w8a16/w16a16 for `bnfold_conv_0` only, or an export that limits the per-channel spread of the SubSpectralNorm affine. Each is a candidate fix for τ=2.
* **esp-ppq rewrites the ONNX it is given, in place.** After the first runs, `deploy/onchip/bcresnet-{2,3,8}/*.onnx` showed as modified (each ~4 KB smaller): same 156 nodes and 95 initializers, logits identical to the committed file (max |Δ| 0.0), so only the serialization changed and every result above stands. The files were restored from git, and `quantize_onchip.py` now hands esp-ppq a temporary copy, so `deploy/onchip/*` stays untouched (verified clean after a rerun). The `.espdl` output is **not byte-reproducible**: a rerun differs in auto-generated `PPQ_Variable_N` / `PPQ_Operation_N` numbering and serialization order. With those names normalized, all 158 initializers (values and exponents) and all 211 ops are identical, so the committed `model.espdl` is the one that was flashed.
* **Flashing, for the record:** the board's auto-reset into download mode failed 3/3 this time. Two attempts that did connect then dropped mid-write ("The chip stopped responding") at both 460800 and 115200. Reliable recipe: board on a direct laptop USB port; hold BOOT, tap EN, release BOOT; then `esptool --before no_reset --after hard_reset write_flash @flash_args` at 115200. Worked 2/2.
* Files: `deploy/esp32_kws/quantize_report_bcresnet-{2,3}.txt`, `quantize_report_bcresnet-8_dwconv.txt`, `device_output_bcresnet-{2,3}.txt`, `ppq_host_check_bcresnet-3.json`; the τ=2 model is `main/models/model.espdl`. The board currently holds the τ=3 firmware, which stops cleanly after the failed allocation.

## Planned next
1. Noise-robustness evaluation (`python -m kws.eval_noise runs/...`) for E1, E3, E4 at clean / 20 / 10 / 5 / 0 / −5 dB.
2. Seeds 1 and 2 for the best configurations.
3. v1_12 and v2_35 benchmarks for the final model(s) and BC-ResNet-8.
4. Small scales (τ = 1, 2, 3) vs BC-ResNet-1/2/3.
5. KD into BC-SSM + NASS if it helps the plain model.
6. Gate visualisation: g_t over spectrograms at different SNRs (qualitative figure).
