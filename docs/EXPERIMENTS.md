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

---

## Planned next
1. Noise-robustness evaluation (`python -m kws.eval_noise runs/...`) for E1, E3, E4 at clean / 20 / 10 / 5 / 0 / −5 dB.
2. Seeds 1 and 2 for the best configurations.
3. v1_12 and v2_35 benchmarks for the final model(s) and BC-ResNet-8.
4. Small scales (τ = 1, 2, 3) vs BC-ResNet-1/2/3.
5. KD into BC-SSM + NASS if it helps the plain model.
6. Gate visualisation: g_t over spectrograms at different SNRs (qualitative figure).
