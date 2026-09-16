# Architecture

This document describes the models in this repository, the reasoning behind each design decision, and what is (and is not) new. Code: `kws/models.py`.

**Contents**
1. [Input features](#1-input-features)
2. [Reference: BC-ResNet](#2-reference-bc-resnet)
3. [First attempt: NativeKWS (stacked conv → Mamba)](#3-first-attempt-nativekws-stacked-conv--mamba)
4. [Idea 1: Broadcasted State-Space block (BC-SSM)](#4-idea-1-broadcasted-state-space-block-bc-ssm)
5. [Idea 2: Noise-Aware Selective Scan (NASS)](#5-idea-2-noise-aware-selective-scan-nass)
6. [Streaming / causal mode](#6-streaming--causal-mode)
7. [Model sizes](#7-model-sizes)
8. [Novelty: honest assessment](#8-novelty-honest-assessment)
9. [Deployment outlook (ESP32-S3)](#9-deployment-outlook-esp32-s3)

---

## 1. Input features

Identical for every model, so comparisons are fair (`kws/data.py:LogMel`):

| Setting | Value |
|---|---|
| Sample rate | 16 kHz, 1 s clips (zero-padded) |
| Window / hop / FFT | 30 ms (480) / 10 ms (160) / 512 |
| Mel bands | 40, natural log of power + 1e-6 |
| Shape fed to the network | `(B, 1, F=40, T=101)` |

This matches BC-ResNet's official code.

---

## 2. Reference: BC-ResNet

*Kim et al., "Broadcasted Residual Learning for Efficient Keyword Spotting", Interspeech 2021.* Ported faithfully in `BCResNet`; the parameter counts match the paper exactly (τ=1: 9,232; τ=2: 27,284; τ=3: 54,168; τ=8: 321,068). Our reproduction of BC-ResNet-8 reaches **98.83 %** on GSC v2-12 (paper: 98.7 %), which validates the training pipeline.

**Topology** (base width `c0 = 8τ`): stem Conv5×5 (stride 2 in frequency) → 4 stages of BC blocks with `[2, 2, 4, 4]` blocks, channels `[c0, 1.5c0, 2c0, 2.5c0]`, frequency stride in stages 1 and 2 (F: 20 → 10 → 5), temporal dilation `2^stage` → classifier (depthwise 5×5 over the remaining 5 bands, 1×1 conv to `4c0`, global average, 1×1 conv to classes).

**BC block**, input `x ∈ R^{C×F×T}`:

```
f2  = SubSpectralNorm( DWConv_{3×1}(x) )          # 2-D, frequency-wise
z   = mean_F(f2)                                  # R^{C×1×T}   (frequency pooling)
f1  = Dropout( Conv_{1×1}( SiLU(BN( DWConv_{1×3, dilated}(z) )) ) )   # 1-D, temporal
out = ReLU( x + f2 + f1 )                         # f1 is broadcast back over F
```

The key insight is that most computation happens on the frequency-pooled 1-D signal and is broadcast back to 2-D. The limitation we target: **the temporal operator is a 3-tap (dilated) convolution**, so each block's temporal receptive field is small (3 frames × dilation 1, 2, 4, 8 = 30–240 ms).

---

## 3. First attempt: NativeKWS (stacked conv → Mamba)

`NativeKWS` (presets S / M / L / H in `kws/train.py`): a thin BC-block front-end, frequency collapse, then *L* bidirectional Mamba blocks (pre-RMSNorm, gated, drop-path) on the 101-frame sequence, mean pooling, linear head.

**Outcome.** M (278 K): 98.47 %, below BC-ResNet-8 (98.83 %) at a similar size. The conv front-end was too thin: most parameters went into the Mamba blocks. The H preset rebalances this (BC-ResNet-depth backbone + 2 light Mamba blocks, 296 K), and with distillation from BC-ResNet-8 it is competitive (see RESULTS.md).

**Causality caveat.** For `NativeKWS`, `--causal` only drops the backward Mamba branch; the conv front-end keeps centred padding (~170 ms look-ahead). Only `BCSSMNet --causal` is fully streaming (Section 6).

**NASS variant.** `NativeKWS(noise_aware=True)` adds the speech-presence gate of Section 5 to both Mamba blocks (+5.6 K params).

**Why it is not the main contribution.** "CNN front-end, then a sequence model" is a standard pattern; it is kept as a baseline and ablation.

---

## 4. Idea 1: Broadcasted State-Space block (BC-SSM)

`BCSSMBlock`, `BCSSMNet`.

### Change
Keep the BC block exactly, but **replace the 1-D temporal convolution in the broadcast path with a selective state-space model** (Mamba-style, S6):

```
f2  = SubSpectralNorm( DWConv_{3×1}(x) )
z   = mean_F(f2)                                   # (B, C, 1, T) → sequence (B, T, C)
s   = SSM_fwd(z) + flip(SSM_bwd(flip(z)))          # causal mode: SSM_fwd only
f1  = Dropout( Conv_{1×1}( SiLU(BN(s)) ) )
out = ReLU( x + f2 + f1 )
```

### The selective SSM (per direction; `SSMBranch`)
For the pooled sequence `u_t ∈ R^C` (one SSM per channel, state size N):

```
v_t        = SiLU( CausalDWConv_{k=4}(u)_t )
Δ_t        = softplus( W_Δ · W_r v_t + b_Δ )      ∈ R^C      (low-rank, r = ⌈C/16⌉)
B_t, C_t   = W_B v_t,  W_C v_t                    ∈ R^N
A          = −exp(A_log)                          ∈ R^{C×N}  (fixed per channel, learned)
h_t        = exp(Δ_t A) ⊙ h_{t−1} + (Δ_t B_t) v_t        (diagonal recurrence)
y_t        = ⟨h_t, C_t⟩ + D ⊙ v_t
```

Training uses an exact chunked log-space parallel scan (`selective_scan`), verified against the sequential recurrence (max abs error ≈ 1e-6).

### Why
* **Every block sees the whole utterance.** In BC-ResNet only the stacked dilations give long context; here each BC-SSM block has an unbounded (input-selected) temporal receptive field at 1-D cost.
* **Stays in the broadcast regime.** The SSM runs on `C` channels × `T` frames after frequency pooling, never on `C × F × T`, so the cost scales like BC-ResNet's 1-D path, not like a 2-D sequence model.
* **The whole network becomes a recurrence.** In causal mode every temporal operator is either a causal conv or a diagonal recurrence, so the model can stream frame-by-frame with constant memory (Section 6).

### Controlled comparison
With `ssm_stages=()` a `BCSSMNet` **is** BC-ResNet (identical parameter count 321,068, verified). By default the SSM replaces the temporal conv in stages 2 and 3 (the eight 4-block-stage blocks); stages 0–1 keep convolutions, where local patterns dominate.

---

## 5. Idea 2: Noise-Aware Selective Scan (NASS)

`SpeechPresenceGate`, `SSMBranch(noise_aware=True)`, targets in `kws/data.py:speech_presence_targets`.

### Intuition
In a selective SSM, the step size Δ_t controls **how much frame t is written into memory**:
* Δ_t → 0: `exp(Δ_t A) → 1` and `Δ_t B_t → 0`, i.e. the state is **kept unchanged**, and the frame is ignored.
* Large Δ_t: the state is overwritten by the current frame.

So Δ is exactly the knob that should say "this frame is just noise, don't store it". Standard Mamba learns this implicitly from the classification loss alone. NASS makes it **explicit and supervised**.

### Mechanism
1. **Speech-presence gate** (shared by all SSM blocks, ~5 K params), computed from the log-mel input:
   `BN1d(40) → Conv1d(40→24, k=5) → SiLU → DWConv1d(k=3) → SiLU → Conv1d(24→1)` → logits `ℓ_t`, `g_t = σ(ℓ_t)`.
   Convolutions are centred (offline) or left-padded (causal).
2. **Gated step size** in every SSM branch, with a learned per-channel strength `w_c = σ(ω_c)` (initialised to 0.5):

   `Δ'_{t,c} = Δ_{t,c} · (1 − w_c (1 − g_t))`

   Channels can learn `w_c → 1` (write only speech frames) or `w_c → 0` (ignore the gate, e.g. to track the noise floor).

### Free frame-level supervision
During augmentation we add the noise ourselves, so for every training clip we know the clean signal `s` and the added noise `n` separately. Per log-mel frame (same 480/160 framing):

```
E_s, E_n = mean-square energy of s, n in the frame
snr      = 10 log10(E_s / E_n)
rel      = 10 log10(E_s / max_t E_s)          (distance from the loudest frame)
abs      = 10 log10(E_s)                      (dBFS)
target_t = σ(snr/3) · σ((rel+25)/3) · σ((abs+65)/3)
```

That is: speech-dominant, not a pause within the word, and not digital silence. Frames removed by SpecAugment's time masks get target 0; silence-class clips are all-zero clean signals, so their target is 0 everywhere. On a real batch: keyword clips ≈ 39 % speech frames, silence clips 0 %.

**Loss:** `L = CE(logits, y) + λ · BCE(ℓ, target)`, λ = 0.5 by default. `λ = 0` gives the *unsupervised gate* ablation (same architecture, gate learned only through the classification loss).

### Why it matters for the deployment goal
* The model gets an explicit, interpretable **"is this speech?" signal per frame**, the same idea as a voice-activity detector, but learned jointly and used *inside* the memory update rather than as a separate hard switch.
* On the MCU the gate costs one tiny conv stack per frame, and `g_t` can also be read out directly (for example, to skip the rest of the network during long stretches of noise).

---

## 6. Streaming / causal mode

`--causal` makes every temporal operator non-anticipative:

| Component | Offline | Causal |
|---|---|---|
| Stem Conv5×5, BC temporal DWConv 1×3, head DWConv 5×5 | centred padding | left padding only |
| SSM | forward + backward branch | forward branch only |
| Speech-presence gate convs | centred | left-padded |

Verified: changing input frames `t ≥ 60` leaves every hidden activation and gate value at `t < 60` bit-identical.

At inference the causal model is a set of FIFO buffers (conv receptive fields) plus one `N`-dimensional state per SSM channel. Each new 10 ms frame costs O(1) work, with no recomputation of the 1 s window. Global average pooling is replaced at deployment by a running/windowed average (future work).

---

## 7. Model sizes

| Model | Params | Notes |
|---|---|---|
| BC-ResNet-8 | 321,068 | reference |
| BC-SSM-7 (τ=7, SSM in stages 2–3, N=8) | 342,200 | idea 1; +6.6 % params vs BC-ResNet-8 |
| BC-SSM-7 + NASS | 349,241 | ideas 1+2; gate 5,025 + per-channel strengths 2,016 |
| BC-SSM-7 + NASS, causal | 300,633 | no backward SSM branches |
| NativeKWS-M / M-causal / H / L | 277,644 / 239,052 / 295,572 / 615,292 | first attempt (Section 3) |

Small scales (τ = 1, 2, 3) are planned; at τ = 1 the SSM + gate overhead is proportionally large (24 K vs 9 K), so small models may need N = 4 and a smaller gate.

---

## 8. Novelty: honest assessment

| Component | Status |
|---|---|
| Broadcast residual blocks, SubSpectralNorm | Existing (BC-ResNet, 2021) |
| Selective SSM (Mamba) | Existing (Gu & Dao, 2023); first applied to KWS by Keyword Mamba (2025), as a large (3.4–5.2 M) pure-Mamba model on patch tokens |
| **SSM as the temporal operator *inside* the broadcast-residual path (BC-SSM)** | Not found in our literature search (Sept 2026). New block design |
| **Supervised, noise-conditioned step size Δ via a speech-presence gate with free SNR labels (NASS)** | Not found in our literature search. New mechanism |
| Fully causal, recurrence-only conv+SSM KWS targeted at an MCU | Not found as a combination |

A full literature review is still needed before claiming "first" in a paper.

---

## 9. Deployment outlook (ESP32-S3)

Known from the ESP-DL operator table (checked Sept 2026):
* Supported int8 on ESP32-S3: Conv (1-D/2-D, depthwise), Add, Mul, AvgPool/GlobalAvgPool, ReLU/Clip/HardSwish/Sigmoid, Exp, Softmax (float32 output), RMSNorm, LayerNorm.
* **No** GELU and **no** scan/cumsum operator. The selective scan will need a small custom C kernel (per-frame recurrence: exp, multiply, add), a good fit for streaming anyway.
* BatchNorm folds into convs; SubSpectralNorm becomes a per-sub-band Mul + Add.
* Planned: integer-friendly SSM (restrict `exp(ΔA)` to a lookup table), and SiLU → HardSwish before quantization-aware training.
