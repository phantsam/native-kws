# Handoff: the training side — the GPU server (as of 2026-11-24)

For whoever picks up the **training server** cold. The ESP32 work happens on a separate
Windows laptop and has its own brief: [`HANDOFF_DEVICE.md`](HANDOFF_DEVICE.md). Read this
file, then [`EXPERIMENTS.md`](EXPERIMENTS.md) (the chronological log of every result,
including the negative ones) and [`RESULTS.md`](RESULTS.md) (auto-generated tables).

Everything below was checked on this machine on 2026-11-24. Where something is not
verifiable from the repo, it says so.

---

## 1. The machine

| | |
|---|---|
| OS | Ubuntu 24.04.5 LTS, kernel 6.8.0-149-generic |
| CPU / RAM | 192 logical cores, 1.5 TB RAM |
| GPUs | **2 × NVIDIA H100 NVL, 94 GB each**, driver 580.196.07 |
| disk | `/home` on a 21 TB array, ~15 TB free; the repo and its caches use ~6 GB |

**The GPUs are shared with other users.** On 2026-11-24 four processes that are not ours
held ~13 GB on GPU 0. `jobqueue.py --min_free_gb` exists for exactly this reason (§4); do
not launch large waves without checking `nvidia-smi` first.

## 2. The repo

| | |
|---|---|
| path | **`/home/prjgnn/native-kws`**, branch `main`, remote `github.com/phantsam/native-kws` |
| not the repo | `/home/prjgnn/ML-KWS-for-MCU` is ARM's unrelated `ML-KWS-for-MCU`. Claude Code sessions have opened there by default — **`cd /home/prjgnn/native-kws` first, every time.** |
| Python | `.venv/`, **Python 3.12.3** |
| key packages | torch 2.11.0+cu128 (CUDA 12.8, both GPUs visible), torchaudio 2.11.0, numpy 2.5.3, onnx 1.23.2, onnxsim 0.8.1, onnxruntime 1.31.1, soundfile 0.14.0, tqdm, tensorboard 2.21.0 |

There is **no activation ritual** — every script, job file and doc calls the interpreter by
path:

```bash
cd /home/prjgnn/native-kws
.venv/bin/python -m kws.train --help
```

`source .venv/bin/activate` also works, but the `.venv/bin/python` form is what the queue
and the shell scripts use, so prefer it in anything you commit.

**To recreate the environment exactly**, from pinned versions (§10 has the full bootstrap
for a different server):

```bash
python3 -m venv .venv      # Python 3.12.x
.venv/bin/pip install -r requirements-train.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

`requirements-train.txt` holds the versions every number in `RESULTS.md` was produced
with. The device side pins its own, incompatible set in
`deploy/esp32_kws/requirements-host.txt` (Python 3.11 + esp-ppq, which holds back torch and
onnx) — **do not try to make one environment serve both machines.**

**esp-ppq is not installed here** (`import ppq` → `ModuleNotFoundError`), and neither is
ESP-IDF. `scripts/quantize_onchip.py` and `scripts/sim_onchip_int8.py` import `esp_ppq`
and therefore **only run on the laptop**. This machine's job ends at
`scripts/export_onchip.py`, which produces the ONNX + calibration data the laptop
quantizes.

Source layout:

| path | what |
|---|---|
| `kws/` | `data.py` (caches, log-mel, augmentation), `models.py` (all architectures), `train.py` (the only training entry point), `eval_noise.py`, `eval_protocols.py` |
| `scripts/` | `prepare_data.py`, `jobqueue.py`, `finalize_when_done.sh`, `collect_results.py`, `export_onchip.py`, `make_eval_features.py` + the three device scripts that need esp-ppq / a serial port |
| `jobs/` | queue job files: `main.txt`, `onchip.txt`, `causal.txt` |
| `runs/` | one directory per training run, plus `runs/logs/` |
| `deploy/onchip/` | **generated** exports for the device (§6) |
| `deploy/esp32_kws/` | the laptop's ESP-IDF project — do not edit it here |
| `rig/` | the INMP441 microphone rig, software-verified only; see [`HANDOFF_MIC.md`](HANDOFF_MIC.md) |

## 3. The dataset cache

`data/` and `cache/` are **gitignored** and exist only on this machine.

```
data/speech_commands_v0.01            2.0 G     data/speech_commands_test_set_v0.01    97 M
data/speech_commands_v0.02            3.3 G     data/speech_commands_test_set_v0.02   154 M
cache/gsc_v1.pt       2.0 G    cache/gsc_v1_test12.pt    94 M    cache/noise.pt   24 M
cache/gsc_v2.pt       3.2 G    cache/gsc_v2_test12.pt   149 M    (5.4 G total, built 2026-09-14)
```

The caches were built with, and are rebuilt by:

```bash
.venv/bin/python scripts/prepare_data.py          # --data data --out cache --workers 32
```

What that writes (`scripts/prepare_data.py`):
* `gsc_v{1,2}.pt` — every clip of the main archive as **int16 padded/truncated to exactly
  16,000 samples**, with `label` (word index), `split` (0 train / 1 valid / 2 test, taken
  from the archive's own `validation_list.txt` and `testing_list.txt`), `words`, `paths`
  and `speakers` (the part of the filename before `_nohash_`, which is what the leakage
  audit in E19 used).
* `gsc_v{1,2}_test12.pt` — the **officially released 12-class test set**, including its
  `_silence_` and `_unknown_` directories, with `class_name` per clip. This is the
  `v1_12` / `v2_12` test set and the one the ESP32 is evaluated on.
* `noise.pt` — the six `_background_noise_` recordings as float32, used by the
  augmentation and by `kws/eval_noise.py`.

**How `data/` itself was downloaded is not recorded in the repo.** The archives are the
standard Google releases (`speech_commands_v0.01.tar.gz`, `v0.02.tar.gz` and the two
`speech_commands_test_set_*.tar.gz`) extracted into those four directories; if you ever
need to re-fetch them, note the exact URLs in a new `EXPERIMENTS.md` entry so the next
person does not have to guess either.

## 4. How jobs run

### Job files

Each non-empty, non-`#` line of a `jobs/*.txt` file is `<run_name> <kws.train args…>`;
`--name <run_name>` is appended automatically. Priority is top-first. The file is re-read
on every poll, so you can append jobs to a running queue.

```
v2_12_bcresnet-2 --task v2_12 --model bcresnet-2 --opt sgd --lr 0.1 --wd 1e-3 --bs 100 --smoothing 0
```

### The queue

```bash
.venv/bin/python scripts/jobqueue.py jobs/onchip.txt --slots 0:4 1:4 --min_free_gb 32
```

`scripts/jobqueue.py` skips a job if `runs/<name>/summary.json` exists or a process with
that `--name` is already alive (it reads `/proc`, so runs started outside the queue count
towards the per-GPU slot limit too), waits for any `--teacher` run to have written
`last_ema.pt`, launches only on a GPU where `nvidia-smi` reports at least `--min_free_gb`
free (default 32, reserved per launch within a poll), and always passes `--resume` so an
interrupted run continues from `runs/<name>/state.pt`.

### As a systemd unit

Long waves run as **user** units so they outlive the shell (E17/E18):

```bash
systemd-run --user --unit=kws-queue --property=KillMode=process \
  /home/prjgnn/native-kws/.venv/bin/python /home/prjgnn/native-kws/scripts/jobqueue.py \
  /home/prjgnn/native-kws/jobs/<file>.txt --slots 0:4 --min_free_gb 20
```

`KillMode=process` is the important part: stopping the unit must not kill the trainers.
Units used so far: `kws-queue`, `kws-queue3`, `kws-finalize`, `kws-queue-causal` — all
finished and gone. The exact flags of past invocations are not all recorded; `runs/logs/queue.log`
preserves the jobs file and the slot map of each start, but not `--min_free_gb`.

**Caveat carried from E17:** user lingering is off, so if *every* login session for this
user ends, the user manager — and the queue with it — stops. Trainers already launched
survive, but nothing new starts.

### Watching it

```bash
systemctl --user status kws-queue          # or whichever unit
journalctl --user -u kws-queue
tail -f runs/logs/queue.log                # launches, teacher blocks, exits
tail -f runs/logs/<run_name>.log           # one run's stdout
pgrep -af kws.train                        # what is actually alive
nvidia-smi
```

**Nothing is running as of 2026-11-24** — no `kws.train` process, no queue unit.

### What a run writes

`runs/<name>/`: `args.json` (every hyperparameter, including `causal`), `log.jsonl` (one
line per epoch: loss, lr, seconds, and every fifth epoch the valid/test accuracy for raw
and EMA weights), `best.pt` (best validation), `last.pt`, `last_ema.pt`, `state.pt`
(optimizer + epoch, for `--resume`), `summary.json` (written on completion — its presence
is what marks a run "done"), and after finalization `noise_eval.json` and
`protocol_eval.json`. Per-run stdout goes to `runs/logs/<name>.log`.

### Finalizing

```bash
scripts/finalize_when_done.sh 0 v2_12_bcresnet-1 v2_12_bcresnet-2
```

Waits for each `summary.json`, then runs `kws.eval_noise` and `kws.eval_protocols` on that
run and regenerates `docs/RESULTS.md` via `scripts/collect_results.py`. It also notices
crashed and stopped runs instead of waiting forever. `RESULTS.md` is **generated** — never
hand-edit it; run `collect_results.py` after any new run finishes.

## 5. Which checkpoints back which number

### README.md headline table

`README.md` reports "mean ± std over 3 seeds, final epoch". That column is
**`last_test_raw`** from each run's `summary.json` — verified: the three
`v2_12_native-H-nass-kdens{,-s1,-s2}` values 98.98 / 99.00 / 99.08 give 99.02 ± 0.05, and
BC-ResNet-8's 98.79 / 98.51 / 98.67 give 98.66 ± 0.14, both matching the README exactly.
Use `last_test_raw`, not `best_by_valid`, if you ever recompute these.

| README row | runs behind it |
|---|---|
| V2-12 ours, 99.02 ± 0.05 | `v2_12_native-H-nass-kdens`, `-s1`, `-s2` |
| V1-12 ours, 98.13 ± 0.04 | `v1_12_native-H-nass-kdens`, `-s1`, `-s2` |
| V2-35 ours, 98.07 ± 0.09 | `v2_35_native-H-nass-kdens`, `-s1`, `-s2` |
| BC-ResNet-8 same pipeline | `v{1_12,2_12,2_35}_bcresnet-8`, `-s1`, `-s2` |

Each student's teachers are the three runs named in its `args.json` `teacher` list
(`<task>_bcresnet-8` + `<task>_native-L` + `<task>_native-M` of the **same** task).
Deleting a teacher's directory does not invalidate the student, but it does make the
result unreproducible.

`deploy/v2_12_native-H-nass-kdens/best.pt` is **force-added to git** (the only `*.pt` in
the repository, overriding `.gitignore`) and is byte-identical to
`runs/v2_12_native-H-nass-kdens/best.pt`, with identical `args.json`. That is the seed-0
final model, committed so the laptop needs no dataset and no GPU.

### The ESP32 line of work

| run | role |
|---|---|
| `v2_12_bcresnet-{1,2,3}` | the small scales trained because τ=8 does not fit the chip (E28). **τ=2 is the deployed model** |
| `v2_12_bcresnet-8` | the reference that validates the pipeline against the BC-ResNet paper |
| `v2_12_bcresnet-{1,2,3,8}-causal` | the streaming prerequisite; the look-ahead cost measured in E33 |

### Superseded or exploratory — keep for the record, do not report

* `v2_12_native-H`, `-nass`, `-kd`, `-kdens`, `-kdT4` — the ablation ladder that selected
  the final configuration **on validation accuracy** (E17). `-kdT4` has a *higher* test
  number (98.96) than the chosen configuration's seed 0 (98.90) but a lower validation
  mean; selecting on test would have been the mistake the protocol exists to prevent.
* `v2_12_bcssm-7*` — BC-SSM, the explored alternative. No clean-accuracy gain; kept
  because its NASS ablations (`-nass`, `-nass-unsup`, `-nass-causal`) are cited in the docs.
* `v2_12_native-M-kd` (stopped at epoch 33) and `v2_12_native-H-kdT4-e300` (stopped at 0)
  are dead runs. `RESULTS.md` shows them as `stopped`.
* `v1_12_native-H`, `v2_35_native-H`, and the `native-L` / `native-M` runs of each task are
  teacher members and baselines, not results in their own right.

## 6. The device export pipeline

This machine's deliverable to the laptop is `deploy/onchip/`. **It is generated. Never
edit anything in it by hand.**

### `scripts/export_onchip.py`

```bash
.venv/bin/python scripts/export_onchip.py --run runs/v2_12_bcresnet-2 [--out <dir>] [--ckpt best.pt] [--no_acc]
```

Writes `deploy/onchip/<model>/` (default: named after the run's model) containing
`<dir basename>.onnx`, `calib_128.npy`, `test_vectors.{h,npz}` and `export_info.json`. The
ONNX file is named after the **output directory**, so `--out deploy/onchip/bcresnet-2-causal`
yields `bcresnet-2-causal.onnx`.

What it does to the graph, and why each step exists:

1. **Exports at opset 13, batch 1**, input `logmel` (1,1,40,101) float32 — 40 mel bands,
   30 ms window, 10 ms hop.
2. **Removes every `BatchNormalization`.** ESP-DL has no such operator. onnxsim folds the
   ones sitting behind a Conv; the **12 that survive** (SubSpectralNorm normalises a
   *reshaped* tensor, so its BN is not directly behind a Conv) are rewritten as
   **depthwise `Conv` with kernel 1** carrying the same affine. They were first written as
   `Mul`+`Add`, which the device rejected on ESP-DL's int8 `Mul` exponent rule
   `out_exp − in0_exp − in1_exp ≥ 0` — and esp-ppq's PC simulation had shown nothing
   (E26/E27). Do not revert that.
3. **Cross-layer equalization of the folds (E30).** ESP32 and ESP32-S3 quantize **per
   tensor**, so one power-of-two exponent must cover a fold whose channel scales span up to
   7,008×, and the small channels quantize to zero. The reshape makes Conv channel `k` map
   onto fold channels `k·S … k·S+S−1`, so scaling the Conv's output channel by `g[k]` and
   dividing those `S` fold channels by `g[k]` is an **exact identity** — not an
   approximation. `g` is the geometric mean over sub-bands, damped by a per-layer `alpha`
   chosen from {0.25, 0.5, 0.75, 1.0} to minimise destroyed channels across the pair
   (blanket equalization made some layers worse). Safety gate: a Conv is only rescaled if
   it has exactly one consumer, so a residual branch is never touched.
4. **Checks every operator against the ESP-DL whitelist** and prints what is missing.
5. **Prints the largest single activation**, which is what has to fit the arena — so you
   know whether a model can run before anything is flashed.
6. Verifies the folded graph against PyTorch and runs 4 real clips (`yes`, `no`, `stop`,
   `_silence_`) through it, storing the host's float logits as the device's test vectors.

### The exact commands behind the committed exports

All four directories were regenerated after E30, on 2026-10-28 at 15:19. Their
`export_info.json` has `"ssn_equalized": true` and `"float_test_accuracy_v2_12": -1.0`,
i.e. `--no_acc` (the float accuracies live in `RESULTS.md` instead):

```bash
for t in 1 2 3 8; do
  .venv/bin/python scripts/export_onchip.py --run runs/v2_12_bcresnet-$t --no_acc
done
```

| directory | params | largest activation | destroyed channels, before → after equalization |
|---|---|---|---|
| `deploy/onchip/bcresnet-1/` | 9,232 | 31.6 KB | 87 → 40 |
| `deploy/onchip/bcresnet-2/` | 27,284 | 63.1 KB | 249 → 74 |
| `deploy/onchip/bcresnet-3/` | 54,168 | 94.7 KB | 324 → 56 |
| `deploy/onchip/bcresnet-8/` | 321,068 | 252.5 KB | 676 → 51 |

### `scripts/make_eval_features.py`

The laptop has no dataset, so the on-chip test set travels as a file of pre-computed
features. Written on 2026-10-28 at 15:34–15:35 by:

```bash
.venv/bin/python scripts/make_eval_features.py --limit 1200   # -> deploy/onchip/eval_features_1200.npz  (8 MB)
.venv/bin/python scripts/make_eval_features.py --limit 0      # -> deploy/onchip/eval_features.npz       (32 MB)
```

Defaults: `--run runs/v2_12_bcresnet-2`, `--cache cache`, `--seed 0`, deterministic. Each
file holds `feat` (float16 log-mels — worst-case error 3.9e-03, far below the device's int8
input step of 0.125, so the chip sees identical int8 either way), `label`, `index` into the
cached test set, **`host_float_pred`** (so the device report is a paired float-vs-int8
comparison with no torch on the laptop), `labels` and `host_float_acc`. `--limit` takes a
**proportional per-class** sample, so subset accuracy estimates the full set.

`deploy/onchip/eval_firmware/app_eval.cpp` is the reference evaluation server written here.
The laptop could not run it as-is — its static 16 KB feature buffer left an 88 KB largest
free block for a 98.64 KB arena — so the laptop keeps its own corrected copy under
`deploy/esp32_kws/main/`. **If you change the protocol, change both.**

## 7. State of play, and what to do next

Measured, on silicon, by the laptop: **BC-ResNet-2 (27,284 parameters) int8 on a plain
PSRAM-less ESP32 scores 97.06 % on the full 4,890-clip GSC v2-12 test set**, against
98.02 % float on the same clips — 0.96 points of int8 cost, 951.6 ms per 1-second window,
98.64 KB arena, weights in flash (E32). The chip is **bit-exact with esp-ppq's host
simulation on 4,890/4,890 clips**, which is why most remaining quantization work needs no
board.

Measured here: the **cost of causality is 0.37 / 0.35 / 0.18 / 0.08 points** at
τ = 1 / 2 / 3 / 8 (E33), so the streaming deployment target is causal BC-ResNet-2 at
**97.67 %** float, still above the published non-causal BC-ResNet-2 (97.4).

Nothing is half-run: no training job is active, the working tree is clean, and
`RESULTS.md` was regenerated after the causal runs so it matches E33.

### Ranked next steps

1. **Export causal BC-ResNet-2 so the laptop can measure its int8 cost** (E33's open
   number, and `HANDOFF_DEVICE.md` §6.2 asks for it). **Verified working on this machine
   on 2026-11-24** against a scratch directory — no code change needed; `--causal` is read
   from the run's `args.json`, the `ZeroPad2d` left-padding exports as `Pad`+`Slice`+
   `Concat` nodes which onnxsim folds away, the operator set comes out identical to the
   non-causal export with nothing off the whitelist, equalization reports 241 → 49 destroyed
   channels, and all 4 test vectors predict correctly in float:
   ```bash
   .venv/bin/python scripts/export_onchip.py --run runs/v2_12_bcresnet-2-causal \
       --out deploy/onchip/bcresnet-2-causal --no_acc
   ```
   Commit the directory and tell the laptop; it then runs `sim_onchip_int8.py` (4 min) and
   only afterwards the board (92 min).
2. **The `go`-class int8 loss may be ours to fix.** Of the 0.96 points, `go` alone accounts
   for 41 of the 47 net-lost clips (98.3 % → 88.1 %), and the damage is input-dependent —
   int8 lowers the `go` logit by 5.47 on `go` clips but 0.78 on everything else — so some
   layer's quantization is destroying the evidence for that one word. Localizing it is a
   **host-only** job and belongs on the laptop (it has esp-ppq and the simulation is exact).
   But the *fix* may well belong here: the suspects are the 74 channels equalization still
   counts as destroyed in τ=2 and `bnfold_conv_11`, the one layer E30's equalization made
   worse (0.46 % → 1.56 %). Candidate export-side responses, none tried: a tighter
   `alpha` search, equalizing against a quantization-error objective rather than a
   destroyed-channel count, or limiting the SubSpectralNorm affine's channel spread during
   training. **Wait for the localization before changing the export** — the standing rule is
   report first, work around second.
3. **The Transpose tax is a model-design change, and it is ours.** 151 ms of the 952 ms
   inference — 15.8 %, across 61 operators — is esp-ppq inserting layout transposes around
   every SubSpectralNorm `Reshape`. No quantization setting removes it; it needs a
   front-end that normalises sub-bands without a reshape (e.g. grouped normalisation over a
   channel-folded layout, or folding the sub-band split into the Conv's channel structure).
   This is the single largest latency win available and it has not been attempted.
4. **Streaming execution (milestone 3 of `ONCHIP.md`).** The memory analysis is done
   (τ=2 streams in 5.9 KB against 63.1 KB whole-window; τ=8 in 23.3 KB against a 391 KB
   arena the chip cannot allocate at all) and the causal models now exist. What does not
   exist is the per-frame step graph: ESP-DL executes whole graphs, so streaming needs the
   layer histories as explicit state inputs and outputs. Designing and exporting that graph
   is a training-side job; measuring its latency is a device-side one. **The per-frame
   latency figures in `ONCHIP.md` are extrapolations from whole-window throughput, not
   measurements** — they are labelled as such, keep them labelled.
5. **τ=1 has never been deployed.** It is exported and it fits comfortably; it is the only
   scale whose estimated streaming latency (~3.2 ms/frame) leaves room for the log-mel
   front-end inside a 10 ms budget.
6. **Items 1–6 of "Planned next" in `EXPERIMENTS.md`** (noise evaluation for E1/E3/E4,
   extra seeds, v1_12 and v2_35 for the final models, KD into BC-SSM+NASS, the gate
   visualisation) predate the ESP32 work and are still open, but none of them is on the
   critical path of the deployment story.
7. **The live microphone** is deferred and has never touched hardware. The rig is written
   and software-verified; see [`HANDOFF_MIC.md`](HANDOFF_MIC.md) and `DATASET.md`.

## 8. Rules

* **`deploy/onchip/*` is generated.** Regenerate with `scripts/export_onchip.py` /
  `scripts/make_eval_features.py`; never hand-edit. The laptop is under instruction not to
  touch it either, and esp-ppq rewrites any ONNX it is handed in place, which is why
  `quantize_onchip.py` copies to a temp file first.
* **`docs/RESULTS.md` is generated** by `scripts/collect_results.py`. Run it, don't edit it.
* **`docs/EXPERIMENTS.md` numbering is shared with the laptop.** Entries are append-only
  and numbered in one sequence across both machines; E26, E29 and E32 came from the device,
  the rest from here. **The next free number is E34.** Record negative results — most of the
  useful entries are negative ones. When a merge interleaves entries, put them back in
  numeric order (the device side has had to do this twice).
* **Never tune anything against recorded microphone audio.** `docs/DATASET.md`
  pre-registers the recorded set as a test set; a model, augmentation or channel simulator
  fitted to it invalidates every number it produces.
* **Selection is on validation accuracy only.** Reporting the best test number across a
  ladder of configurations is the failure mode E17 and E19 exist to prevent.
* **No datasets, no audio, no large binaries in git.** `data/`, `cache/` and `rig/out/` are
  gitignored; the only committed checkpoint is the force-added 1.2 MB
  `deploy/v2_12_native-H-nass-kdens/best.pt`. The biggest tracked file is
  `deploy/onchip/eval_features.npz` at 32 MB; keep new binaries well under that.
* **Commit and push only when the owner asks**, and never rebase or force-push a branch the
  other machine is working on. `main` is the integration branch; device work arrives as a
  fast-forward from `e26-esp32-bringup`. Commit messages: a one-line summary of the
  *result*, not the activity, then the detail.

## 9. The second board

A second board has reached the laptop — reported as "an ESP32 with a camera", not yet
plugged in or identified. `HANDOFF_DEVICE.md` §7 has the identification procedure and the
firmware consequences. **Nothing needs doing here until the chip is known**, and nothing
about the existing results depends on it. When the laptop reports what it is:

* **If it is an ESP32-S3:** no export change. The S3 is **per-tensor too**, so E30's
  equalization is still required — this was a correction to an earlier wrong claim of mine,
  recorded in E29/E31. Only the laptop's esp-ppq `--target` changes (`esp32s3`). What the
  S3 buys is speed (Espressif's own ESP-NN figure is ~7× on a comparable int8 model, *not*
  measured for this graph) and, with PSRAM, the ability to run τ=3 and τ=8 whole-window —
  which would give measured silicon accuracy for two scales board 1 physically cannot hold.
  That is worth exporting nothing new for: `deploy/onchip/bcresnet-{3,8}/` already exist.
* **If it is an ESP32-P4:** it quantizes **per channel**, so the fold spread stops
  mattering and an un-equalized export would also work. Keep equalization on anyway — it is
  an exact identity, so it costs nothing, and it keeps one export serving every target.
* **Do not retire board 1.** E31's streaming argument depends on its 108 KB contiguous
  limit binding; demonstrating streaming on a board with PSRAM proves nothing was
  constrained. The new board is a second platform, not a replacement.
* **For the live-mic phase** an on-board microphone would change the capture path: likely
  different pins and possibly PDM rather than I2S, so `rig/firmware/i2s_stream` would not
  apply unchanged. Nothing to do until the mic phase starts.

## 10. Rebuilding this setup on a different server

Nothing here is tied to this host. In order:

```bash
# 1. the repo
git clone https://github.com/phantsam/native-kws && cd native-kws

# 2. the environment (Python 3.12.x; 3.11 also works, 3.13 untested here)
python3 -m venv .venv
.venv/bin/pip install -r requirements-train.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# 3. the dataset: four public Google archives, extracted to these exact directory names
mkdir -p data && cd data
#   speech_commands_v0.01.tar.gz          -> data/speech_commands_v0.01/
#   speech_commands_v0.02.tar.gz          -> data/speech_commands_v0.02/
#   speech_commands_test_set_v0.01.tar.gz -> data/speech_commands_test_set_v0.01/
#   speech_commands_test_set_v0.02.tar.gz -> data/speech_commands_test_set_v0.02/
cd ..

# 4. the caches (deterministic; ~5.4 GB out)
.venv/bin/python scripts/prepare_data.py

# 5. sanity-check against a known number before trusting anything
.venv/bin/python -m kws.train --task v2_12 --model bcresnet-2 \
    --opt sgd --lr 0.1 --wd 1e-3 --bs 100 --smoothing 0 --name check_bcresnet-2
#   expect ~98.0 on v2_12 (E28 got 98.02 best-by-valid); ~15 s/epoch on an H100
```

Hardware requirements are modest for everything except the full waves: one CUDA GPU is
enough, and the BC-ResNet scales train in 15–17 s/epoch (≈1 hour for 200 epochs) on an
H100. The 2-GPU, 12-slot waves in `jobs/main.txt` were sized for this machine; set
`--slots` to what the new one actually has.

A CPU-only machine is fine for `scripts/export_onchip.py`, `make_eval_features.py` and the
whole of `rig/` — install with the `cpu` index instead and drop the `+cu128` suffixes.

**What would *not* transfer:** exact checkpoint reproduction. `kws/train.py` seeds
`torch.manual_seed(args.seed)` and the data generators, but sets
`cudnn.benchmark = True` and does not call `use_deterministic_algorithms`, so a retrained
run lands within noise of the recorded number rather than on it exactly. That is the whole
argument for §11.

## 11. What exists only on this server — the backup list

`git ls-files` covers everything in the repo except model weights. Concretely:

| what | size | in git? | if lost |
|---|---|---|---|
| **`runs/**/*.pt`** — 161 files: `best.pt`, `last.pt`, `last_ema.pt`, `state.pt` per run | **265 MB** | **no** (`.gitignore`: `*.pt`) | **irreplaceable.** Retraining gives a run within noise, not the same weights, so every exported ONNX and every number tied to a specific checkpoint would shift. **Back this up.** |
| `cache/*.pt` — 5 tensor caches | 5.4 GB | no | rebuild: `scripts/prepare_data.py`, deterministic, needs `data/` |
| `data/` — 4 extracted archives | 5.5 GB | no | re-download the public archives (§10 step 3) |
| `.venv/` | ~7 GB | no | rebuild from `requirements-train.txt` |
| `runs/<name>/{args.json,log.jsonl,summary.json,noise_eval.json,protocol_eval.json}` | small | **yes**, all 259 files | nothing to do |
| `runs/logs/*.log`, `runs/logs/queue.log` | small | **yes** | nothing to do |
| `deploy/onchip/*` exports, `deploy/v2_12_native-H-nass-kdens/best.pt` | 39 MB | **yes** (the `.pt` is force-added) | nothing to do |
| `docs/`, `kws/`, `scripts/`, `jobs/`, `rig/` | small | **yes** | nothing to do |

**So the backup is one command:**

```bash
cd /home/prjgnn/native-kws
find runs -name '*.pt' -print0 | tar --null -czf ~/native-kws-checkpoints-$(date +%F).tgz -T -
# 161 files, ~265 MB; everything else is in git or regenerable
```

If you want to be able to *resume* interrupted training rather than just evaluate, keep
`state.pt` too — it is included above. If you only care about evaluating and exporting,
`best.pt` + `last_ema.pt` per run is enough and roughly halves the size.

Nothing on this machine holds credentials for the project: there is no token, key or
`.netrc` in the repo, and pushes use whatever git credential helper the login session
provides.
