"""Pre-compute log-mel features for on-chip test-set evaluation.

The machine with the board attached has no dataset, so the features travel as a file.
float16 is used because its worst-case error here is 3.9e-03, far below the int8 input
step of 0.125 that the device quantizes to -- the device sees the same int8 either way.

Also stores the host's float32 prediction per clip, so scripts/eval_onchip.py can
report the int8-vs-float delta without needing torch on the laptop.

  python scripts/make_eval_features.py --limit 1200     # committed subset, ~8 MB
  python scripts/make_eval_features.py --limit 0        # full 4,890 clips, ~32 MB
"""
import argparse
import os

import numpy as np
import torch

from kws.data import LogMel
from kws.train import load_run

LABELS12 = ["_silence_", "_unknown_", "down", "go", "left", "no", "off", "on", "right", "stop",
            "up", "yes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v2_12_bcresnet-2")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--limit", type=int, default=1200, help="0 = the whole test set")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = torch.load(f"{a.cache}/gsc_v2_test12.pt")
    label = np.array([LABELS12.index(c) for c in d["class_name"]])
    idx = np.arange(len(label))
    if a.limit:
        # proportional per class, so accuracy on the subset estimates the full set
        rng = np.random.default_rng(a.seed)
        take = []
        for c in range(len(LABELS12)):
            pool = idx[label == c]
            n = max(1, int(round(a.limit * len(pool) / len(label))))
            take.append(rng.permutation(pool)[:n])
        idx = np.sort(np.concatenate(take))
    print(f"{len(idx)} clips of {len(label)}; class counts {np.bincount(label[idx], minlength=12).tolist()}")

    fe = LogMel()
    model = load_run(a.run, 12, torch.device("cpu"), ckpt="best.pt").eval()
    torch.set_num_threads(4)
    feats, preds = [], []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            batch = d["audio"][idx[i : i + 256]].float() / 32768.0
            f = fe(batch)
            feats.append(f[:, 0].numpy().astype(np.float16))
            preds.append(model(f).argmax(-1).numpy())
    feat = np.concatenate(feats)
    pred = np.concatenate(preds)
    acc = 100.0 * (pred == label[idx]).mean()
    print(f"host float32 accuracy on this set: {acc:.2f}%")

    out = a.out or f"deploy/onchip/eval_features{'' if not a.limit else f'_{len(idx)}'}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, feat=feat, label=label[idx], index=idx, host_float_pred=pred,
                        labels=np.array(LABELS12), run=a.run, host_float_acc=acc)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
