"""Evaluate trained runs on a recorded (replayed) test set.

Reports the paired comparison that the replay protocol is for: the same model, the
same clips, the same labels, scored on the cached originals and on their
recordings through the INMP441. The difference is the microphone channel.
Accuracy is also broken down by the SNR measured in each clip's leading gap.

usage: python -m rig.eval_real cache/real_quiet_1m.pt runs/v2_12_native-H-nass-kdens
"""
import argparse
import json
import os

import torch

from kws.data import LogMel
from kws.train import load_run

BUCKETS = [(-100, 0), (0, 5), (5, 10), (10, 20), (20, 100)]


@torch.no_grad()
def predict(model, frontend, audio, device, bs=1024):
    out = []
    for i in range(0, len(audio), bs):
        x = audio[i : i + bs].to(device).float() / 32768.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(frontend(x)).argmax(-1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recorded", help="cache file written by rig/align.py")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    device = torch.device("cuda")
    frontend = LogMel().to(device)

    rec = torch.load(a.recorded)
    y = rec["label"]
    snr = rec["snr_db"]
    print(f"{a.recorded}: {len(y)} clips, median SNR {snr.median():.1f} dB, "
          f"note={rec['meta'].get('note', '')!r}")

    results = {}
    for run in a.runs:
        cfg = json.load(open(os.path.join(run, "args.json")))
        ref = torch.load(f"{a.cache}/gsc_v{int(cfg['task'][1])}_test12.pt")["audio"][rec["index"]]
        model = load_run(run, len(rec["labels"]), device, ckpt=a.ckpt)

        ok_ref = (predict(model, frontend, ref, device) == y).float()
        ok_rec = (predict(model, frontend, rec["audio"], device) == y).float()
        r = {"cached": round(100 * ok_ref.mean().item(), 2),
             "recorded": round(100 * ok_rec.mean().item(), 2),
             "broke": int(((ok_ref == 1) & (ok_rec == 0)).sum()),
             "fixed": int(((ok_ref == 0) & (ok_rec == 1)).sum()),
             "by_snr": {}}
        r["channel_cost"] = round(r["cached"] - r["recorded"], 2)
        for lo, hi in BUCKETS:
            m = (snr >= lo) & (snr < hi)
            if m.sum() >= 20:
                r["by_snr"][f"{lo}..{hi}dB"] = [int(m.sum()),
                                                round(100 * ok_rec[m].mean().item(), 2)]
        name = os.path.basename(run.rstrip("/"))
        results[name] = r
        print(f"{name:34s} cached {r['cached']:6.2f}  recorded {r['recorded']:6.2f}  "
              f"channel cost {r['channel_cost']:+5.2f}  (broke {r['broke']}, fixed {r['fixed']})")
        for k, (n, acc) in r["by_snr"].items():
            print(f"{'':36s}SNR {k:>10s}  n={n:5d}  {acc:6.2f}")

    out = os.path.splitext(a.recorded)[0] + "_eval.json"
    with open(out, "w") as f:
        json.dump({"recorded_meta": rec["meta"], "runs": results}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
