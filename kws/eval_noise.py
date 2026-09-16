"""Noise-robustness evaluation: test accuracy with background noise mixed in at fixed SNRs.

Noise comes from the GSC _background_noise_ recordings (white/pink noise, running
tap, exercise bike, dishes). Silence clips are left as they are.

usage: python -m kws.eval_noise runs/v2_12_native-M [runs/...] --ckpt best.pt
"""
import argparse
import json
import os

import torch

from kws.data import GSC, LogMel
from kws.train import load_run

SNRS = [None, 20, 10, 5, 0, -5]


@torch.no_grad()
def noisy_accuracy(model, frontend, data, snr, device, seed=0, bs=1024):
    g = torch.Generator(device=device).manual_seed(seed)
    audio, label = data.test
    correct = 0
    for i in range(0, len(audio), bs):
        x = audio[i : i + bs].to(device).float() / 32768.0
        y = label[i : i + bs].to(device)
        if snr is not None:
            noise = data.noise[torch.randint(len(data.noise), (len(x),), device=device, generator=g)]
            p_sig = x.pow(2).mean(1, keepdim=True).clamp_min(1e-10)
            p_noise = noise.pow(2).mean(1, keepdim=True).clamp_min(1e-10)
            scale = torch.sqrt(p_sig / (p_noise * 10 ** (snr / 10)))
            keep = (y != 0).float()[:, None] if data.num_classes == 12 else 1.0
            x = (x + keep * scale * noise).clamp(-1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            correct += (model(frontend(x)).argmax(-1) == y).sum().item()
    return 100.0 * correct / len(audio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()
    device = torch.device("cuda")
    frontend = LogMel().to(device)
    datasets = {}
    for run in args.runs:
        cfg = json.load(open(os.path.join(run, "args.json")))
        if cfg["task"] not in datasets:
            datasets[cfg["task"]] = GSC(cfg["task"], args.cache, device)
        data = datasets[cfg["task"]]
        model = load_run(run, data.num_classes, device, ckpt=args.ckpt)
        res = {("clean" if s is None else f"{s}dB"): round(noisy_accuracy(model, frontend, data, s, device), 2) for s in SNRS}
        print(os.path.basename(run.rstrip("/")), json.dumps(res), flush=True)
        with open(os.path.join(run, "noise_eval.json"), "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
