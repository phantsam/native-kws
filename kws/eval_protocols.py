"""Evaluate checkpoints under both 12-class test protocols used in the literature.

  official : the separately released 12-class test set (speech_commands_test_set_v0.0x),
             4,890 clips for v2. Used by BC-ResNet (and by our training runs).
  list     : built from testing_list.txt: all keyword clips + 10 % unknown (random other
             words from the test split) + 10 % silence (background-noise slices scaled by
             U(0, 1)). Used by KWT (kws_streaming) and Keyword Mamba (NeMo-style script).
             The exact random unknown/silence draws of those repos cannot be reproduced
             bit-for-bit, so this is the same protocol with our own seed.

Both use the same training split, so no retraining is needed.

usage: python -m kws.eval_protocols runs/<run> [runs/...] [--ckpt best.pt]
"""
import argparse
import json
import math
import os

import torch

from kws.data import GSC, KEYWORDS10, LABELS12, LogMel
from kws.train import evaluate, load_run


def list_protocol_test(cache_dir, ver, noise, seed=0):
    main = torch.load(f"{cache_dir}/gsc_v{ver}.pt")
    words, label, split = main["words"], main["label"].long(), main["split"].long()
    to12 = torch.tensor([LABELS12.index(w) if w in KEYWORDS10 else 1 for w in words])[label]
    g = torch.Generator().manual_seed(seed)
    te = split == 2
    kw_audio, kw_label = main["audio"][te & (to12 >= 2)], to12[te & (to12 >= 2)]
    n_extra = math.ceil(0.1 * len(kw_audio))
    unk = main["audio"][te & (to12 == 1)]
    unk = unk[torch.randperm(len(unk), generator=g)[:n_extra]]
    idx = torch.randint(len(noise), (n_extra,), generator=g)
    sil = ((noise.cpu()[idx] * torch.rand(n_extra, 1, generator=g)).clamp(-1, 1) * 32767).round().to(torch.int16)
    audio = torch.cat([kw_audio, unk, sil])
    labels = torch.cat([kw_label, torch.ones(n_extra, dtype=torch.long), torch.zeros(n_extra, dtype=torch.long)])
    return audio, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()
    device = torch.device("cuda")
    frontend = LogMel().to(device)
    cache = {}
    for run in args.runs:
        cfg = json.load(open(os.path.join(run, "args.json")))
        task = cfg["task"]
        if task.endswith("_35"):
            print(f"{run}: 35-class task has a single protocol (testing_list); see summary.json")
            continue
        if task not in cache:
            data = GSC(task, args.cache, device)
            cache[task] = (data, list_protocol_test(args.cache, int(task[1]), data.noise))
        data, list_test = cache[task]
        res = {"ckpt": args.ckpt}
        for ck in [args.ckpt, "last.pt"]:
            if not os.path.exists(os.path.join(run, ck)):
                continue
            model = load_run(run, data.num_classes, device, ckpt=ck)
            res[ck] = {
                "official": round(evaluate(model, frontend, data.test, device), 2),
                "list": round(evaluate(model, frontend, list_test, device), 2),
            }
        res["list_test_size"] = len(list_test[1])
        print(os.path.basename(run.rstrip("/")), json.dumps(res), flush=True)
        with open(os.path.join(run, "protocol_eval.json"), "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
