"""Decode Google Speech Commands into compact int16 tensor caches.

Produces, for version v in {1, 2}:
  cache/gsc_v{v}.pt       all clips of the main archive with word label and split
  cache/gsc_v{v}_test12.pt  the official 12-class test set (with _silence_/_unknown_)
  cache/noise.pt          background-noise recordings (float32)
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf
import torch

SR = 16000


def load_clip(path):
    x, sr = sf.read(path, dtype="int16")
    assert sr == SR, path
    if x.ndim > 1:
        x = x[:, 0]
    out = np.zeros(SR, dtype=np.int16)
    out[: min(len(x), SR)] = x[:SR]
    return out


def read_list(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def build_main(root, workers):
    words = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith("_")
    )
    valid = read_list(os.path.join(root, "validation_list.txt"))
    test = read_list(os.path.join(root, "testing_list.txt"))
    rel_paths, labels, splits, speakers = [], [], [], []
    for wi, w in enumerate(words):
        for fn in sorted(os.listdir(os.path.join(root, w))):
            if not fn.endswith(".wav"):
                continue
            rel = f"{w}/{fn}"
            rel_paths.append(rel)
            labels.append(wi)
            splits.append(1 if rel in valid else 2 if rel in test else 0)
            speakers.append(fn.split("_nohash_")[0])
    with Pool(workers) as p:
        audio = p.map(load_clip, [os.path.join(root, r) for r in rel_paths], chunksize=256)
    return {
        "audio": torch.from_numpy(np.stack(audio)),
        "label": torch.tensor(labels, dtype=torch.int16),
        "split": torch.tensor(splits, dtype=torch.int8),
        "words": words,
        "paths": rel_paths,
        "speakers": speakers,
    }


def build_test12(root, workers):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    rel_paths, names = [], []
    for c in classes:
        for fn in sorted(os.listdir(os.path.join(root, c))):
            if fn.endswith(".wav"):
                rel_paths.append(f"{c}/{fn}")
                names.append(c)
    with Pool(workers) as p:
        audio = p.map(load_clip, [os.path.join(root, r) for r in rel_paths], chunksize=256)
    return {"audio": torch.from_numpy(np.stack(audio)), "class_name": names, "paths": rel_paths}


def build_noise(root):
    noise_dir = os.path.join(root, "_background_noise_")
    clips = []
    for fn in sorted(os.listdir(noise_dir)):
        if fn.endswith(".wav"):
            x, sr = sf.read(os.path.join(noise_dir, fn), dtype="float32")
            assert sr == SR
            clips.append(torch.from_numpy(x))
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for v in (1, 2):
        main_root = os.path.join(args.data, f"speech_commands_v0.0{v}")
        test_root = os.path.join(args.data, f"speech_commands_test_set_v0.0{v}")
        d = build_main(main_root, args.workers)
        torch.save(d, os.path.join(args.out, f"gsc_v{v}.pt"))
        counts = torch.bincount(d["split"].long(), minlength=3).tolist()
        print(f"v{v}: {len(d['paths'])} clips, {len(d['words'])} words, train/valid/test={counts}")
        t = build_test12(test_root, args.workers)
        torch.save(t, os.path.join(args.out, f"gsc_v{v}_test12.pt"))
        print(f"v{v} official 12-class test set: {len(t['paths'])} clips")
        if v == 2:
            noise = build_noise(main_root)
            torch.save(noise, os.path.join(args.out, "noise.pt"))
            print(f"noise: {len(noise)} files, {sum(len(n) for n in noise) / SR:.0f} s")


if __name__ == "__main__":
    main()
