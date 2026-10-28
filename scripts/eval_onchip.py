"""Measure test-set accuracy on the ESP32 itself.

Streams pre-computed log-mel features to the eval firmware
(deploy/onchip/eval_firmware/app_eval.cpp), collects the device's prediction for
each clip, and reports the accuracy actually measured on the chip next to the
host's float32 accuracy on the same clips.

The paired delta is the number this exists to produce: both columns are the same
model on the same clips, so the difference is int8 quantization and nothing else.

At ~0.95 s per inference the full 4,890-clip set takes about 77 minutes, so results
are written after every clip and --resume continues an interrupted run.

  python scripts/eval_onchip.py --features deploy/onchip/eval_features_1200.npz \
      --port COM5 --out onchip_eval_bcresnet-2.json
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import serial

MAGIC = 0xA5A5A5A5
FEAT = 40 * 101


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--port", required=True, help="COM5 on Windows, /dev/ttyUSB0 on Linux")
    ap.add_argument("--baud", type=int, default=2000000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (smoke test)")
    a = ap.parse_args()

    d = np.load(a.features, allow_pickle=True)
    feat, label = d["feat"].astype(np.float32), d["label"]
    index, host_pred = d["index"], d["host_float_pred"]
    labels = [str(x) for x in d["labels"]]
    n = len(label) if not a.limit else min(a.limit, len(label))
    print(f"{a.features}: {len(label)} clips, using {n}; host float accuracy "
          f"{float(d['host_float_acc']):.2f}% (full file)")

    done = {}
    if a.resume and os.path.exists(a.out):
        done = {int(k): v for k, v in json.load(open(a.out))["per_clip"].items()}
        print(f"resuming: {len(done)} clips already measured")

    ser = serial.Serial(a.port, a.baud, timeout=30)
    ser.reset_input_buffer()
    time.sleep(0.3)
    t0 = time.time()
    try:
        for i in range(n):
            if i in done:
                continue
            ser.write(struct.pack("<II", MAGIC, i) + feat[i].reshape(-1).tobytes())
            ser.flush()
            while True:
                line = ser.readline().decode("ascii", "replace").strip()
                if not line:
                    raise RuntimeError(f"device went silent on clip {i}")
                if line.startswith("R "):
                    break           # anything else is a log line sharing the UART
            p = line.split()
            if int(p[1]) != i:
                raise RuntimeError(f"reply index {p[1]} != request {i}; stream desynchronised")
            done[i] = {"pred": int(p[2]), "us": int(p[3]), "logits": [int(v) for v in p[4:16]]}
            if len(done) % 25 == 0 or i == n - 1:
                ok = sum(done[k]["pred"] == int(label[k]) for k in done)
                el = time.time() - t0
                rate = el / max(1, len(done))
                print(f"\r{len(done):5d}/{n}  chip acc {100 * ok / len(done):6.2f}%  "
                      f"{rate:.2f} s/clip  eta {(n - len(done)) * rate / 60:5.1f} min",
                      end="", file=sys.stderr, flush=True)
                save(a.out, a.features, labels, done, label, host_pred, index)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    finally:
        ser.close()
        save(a.out, a.features, labels, done, label, host_pred, index)

    k = sorted(done)
    chip = np.array([done[i]["pred"] for i in k])
    y, hf = label[k], host_pred[k]
    print(f"\n\nclips measured on device : {len(k)}")
    print(f"ON-CHIP int8 accuracy    : {100 * (chip == y).mean():.2f}%")
    print(f"host float32, same clips : {100 * (hf == y).mean():.2f}%")
    print(f"cost of int8             : {100 * ((hf == y).mean() - (chip == y).mean()):+.2f} points")
    print(f"chip disagrees with float: {(chip != hf).sum()} of {len(k)} clips")
    print(f"mean inference           : {np.mean([done[i]['us'] for i in k]) / 1000:.1f} ms")
    err = {}
    for j in range(len(k)):
        if chip[j] != y[j]:
            key = f"{labels[y[j]]} -> {labels[chip[j]]}"
            err[key] = err.get(key, 0) + 1
    print("top confusions:", ", ".join(f"{m} x{c}" for m, c in
                                       sorted(err.items(), key=lambda x: -x[1])[:8]) or "none")
    print(f"\nwrote {a.out}")


def save(path, features, labels, done, label, host_pred, index):
    k = sorted(done)
    chip = np.array([done[i]["pred"] for i in k])
    json.dump({"features": features, "labels": labels, "n": len(k),
               "chip_int8_acc": float(100 * (chip == label[k]).mean()),
               "host_float_acc_same_clips": float(100 * (host_pred[k] == label[k]).mean()),
               "mean_ms": float(np.mean([done[i]["us"] for i in k]) / 1000),
               "per_clip": {str(i): done[i] for i in k}},
              open(path, "w"))


if __name__ == "__main__":
    main()
