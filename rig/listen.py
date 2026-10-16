"""Live keyword spotting on the INMP441 stream: speak, and watch the model fire.

This is the fastest way to find out whether the whole chain works -- wiring, I2S
slot and bit alignment, gain, DC offset, the front-end and the model -- before any
of it is committed to the microcontroller. It runs the PyTorch model on the host,
so it says nothing about on-device latency; it only answers "does this model
recognise this microphone".

Posterior smoothing over a few windows plus a refractory period is the standard
KWS detection rule; without it a single keyword fires several times as it slides
through the window.

usage: python -m rig.listen runs/v2_12_native-H-nass-kdens --port /dev/ttyUSB0
"""
import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

from kws.data import LogMel
from kws.train import load_run
from rig.stream import SR, Stream, dbfs

LABELS12 = ["_silence_", "_unknown_", "down", "go", "left", "no", "off", "on", "right", "stop",
            "up", "yes"]
TARGET_RMS = 0.0641  # median cached keyword-clip RMS; see rig/record_live.py


def meter(level_db, width=24):
    n = int(np.clip((level_db + 60) / 60, 0, 1) * width)
    return "#" * n + "-" * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hop", type=float, default=0.1, help="seconds between evaluations")
    ap.add_argument("--smooth", type=int, default=3, help="windows averaged before deciding")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--release", type=float, default=0.4,
                    help="re-arm only once the fired class falls below this")
    ap.add_argument("--refractory", type=float, default=0.3,
                    help="minimum seconds muted after a hit")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads; more is not faster and oversubscription is slower")
    ap.add_argument("--agc", action="store_true",
                    help="scale the window to the cached clips' loudness before the model")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    labels = LABELS12
    run_labels = os.path.join(a.run, "args.json")
    if os.path.exists(run_labels) and json.load(open(run_labels))["task"].endswith("_35"):
        sys.exit("listen.py expects a 12-class run (v1_12 / v2_12)")
    model = load_run(a.run, len(labels), device, ckpt=a.ckpt)
    model.eval()
    frontend = LogMel().to(device)

    st = Stream(a.port)
    ring = np.zeros(SR)
    hop_n = max(1, int(a.hop * SR))
    probs = deque(maxlen=a.smooth)
    mute_until = 0.0
    fired = None   # hysteresis: a word stays in the 1 s window after it ends, so re-arm on
    t = 0.0        # the probability falling rather than on a timer that must guess how long
    slow = None
    t_start = time.time()
    print(f"listening on {a.port} -- say one of: {', '.join(labels[2:])}   (Ctrl-C to stop)\n")

    try:
        while True:
            chunk, _ = st.read(hop_n / SR)
            ring = np.concatenate([ring[len(chunk) :], chunk])
            t += len(chunk) / SR

            x = ring - ring.mean()
            rms = float(np.sqrt((x**2).mean()))
            if a.agc and rms > 1e-5:
                x = x * (TARGET_RMS / rms)
            with torch.no_grad():
                p = model(frontend(torch.tensor(x, dtype=torch.float32, device=device)[None]))
                p = p.softmax(-1)[0].cpu().numpy()
            probs.append(p)
            if slow is None:
                slow = time.time() - t_start > 2.5 * (len(chunk) / SR) * 5
            sm = np.mean(probs, axis=0)
            k = int(sm.argmax())

            line = f"\r{meter(dbfs(ring))} {dbfs(ring):6.1f} dBFS  {labels[k]:>9s} {sm[k]:4.2f}"
            print(line + "    ", end="", flush=True)
            if fired is not None and sm[fired] < a.release and t >= mute_until:
                fired = None
            if fired is None and k >= 2 and sm[k] >= a.threshold and t >= mute_until:
                print(f"\r{' ' * (len(line) + 6)}\r>>> {labels[k].upper()}  ({sm[k]:.2f})")
                fired = k
                mute_until = t + a.refractory
        if slow:
            print()
            print(f"NOTE: inference is slower than real time on this machine; raise --hop "
                  f"above {(time.time() - t_start) / max(t, 1e-9) * a.hop:.2f}s or use --device cuda")
            slow = False
    except KeyboardInterrupt:
        print("\nstopped")
    except RuntimeError as e:
        print(f"\n{e}")
    finally:
        st.close()


if __name__ == "__main__":
    main()
