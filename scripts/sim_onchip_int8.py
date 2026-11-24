"""Predict on-chip int8 accuracy on the host, with esp-ppq's simulation of the quantized model.

E32 showed the ESP32 is bit-exact with esp-ppq: identical int8 logits on 4,890 / 4,890
test clips. So this runs in ~4 minutes what scripts/eval_onchip.py measures on the board
in ~92, and is the way to screen a quantization or export change before flashing it.

The export in --src is quantized exactly as scripts/quantize_onchip.py does (into a
temporary file -- the deployed model.espdl is not touched), then run over the clips in
--features. With --chip, the device's results from eval_onchip.py are compared with the
simulation clip by clip, logit by logit.

  python scripts/sim_onchip_int8.py --src deploy/onchip/bcresnet-2 \
      --features deploy/onchip/eval_features.npz \
      --chip deploy/esp32_kws/onchip_eval_bcresnet-2.json
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from esp_ppq.executor import TorchExecutor

from quantize_onchip import INPUT_SHAPE, quantize


def output_step(q):
    """The power-of-two step the dequantized int8 logits sit on (2**exponent)."""
    for e in range(4, -12, -1):
        if np.allclose(q / 2.0**e, np.rint(q / 2.0**e), atol=1e-4):
            return 2.0**e
    raise SystemExit("simulated logits are not on a power-of-two grid")


def confusions(pred, y, labels, n=8):
    err = {}
    for p, t in zip(pred, y):
        if p != t:
            key = f"{labels[t]} -> {labels[p]}"
            err[key] = err.get(key, 0) + 1
    return ", ".join(f"{m} x{c}" for m, c in sorted(err.items(), key=lambda x: -x[1])[:n]) or "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="export directory, e.g. deploy/onchip/bcresnet-2")
    ap.add_argument("--features", default="deploy/onchip/eval_features.npz")
    ap.add_argument("--target", default="c", help='esp-ppq target: "c" (plain ESP32), "esp32s3", "esp32p4"')
    ap.add_argument("--int16-ops", nargs="*", default=[])
    ap.add_argument("--chip", help="results JSON from eval_onchip.py to compare against")
    ap.add_argument("--out", help="save the simulated int8 logits (.npz)")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    graph = quantize(a.src, tmp / "model.espdl", a.target, a.int16_ops, error_report=False, verbose=0)
    shutil.rmtree(tmp, ignore_errors=True)
    ex = TorchExecutor(graph=graph, device="cpu")

    d = np.load(a.features, allow_pickle=True)
    feat = d["feat"].astype(np.float32)
    y, hf, labels = d["label"], d["host_float_pred"], [str(x) for x in d["labels"]]
    q = np.zeros((len(feat), len(labels)), np.float32)
    with torch.no_grad():
        for i in range(len(feat)):
            q[i] = ex(torch.from_numpy(feat[i].reshape(INPUT_SHAPE)))[0].reshape(-1).numpy()
    step = output_step(q)
    qi = np.rint(q / step).astype(int)
    sp = qi.argmax(1)       # first maximum, as the firmware's argmax does
    if a.out:
        np.savez(a.out, int8_logits=qi, step=step, index=d["index"])

    print(f"\n{a.features}: {len(y)} clips, target {a.target}, output step {step}")
    print(f"esp-ppq int8 (simulated) : {100 * (sp == y).mean():.2f}%")
    print(f"host float32, same clips : {100 * (hf == y).mean():.2f}%")
    print(f"cost of int8             : {100 * ((hf == y).mean() - (sp == y).mean()):.2f} points")
    print(f"int8 disagrees with float: {(sp != hf).sum()} of {len(y)} clips")
    print(f"top confusions: {confusions(sp, y, labels)}")

    if a.chip:
        r = json.load(open(a.chip))["per_clip"]
        k = sorted(int(i) for i in r)
        chip = np.array([r[str(i)]["logits"] for i in k])
        same = (chip == qi[k]).all(1)
        print(f"\nchip vs simulation on the {len(k)} measured clips: identical logits on {same.sum()}, "
              f"max |diff| {np.abs(chip - qi[k]).max()} LSB, "
              f"argmax differs on {(np.array([r[str(i)]['pred'] for i in k]) != sp[k]).sum()}")


if __name__ == "__main__":
    main()
