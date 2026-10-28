"""Quantize deploy/onchip/bcresnet8.onnx to an int8 .espdl for the plain ESP32 (docs/ONCHIP.md).

Uses esp-ppq PTQ with target "c" (ESP-DL's C kernels, which is what the ESP32 runs),
calibrated on deploy/onchip/calib_128.npy. The "yes" test vector is embedded as the
model's test input so `model->test()` on the chip checks bit-exactness against PPQ.

Afterwards the quantized graph is run on the host for all 4 test vectors and compared
with the float logits, so a host-side quantization failure is separated from an
on-chip one.

    python scripts/quantize_onchip.py [--out deploy/esp32_kws/main/models/bcresnet8.espdl]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, get_target_platform
from esp_ppq.executor import TorchExecutor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "deploy" / "onchip"
INPUT_SHAPE = [1, 1, 40, 101]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "deploy" / "esp32_kws" / "main" / "models" / "bcresnet8.espdl"))
    ap.add_argument("--int16-ops", nargs="*", default=[],
                    help="ONNX node names to dispatch to w16a16 (mixed precision); default is all-int8")
    args = ap.parse_args()

    setting = QuantizationSettingFactory.espdl_setting()
    for name in args.int16_ops:
        setting.dispatching_table.append(name, get_target_platform("c", quant_type="w16a16"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    calib = np.load(SRC / "calib_128.npy").astype(np.float32).reshape(-1, *INPUT_SHAPE[1:])
    print(f"calibration tensors: {calib.shape}, range [{calib.min():.3f}, {calib.max():.3f}]")
    calib_list = [torch.from_numpy(c[None]) for c in calib]  # each [1,1,40,101]

    tv = np.load(SRC / "test_vectors.npz")
    tv_in = tv["feat"].astype(np.float32).reshape(-1, *INPUT_SHAPE[1:])
    tv_logits = tv["logits"]
    tv_labels = tv["label"]

    graph = espdl_quantize_onnx(
        onnx_import_file=str(SRC / "bcresnet8.onnx"),
        espdl_export_file=str(out),
        calib_dataloader=calib_list,
        calib_steps=len(calib_list),
        input_shape=INPUT_SHAPE,
        inputs=[torch.from_numpy(tv_in[0:1])],
        target="c",
        quant_type="w8a8",
        setting=setting,
        collate_fn=lambda x: x.to("cpu"),
        device="cpu",
        error_report=True,
        export_test_values=True,
        verbose=1,
    )

    executor = TorchExecutor(graph=graph, device="cpu")
    results = []
    for i in range(len(tv_in)):
        q = executor(torch.from_numpy(tv_in[i : i + 1]))[0].detach().cpu().numpy().reshape(-1)
        f = tv_logits[i].reshape(-1)
        top2 = np.sort(q)[-2:]
        results.append(
            {
                "expected": int(tv_labels[i]),
                "host_float_argmax": int(f.argmax()),
                "host_int8_argmax": int(q.argmax()),
                "int8_margin": float(top2[1] - top2[0]),
                "max_abs_diff": float(np.abs(q - f).max()),
                "int8_logits": [round(float(v), 4) for v in q],
            }
        )
        print(f"tv{i}: expected {tv_labels[i]}  float {f.argmax()}  int8(ppq) {q.argmax()}  "
              f"margin {top2[1] - top2[0]:.3f}  max|dq| {np.abs(q - f).max():.3f}")
    (out.parent / "ppq_host_check.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
