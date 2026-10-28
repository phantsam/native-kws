"""Quantize one deploy/onchip/<model>/ export to an int8 .espdl for the plain ESP32 (docs/ONCHIP.md).

Uses esp-ppq PTQ with target "c" (ESP-DL's C kernels, which is what the ESP32 runs),
calibrated on that directory's calib_128.npy. The ONNX, calib_128.npy and
test_vectors.npz all come from the one --src directory: they must belong together.
The "yes" test vector is embedded as the model's test input so `model->test()` on
the chip checks bit-exactness against PPQ.

Afterwards the quantized graph is run on the host for all 4 test vectors and compared
with the float logits, so a host-side quantization failure is separated from an
on-chip one.

    python scripts/quantize_onchip.py --src deploy/onchip/bcresnet-2 --out deploy/esp32_kws/main/models/model.espdl

The output is always called model.espdl so the embedded-binary symbol
(_binary_model_espdl_start) does not change with the model scale.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, get_target_platform
from esp_ppq.executor import TorchExecutor

ROOT = Path(__file__).resolve().parents[1]
INPUT_SHAPE = [1, 1, 40, 101]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "deploy" / "onchip" / "bcresnet-2"),
                    help="one export directory: <model>.onnx + calib_128.npy + test_vectors.npz")
    ap.add_argument("--out", default=str(ROOT / "deploy" / "esp32_kws" / "main" / "models" / "model.espdl"))
    ap.add_argument("--int16-ops", nargs="*", default=[],
                    help="ONNX node names to dispatch to w16a16 (mixed precision); default is all-int8")
    args = ap.parse_args()

    setting = QuantizationSettingFactory.espdl_setting()
    for name in args.int16_ops:
        setting.dispatching_table.append(name, get_target_platform("c", quant_type="w16a16"))
    src = Path(args.src)
    onnx_files = sorted(src.glob("*.onnx"))
    if len(onnx_files) != 1:
        raise SystemExit(f"expected exactly one .onnx in {src}, found {[f.name for f in onnx_files]}")
    print(f"source: {onnx_files[0]}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    calib = np.load(src / "calib_128.npy").astype(np.float32).reshape(-1, *INPUT_SHAPE[1:])
    print(f"calibration tensors: {calib.shape}, range [{calib.min():.3f}, {calib.max():.3f}]")
    calib_list = [torch.from_numpy(c[None]) for c in calib]  # each [1,1,40,101]

    tv = np.load(src / "test_vectors.npz")
    tv_in = tv["feat"].astype(np.float32).reshape(-1, *INPUT_SHAPE[1:])
    tv_logits = tv["logits"]
    tv_labels = tv["label"]

    # esp-ppq re-saves the ONNX it is given, so hand it a copy: deploy/onchip/* must stay untouched.
    tmp = Path(tempfile.mkdtemp())
    onnx_copy = tmp / onnx_files[0].name
    shutil.copyfile(onnx_files[0], onnx_copy)

    graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_copy),
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

    shutil.rmtree(tmp, ignore_errors=True)

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
