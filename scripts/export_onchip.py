"""Export BC-ResNet-8 for the ESP32, with int8 calibration data and on-chip test vectors.

Milestone 1 of docs/ONCHIP.md needs three things on the device: the network, a few
real inputs, and the logits the host produced for them. This writes all three into
deploy/onchip/ so the device-side work needs no dataset and no GPU.

The graph is post-processed so every operator is one ESP-DL supports: onnxsim folds
BatchNorm into a preceding Conv where the pattern allows, and the BatchNorms that
survive -- BC-ResNet's SubSpectralNorm normalises a reshaped tensor, so its BN does
not sit directly behind a Conv -- are rewritten as the per-channel affine they
already are at inference time, which is a whitelisted Mul plus Add.

usage: python scripts/export_onchip.py
"""
import json
import os

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

from kws.data import LogMel
from kws.train import load_run

OUT = "deploy/onchip"
RUN = "runs/v2_12_bcresnet-8"
CKPT = "best.pt"
LABELS12 = ["_silence_", "_unknown_", "down", "go", "left", "no", "off", "on", "right", "stop",
            "up", "yes"]
# ESP-DL int8 operators for ESP32/ESP32-S3, from Espressif's operator support table
# (recorded in docs/LITERATURE.md §6). BatchNormalization is deliberately absent.
ESPDL_OPS = {
    "Conv", "Gemm", "MatMul", "Add", "Mul", "Div", "Sub", "Pow", "AveragePool",
    "GlobalAveragePool", "ReduceMean", "Relu", "Clip", "LeakyRelu", "PRelu", "HardSwish",
    "HardSigmoid", "Sigmoid", "Tanh", "Elu", "Swish", "Exp", "Log", "Sqrt", "Softmax",
    "LogSoftmax", "LayerNormalization", "RMSNormalization", "LSTM", "GRU", "Transpose",
    "Reshape", "Concat", "Pad", "Slice", "Split", "Resize", "Identity", "Flatten",
    "Squeeze", "Unsqueeze", "Shape", "Constant", "Gather",
}


def bn_to_mul_add(model):
    """Rewrite every surviving BatchNormalization as Mul + Add (inference-time affine)."""
    model = onnx.shape_inference.infer_shapes(model)
    g = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    rank = {vi.name: len(vi.type.tensor_type.shape.dim)
            for vi in list(g.value_info) + list(g.input) + list(g.output)}
    nodes, new_init, n = [], [], 0
    for node in g.node:
        if node.op_type != "BatchNormalization":
            nodes.append(node)
            continue
        x, s_n, b_n, m_n, v_n = node.input[:5]
        eps = next((a.f for a in node.attribute if a.name == "epsilon"), 1e-5)
        scale = init[s_n] / np.sqrt(init[v_n] + eps)
        shift = init[b_n] - init[m_n] * scale
        shape = [1] * rank.get(x, 4)
        shape[1] = scale.shape[0]
        sc = numpy_helper.from_array(scale.reshape(shape).astype(np.float32), f"bnfold_s_{n}")
        sh = numpy_helper.from_array(shift.reshape(shape).astype(np.float32), f"bnfold_b_{n}")
        mid = f"bnfold_mid_{n}"
        new_init += [sc, sh]
        nodes.append(helper.make_node("Mul", [x, sc.name], [mid], name=f"bnfold_mul_{n}"))
        nodes.append(helper.make_node("Add", [mid, sh.name], [node.output[0]],
                                      name=f"bnfold_add_{n}"))
        n += 1
    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(new_init)
    print(f"  rewrote {n} BatchNormalization node(s) as Mul+Add")
    return model


def strip_unused_initializers(model):
    used = {i for node in model.graph.node for i in node.input}
    keep = [i for i in model.graph.initializer if i.name in used]
    dropped = len(model.graph.initializer) - len(keep)
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    print(f"  dropped {dropped} unused initializer(s)")
    return model


def make_espdl_ready(path):
    m = onnx.load(path)
    before = sorted({n.op_type for n in m.graph.node})
    import onnxsim
    m, ok = onnxsim.simplify(m)
    print(f"  onnxsim: {'ok' if ok else 'CHECK FAILED'}")
    m = strip_unused_initializers(bn_to_mul_add(m))
    m = onnx.shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    onnx.save(m, path)
    after = sorted({n.op_type for n in m.graph.node})
    missing = [o for o in after if o not in ESPDL_OPS]
    print(f"  ops before: {before}")
    print(f"  ops after : {after}")
    print(f"  not on the ESP-DL whitelist: {missing if missing else 'none'}")
    return after, missing


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cpu")
    torch.set_num_threads(4)
    model = load_run(RUN, 12, dev, ckpt=CKPT).eval()
    fe = LogMel().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{RUN} loaded, {n_par} params")

    d = torch.load("cache/gsc_v2_test12.pt")
    audio, names = d["audio"], d["class_name"]
    y = torch.tensor([LABELS12.index(c) for c in names])
    with torch.no_grad():
        correct = sum((model(fe(audio[i : i + 512].float() / 32768.0)).argmax(-1)
                       == y[i : i + 512]).sum().item() for i in range(0, len(audio), 512))
        acc = 100 * correct / len(audio)
        print(f"PyTorch float accuracy on the official v2_12 test set: {acc:.2f}%")

        onnx_path = f"{OUT}/bcresnet8.onnx"
        torch.onnx.export(model, (torch.zeros(1, 1, 40, 101),), onnx_path,
                          input_names=["logmel"], output_names=["logits"],
                          opset_version=13, dynamo=False)
        ops, missing = make_espdl_ready(onnx_path)
        print(f"wrote {onnx_path} ({os.path.getsize(onnx_path) / 1024:.0f} KB)")

        rng = np.random.default_rng(0)
        pick = rng.permutation(len(audio))[:128]
        calib = fe(audio[pick].float() / 32768.0).numpy().astype(np.float32)
        np.save(f"{OUT}/calib_128.npy", calib)
        print(f"wrote calib_128.npy {calib.shape}")

        want = ["yes", "no", "stop", "_silence_"]
        feats, labs, logits, srcs = [], [], [], []
        for w in want:
            i = int(next(k for k in range(len(names)) if names[k] == w))
            f = fe(audio[i : i + 1].float() / 32768.0)
            lg = model(f)[0]
            feats.append(f[0, 0].numpy())
            labs.append(LABELS12.index(w))
            logits.append(lg.numpy())
            srcs.append(d["paths"][i])
            print(f"  {w:10s} -> host predicts {LABELS12[int(lg.argmax())]:10s} ({d['paths'][i]})")

    np.savez(f"{OUT}/test_vectors.npz", feat=np.stack(feats), label=np.array(labs),
             logits=np.stack(logits), labels=np.array(LABELS12))
    with open(f"{OUT}/test_vectors.h", "w") as f:
        f.write("// Generated by scripts/export_onchip.py -- do not edit.\n")
        f.write("// 4 log-mel inputs (40 mel bands x 101 frames, mel-major) with the float\n")
        f.write("// logits the host produced, so an on-chip run can be checked numerically.\n")
        f.write("#pragma once\n\n")
        f.write(f"#define TV_COUNT {len(feats)}\n#define TV_MELS 40\n#define TV_FRAMES 101\n")
        f.write("#define TV_CLASSES 12\n\n")
        f.write("static const char *const tv_class_names[TV_CLASSES] = {"
                + ", ".join(f'"{l}"' for l in LABELS12) + "};\n\n")
        f.write("static const int tv_expected_label[TV_COUNT] = {"
                + ", ".join(str(l) for l in labs) + "};\n\n")
        f.write("static const float tv_host_logits[TV_COUNT][TV_CLASSES] = {\n")
        for lg in logits:
            f.write("  {" + ", ".join(f"{v:.6f}f" for v in lg) + "},\n")
        f.write("};\n\n")
        f.write("static const float tv_input[TV_COUNT][TV_MELS * TV_FRAMES] = {\n")
        for feat in feats:
            f.write("  {" + ",".join(f"{v:.5f}f" for v in feat.reshape(-1)) + "},\n")
        f.write("};\n")
    print(f"wrote test_vectors.h ({os.path.getsize(f'{OUT}/test_vectors.h') / 1024:.0f} KB)")

    with open(f"{OUT}/export_info.json", "w") as f:
        json.dump({"source_run": RUN, "ckpt": CKPT, "params": n_par,
                   "float_test_accuracy_v2_12": round(acc, 2), "opset": 13,
                   "input": "logmel (1,1,40,101) float32, 40 mel bands, 30 ms window, 10 ms hop",
                   "batchnorm_folded": True, "onnx_ops": ops,
                   "ops_missing_from_espdl_whitelist": missing,
                   "test_vector_sources": srcs}, f, indent=2)
    print("wrote export_info.json")


if __name__ == "__main__":
    main()
