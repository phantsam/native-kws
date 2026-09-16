import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from kws.data import GSC, Augment, LogMel, eval_batches, spec_augment, speech_presence_targets
from kws.models import build_model, count_params

PRESETS = {
    "S": dict(stem=24, channels=(24, 32, 48), blocks=(2, 2, 2), dim=64, ssm_layers=2, expand=2, state=12),
    "M": dict(stem=32, channels=(32, 48, 64), blocks=(2, 2, 2), dim=96, ssm_layers=3, expand=2, state=16),
    "L": dict(stem=48, channels=(48, 64, 96), blocks=(2, 2, 3), dim=128, ssm_layers=4, expand=2, state=16),
    # Hybrid: BC-ResNet-depth conv backbone + light Mamba back-end.
    "H": dict(stem=96, channels=(48, 72, 96, 120), blocks=(2, 2, 4, 4), freq_strides=(1, 2, 2, 1),
              dim=96, ssm_layers=2, expand=1.5, state=16),
}


def model_kwargs(cfg):
    """Constructor kwargs for a run config (dict of CLI args; older runs lack newer keys)."""
    if cfg["model"].startswith("bcssm"):
        return dict(ssm_stages=tuple(cfg.get("ssm_stages", (2, 3))), state=cfg.get("state", 8),
                    causal=cfg["causal"], noise_aware=cfg.get("noise_aware", False),
                    dropout=cfg.get("dropout", 0.1))
    return dict(PRESETS[cfg["size"]], causal=cfg["causal"], dropout=cfg.get("dropout", 0.1),
                drop_path=cfg.get("drop_path", 0.1), noise_aware=cfg.get("noise_aware", False))


def load_run(run_dir, num_classes, device, ckpt="last_ema.pt"):
    cfg = json.load(open(os.path.join(run_dir, "args.json")))
    model = build_model(cfg["model"], num_classes, **model_kwargs(cfg)).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, ckpt), map_location=device))
    return model.eval()


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="v2_12", choices=["v1_12", "v2_12", "v2_35"])
    ap.add_argument("--model", default="native", help="native | bcresnet-<tau> | bcssm-<tau>")
    ap.add_argument("--size", default="M", choices=list(PRESETS))
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--opt", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--specaug_f", type=int, default=7)
    ap.add_argument("--specaug_t", type=int, default=20)
    ap.add_argument("--no_specaug", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--teacher", nargs="*", default=[], help="run dirs whose averaged logits are distilled")
    ap.add_argument("--kd_alpha", type=float, default=0.7)
    ap.add_argument("--kd_T", type=float, default=2.0)
    # BC-SSM (idea 1) and noise-aware selective scan (idea 2)
    ap.add_argument("--ssm_stages", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--state", type=int, default=8)
    ap.add_argument("--noise_aware", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile the training forward pass")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/state.pt if it exists")
    ap.add_argument("--gate_weight", type=float, default=0.5, help="weight of the speech-presence loss (0 = unsupervised gate)")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, frontend, split, device, bs=1024):
    model.eval()
    correct, total = 0, 0
    for x, y in eval_batches(*split, bs, device):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(frontend(x)).argmax(-1)
        correct += (pred == y).sum().item()
        total += len(y)
    return 100.0 * correct / total


def param_groups(model, wd):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or getattr(p, "_no_weight_decay", False) else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    name = args.name or f"{args.task}_{args.model}{'' if args.model.startswith('bc') else '-' + args.size}" + (f"-s{args.seed}" if args.seed else "")
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    data = GSC(args.task, args.cache, device, seed=0)  # fixed data seed: identical validation set for every run
    model = build_model(args.model, data.num_classes, **model_kwargs(vars(args))).to(device)
    use_gate = getattr(model, "noise_aware", False)
    train_model = torch.compile(model) if args.compile else model
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    frontend = LogMel().to(device)
    augment = Augment(data.noise, has_silence=not args.task.endswith("_35"))
    teachers = [load_run(t, data.num_classes, device) for t in args.teacher]
    for t, tm in zip(args.teacher, teachers):
        print(f"teacher {t}: test {evaluate(tm, frontend, data.test, device):.2f}", flush=True)
    n_params = count_params(model)
    print(f"{name}: {n_params} params | train kw {len(data.train_audio)} | "
          f"valid {len(data.valid[1])} | test {len(data.test[1])}", flush=True)

    if args.opt == "sgd":
        opt = torch.optim.SGD(param_groups(model, args.wd), lr=args.lr, momentum=0.9)
    else:
        opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.98))

    gen = torch.Generator(device=device).manual_seed(args.seed)
    steps_per_epoch = sum(1 for _ in data.train_epoch(args.bs, gen))
    total, warm = steps_per_epoch * args.epochs, steps_per_epoch * args.warmup
    step, best, start_epoch = 0, {"valid": -1.0}, 1
    state_path, log_path = os.path.join(out_dir, "state.pt"), os.path.join(out_dir, "log.jsonl")
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        step, best, start_epoch = st["step"], st["best"], st["epoch"] + 1
        torch.set_rng_state(st["rng_cpu"].cpu())
        torch.cuda.set_rng_state(st["rng_cuda"].cpu())
        # keep only log records up to the checkpointed epoch
        recs = [l for l in open(log_path) if json.loads(l)["epoch"] <= st["epoch"]] if os.path.exists(log_path) else []
        open(log_path, "w").writelines(recs)
        print(f"resumed from epoch {st['epoch']} (step {step})", flush=True)
    elif os.path.exists(log_path):
        os.remove(log_path)  # fresh start: never mix records of different attempts
    log = open(log_path, "a")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, loss_sum, n_seen, gate_sum, gate_agree = time.time(), 0.0, 0, 0.0, 0.0
        for x, y in data.train_epoch(args.bs, gen):
            lr = args.lr * (step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm))))
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.no_grad():
                if use_gate:
                    mix, clean, noise = augment(x, y, return_parts=True)
                    feats = frontend(mix)
                    target = speech_presence_targets(clean, noise)
                else:
                    feats = frontend(augment(x, y))
                if not args.no_specaug:
                    feats, time_keep = spec_augment(feats, args.specaug_f, args.specaug_t, return_time_keep=True)
                    if use_gate:  # a masked frame shows no speech
                        target = target * time_keep
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if use_gate:
                    logits, gate_logits = train_model(feats, return_gate=True)
                else:
                    logits = train_model(feats)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=args.smoothing)
            if use_gate:
                gate_loss = F.binary_cross_entropy_with_logits(gate_logits.float(), target)
                if args.gate_weight > 0:
                    loss = loss + args.gate_weight * gate_loss
                gate_sum += gate_loss.item() * len(y)
                gate_agree += ((gate_logits > 0) == (target > 0.5)).float().mean().item() * len(y)
            if teachers:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    t_logits = torch.stack([tm(feats).float() for tm in teachers]).mean(0)
                T = args.kd_T
                kd = F.kl_div(F.log_softmax(logits.float() / T, -1), F.log_softmax(t_logits / T, -1),
                              reduction="batchmean", log_target=True) * T * T
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update_parameters(model)
            step += 1
            loss_sum += loss.item() * len(y)
            n_seen += len(y)

        rec = {"epoch": epoch, "loss": loss_sum / n_seen, "lr": lr, "sec": time.time() - t0}
        if use_gate:
            rec["gate_loss"] = gate_sum / n_seen
            rec["gate_frame_acc"] = 100.0 * gate_agree / n_seen
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            for tag, m in (("raw", model), ("ema", ema.module)):
                rec[f"valid_{tag}"] = evaluate(m, frontend, data.valid, device)
                rec[f"test_{tag}"] = evaluate(m, frontend, data.test, device)
            v = max(rec["valid_raw"], rec["valid_ema"])
            if v > best["valid"]:
                tag = "ema" if rec["valid_ema"] >= rec["valid_raw"] else "raw"
                best = {"valid": v, "test": rec[f"test_{tag}"], "epoch": epoch, "which": tag}
                torch.save((ema.module if tag == "ema" else model).state_dict(), os.path.join(out_dir, "best.pt"))
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if epoch % args.eval_every == 0:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "epoch": epoch, "best": best, "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state()}, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)  # atomic: a kill never leaves a torn checkpoint

    torch.save(ema.module.state_dict(), os.path.join(out_dir, "last_ema.pt"))
    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    summary = {"name": name, "params": n_params, "best_by_valid": best,
               "last_test_raw": rec["test_raw"], "last_test_ema": rec["test_ema"]}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
