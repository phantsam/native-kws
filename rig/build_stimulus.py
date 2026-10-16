"""Build the replay stimulus: the official Speech Commands test clips, back to back.

Playing the *same* test clips through a speaker into the INMP441 gives paired data:
the clip identity, the speaker and the label are unchanged, so the accuracy
difference between the cached clip and its recording is the microphone channel
alone, with no speaker or vocabulary confound.

The stimulus is cut into segments, each opening with a frequency sweep. Playback
clock and ESP32-S3 I2S clock are independent, so the recording drifts a few
hundred ppm against the stimulus; align.py re-anchors on every sweep and
interpolates between them, which keeps the per-clip error far below the gap.

usage: python rig/build_stimulus.py --out rig/stimulus --task v2_12
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch

SR = 16000
LABELS12 = ["_silence_", "_unknown_", "down", "go", "left", "no", "off", "on", "right", "stop",
            "up", "yes"]


def sweep(seconds, f0, f1, sr):
    t = np.arange(int(seconds * sr)) / sr
    x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * seconds) * t**2))
    env = np.minimum(1.0, np.minimum(t, seconds - t) / 0.01)  # 10 ms fades
    return (x * env).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rig/stimulus")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--task", default="v2_12")
    ap.add_argument("--gap", type=float, default=0.5, help="silence between clips (s)")
    ap.add_argument("--per_segment", type=int, default=50, help="clips between resync sweeps")
    ap.add_argument("--max_minutes", type=float, default=20, help="split into files of this length")
    ap.add_argument("--sr_out", type=int, default=48000, help="write at the sound card's rate")
    ap.add_argument("--limit", type=int, default=0, help="use only N clips (dry run)")
    ap.add_argument("--seed", type=int, default=0, help="clip order; fixed so takes are comparable")
    a = ap.parse_args()

    ver = int(a.task[1])
    d = torch.load(f"{a.cache}/gsc_v{ver}_test12.pt")
    audio = d["audio"].numpy()
    labels = np.array([LABELS12.index(c) for c in d["class_name"]])
    # The cached set is ordered by class. Shuffle so class never correlates with
    # position in the recording, and so a short --limit take stays representative.
    order = np.random.default_rng(a.seed).permutation(len(audio))
    if a.limit:
        order = order[: a.limit]
    audio, labels = audio[order], labels[order]
    print(f"{len(audio)} clips from the official {a.task} test set")

    chirp = sweep(0.1, 2000, 6000, SR)
    lead = np.zeros(int(0.3 * SR), np.float32)
    gap = np.zeros(int(a.gap * SR), np.float32)
    per_file = int(a.max_minutes * 60 / (a.per_segment * (1 + a.gap) + 0.5)) * a.per_segment

    os.makedirs(a.out, exist_ok=True)
    manifest = {"sr_stim": SR, "chirp_seconds": 0.1, "chirp_f0": 2000, "chirp_f1": 6000,
                "clip_seconds": 1.0, "gap_seconds": a.gap, "labels": LABELS12,
                "task": a.task, "files": []}

    for fi, start in enumerate(range(0, len(audio), per_file)):
        part = list(range(start, min(start + per_file, len(audio))))
        buf, segs, cur = [lead], [], len(lead)
        for s in range(0, len(part), a.per_segment):
            buf.append(chirp)
            seg = {"chirp_start": cur, "clips": []}
            cur += len(chirp)
            buf.append(gap)
            cur += len(gap)
            for i in part[s : s + a.per_segment]:
                clip = audio[i].astype(np.float32) / 32768.0
                seg["clips"].append({"index": int(order[i]), "label": int(labels[i]),
                                     "start": cur,
                                     "rms": float(np.sqrt((clip**2).mean()))})
                buf.append(clip)
                cur += SR
                buf.append(gap)
                cur += len(gap)
            segs.append(seg)
        x = np.concatenate(buf)
        x = x / max(np.abs(x).max(), 1e-9) * 0.7  # one global gain: relative levels preserved
        if a.sr_out != SR:
            x = torch.nn.functional.interpolate(
                torch.from_numpy(x)[None, None], scale_factor=a.sr_out / SR,
                mode="linear", align_corners=False)[0, 0].numpy()
        name = f"stimulus_{fi:02d}.wav"
        sf.write(os.path.join(a.out, name), x, a.sr_out, subtype="PCM_16")
        manifest["files"].append({"name": name, "n_clips": len(part),
                                  "seconds_at_16k": int(cur) / SR, "segments": segs})
        print(f"{name}: {len(part)} clips, {cur / SR / 60:.1f} min")

    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    total = sum(f["seconds_at_16k"] for f in manifest["files"])
    print(f"manifest.json written; {total / 60:.1f} min per condition")


if __name__ == "__main__":
    main()
