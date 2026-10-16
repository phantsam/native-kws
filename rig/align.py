"""Cut a replay recording into labelled 1 s clips.

Finds every resync sweep by matched filtering, maps stimulus time to recording
time by interpolating between consecutive sweeps (so playback/I2S clock drift is
removed rather than accumulated), and writes a cache file in the same layout as
scripts/prepare_data.py, so the recorded set can be evaluated by the normal
pipeline with no retraining.

Each clip also gets a measured SNR: the noise floor is read from the silent gap
in front of it, which is what makes the recorded set usable as a noise-robustness
benchmark instead of only an accuracy number.

usage: python rig/align.py rig/out/quiet_1m.wav --stimulus rig/stimulus \
           --out cache/real_quiet_1m.pt
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch

SR = 16000


def sweep(seconds, f0, f1, sr):
    t = np.arange(int(seconds * sr)) / sr
    x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * seconds) * t**2))
    env = np.minimum(1.0, np.minimum(t, seconds - t) / 0.01)
    return (x * env).astype(np.float32)


def matched_filter(x, template):
    n = 1 << int(np.ceil(np.log2(len(x) + len(template))))
    c = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(template, n)), n)
    return np.abs(c[: len(x)])


def dc_removed(x, fc=20.0, sr=SR):
    """Remove the INMP441's DC offset: subtract a moving average below `fc`."""
    win = int(sr / fc)
    pad = np.pad(x, (win, win), mode="edge")
    k = np.ones(win) / win
    base = np.convolve(pad, k, mode="same")[win : win + len(x)]
    return x - base


def find_chirps(rec, nominal, template, max_lead=30.0, guard=0.4):
    """Recording-sample position of every sweep whose stimulus position is `nominal`.

    Tracked sequentially: each sweep is predicted from the clock ratio measured over
    the sweeps already found, then refined by a local search. The prediction error
    therefore stays at the per-segment drift (milliseconds) instead of accumulating
    over the whole recording.
    """
    corr = matched_filter(rec, template)
    g = int(guard * SR)
    head = min(len(corr), nominal[0] + int(max_lead * SR))
    first = int(np.argmax(corr[:head]))
    found, res, scale = [first], [0.0], 1.0
    for k, c in enumerate(nominal[1:], start=1):
        pred = int(first + (c - nominal[0]) * scale)
        lo, hi = max(0, pred - g), min(len(corr), pred + g)
        if hi - lo < 100:
            found.append(pred)
            res.append(0.0)
            continue
        p = lo + int(np.argmax(corr[lo:hi]))
        found.append(p)
        res.append((p - pred) / SR * 1000)
        scale = (p - first) / (c - nominal[0])
    return np.array(found), np.array(res), scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--stimulus", default="rig/stimulus")
    ap.add_argument("--file", default=None, help="stimulus file this recording replays")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--task", default="v2_12")
    ap.add_argument("--gain", type=float, default=0, help="0 = match the cached set's RMS")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.stimulus, "manifest.json")))
    name = a.file or os.path.basename(a.recording).split("__")[-1].replace(".wav", ".wav")
    entry = next((f for f in man["files"] if f["name"] == name), None)
    if entry is None:
        if len(man["files"]) != 1:
            raise SystemExit(f"say which stimulus file this is with --file "
                             f"(one of {[f['name'] for f in man['files']]})")
        entry = man["files"][0]

    rec, sr = sf.read(a.recording, dtype="int32")
    assert sr == SR, f"{sr} Hz recording, expected {SR}"
    if rec.ndim > 1:
        rec = rec[:, 0]
    x = dc_removed(rec.astype(np.float64) / 2**31)

    template = sweep(man["chirp_seconds"], man["chirp_f0"], man["chirp_f1"], SR)
    nominal = [s["chirp_start"] for s in entry["segments"]]
    chirps, resid, scale = find_chirps(x, nominal, template)
    drift_ppm = (scale - 1) * 1e6
    print(f"{len(chirps)} sweeps, clock {drift_ppm:+.0f} ppm, "
          f"residual max {np.abs(resid).max():.1f} ms / rms {np.sqrt((resid**2).mean()):.1f} ms")
    if np.abs(resid).max() > 100:
        print("WARNING: a sweep is more than 100 ms off; the take may have a dropout")

    gap_n = int(man["gap_seconds"] * SR)
    clips, labels, idx, snr = [], [], [], []
    for si, seg in enumerate(entry["segments"]):
        c0, n0 = chirps[si], nominal[si]
        if si + 1 < len(chirps):
            s = (chirps[si + 1] - c0) / (nominal[si + 1] - n0)
        else:
            s = scale
        for cl in seg["clips"]:
            st = int(round(c0 + (cl["start"] - n0) * s))
            if st < 0 or st + SR > len(x):
                continue
            w = x[st : st + SR]
            # noise floor from the middle of the silent gap before this clip
            g0 = st - gap_n + gap_n // 5
            noise = x[max(0, g0) : max(0, g0) + max(1, gap_n * 3 // 5)]
            pn = float((noise**2).mean()) if len(noise) > 10 else 0.0
            ps = float((w**2).mean())
            snr.append(10 * np.log10(max(ps - pn, 1e-12) / max(pn, 1e-12)))
            clips.append(w)
            labels.append(cl["label"])
            idx.append(cl["index"])

    clips = np.stack(clips)
    ref = torch.load(f"{a.cache}/gsc_v{int(a.task[1])}_test12.pt")["audio"].numpy()
    gain = a.gain or (np.sqrt((ref.astype(np.float64) / 32768) ** 2).mean()
                      / max(np.sqrt((clips**2).mean()), 1e-12))
    out = np.clip(clips * gain * 32768, -32768, 32767).astype(np.int16)
    clipped = float((np.abs(clips * gain) >= 0.999).mean())

    meta = {"source": os.path.abspath(a.recording), "stimulus_file": entry["name"],
            "note": a.note, "gain": float(gain), "drift_ppm": float(drift_ppm),
            "residual_ms_max": float(np.abs(resid).max()), "clipped_fraction": clipped,
            "snr_db_median": float(np.median(snr)), "n_clips": int(len(out))}
    torch.save({"audio": torch.from_numpy(out), "label": torch.tensor(labels),
                "index": torch.tensor(idx), "snr_db": torch.tensor(snr, dtype=torch.float32),
                "labels": man["labels"], "meta": meta}, a.out)
    print(f"{a.out}: {len(out)} clips, median SNR {meta['snr_db_median']:.1f} dB, "
          f"gain x{gain:.2f}, clipped {clipped * 100:.3f}%")


if __name__ == "__main__":
    main()
