"""Record real speakers saying the keywords, through the INMP441.

This is set B of docs/DATASET.md: the realism check the replay set cannot give you,
because it has real vocal effort, real distance and real room. It is deliberately
NOT a benchmark -- a handful of speakers has no statistical power, and any number
from it must be reported with the speaker count attached.

Prompts one word at a time in a fixed-seed random order (so the speaker cannot fall
into a rhythm, and so any drift in their voice does not line up with one word),
trims each recording to the 1 s window the model expects, and rejects takes that
were silent or clipped so they can be repeated on the spot.

Runs on the laptop the board is plugged into: needs pyserial, numpy and soundfile,
not torch. Clips are written as WAVs plus an index; make_set.py turns a session
into a cache file on the training machine.

usage: python rig/record_live.py --speaker s01 --distance 1.0 --reps 10
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf

from rig.stream import SR, Stream, dbfs

KEYWORDS10 = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
# Median per-clip RMS of the cached Speech Commands v2 keyword test clips, so live
# takes land at a comparable loudness instead of whatever the room happened to give.
TARGET_RMS = 0.0641


def trim(x, out_len=SR, frame=320):
    """Centre a 1 s window on the spoken word using short-time energy."""
    n = len(x) // frame * frame
    e = (x[:n] ** 2).reshape(-1, frame).mean(1)
    e = np.convolve(e, np.ones(5) / 5, mode="same")
    if e.max() <= 0:
        return x[:out_len], -np.inf
    above = np.flatnonzero(e > 0.1 * e.max())
    centre = int((above[0] + above[-1] + 1) / 2 * frame)
    start = int(np.clip(centre - out_len // 2, 0, max(0, len(x) - out_len)))
    w = x[start : start + out_len]
    if len(w) < out_len:
        w = np.pad(w, (0, out_len - len(w)))
    return w, float(10 * np.log10(e.max() / max(e.min(), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True, help="anonymous id, e.g. s01 -- not a name")
    ap.add_argument("--distance", type=float, required=True, help="metres from the microphone")
    ap.add_argument("--out", default="rig/out/live")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--words", nargs="+", default=KEYWORDS10)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--window", type=float, default=2.0, help="recording window per prompt (s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    session = os.path.join(a.out, f"{a.speaker}_{a.distance:g}m")
    os.makedirs(session, exist_ok=True)
    st = Stream(a.port)

    print("Recording 5 s of room noise -- stay quiet.")
    noise, _ = st.read(5.0)
    sf.write(os.path.join(session, "_noise.wav"), noise, SR, subtype="PCM_24")
    noise_rms = float(np.sqrt((noise**2).mean()))
    print(f"  noise floor {20 * np.log10(max(noise_rms, 1e-12)):.1f} dBFS, "
          f"peak {dbfs(noise):.1f} dBFS\n")
    if dbfs(noise) > -3:
        print("  WARNING: the room is very loud or the mic is clipping\n")

    prompts = [(w, r) for w in a.words for r in range(a.reps)]
    np.random.default_rng(a.seed).shuffle(prompts)
    index, done = [], 0
    print(f"{len(prompts)} prompts. Speak when you see GO. Ctrl-C to stop early.\n")

    try:
        for i, (word, rep) in enumerate(prompts, 1):
            while True:
                print(f"[{i:3d}/{len(prompts)}]  say:  {word.upper():6s}", end="", flush=True)
                st.flush()
                time.sleep(0.25)
                print("   GO", end="", flush=True)
                x, drops = st.read(a.window)
                x = x - x.mean()
                w, contrast = trim(x)
                peak = dbfs(w)
                if peak < -45:
                    print(f"   nothing heard ({peak:.0f} dBFS) -- again")
                    continue
                if peak > -1:
                    print(f"   clipped ({peak:.0f} dBFS) -- move back, again")
                    continue
                if drops:
                    print("   dropped blocks -- again")
                    continue
                rms = float(np.sqrt((w**2).mean()))
                fn = f"{word}_{rep:02d}.wav"
                sf.write(os.path.join(session, fn), w, SR, subtype="PCM_24")
                index.append({"file": fn, "word": word, "rep": rep,
                              "peak_dbfs": round(peak, 1), "rms": rms,
                              "snr_db": round(float(20 * np.log10(rms / max(noise_rms, 1e-12))), 1),
                              "contrast_db": round(contrast, 1)})
                done += 1
                print(f"   ok  {peak:5.0f} dBFS  SNR {index[-1]['snr_db']:5.1f} dB")
                break
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        st.close()

    meta = {"speaker": a.speaker, "distance_m": a.distance, "note": a.note,
            "noise_rms": noise_rms, "target_rms": TARGET_RMS, "sr": SR,
            "recorded": done, "planned": len(prompts),
            "words": a.words, "clips": index}
    with open(os.path.join(session, "index.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if index:
        snrs = [c["snr_db"] for c in index]
        print(f"\n{session}: {done} clips, median SNR {np.median(snrs):.1f} dB")
    print("Next: copy the session folder to the training machine and run rig/make_set.py")


if __name__ == "__main__":
    main()
