"""Turn recorded live sessions into a cache file the evaluation pipeline can read.

Applies one gain per session so the clips sit at the loudness of the cached
Speech Commands clips -- per-session, not per-clip, because the level differences
*within* a session (distance, vocal effort) are part of what is being measured.

usage: python rig/make_set.py rig/out/live/s01_1m [more sessions...] --out cache/real_live.pt
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    audio, labels, speakers, dists, snrs, skipped = [], [], [], [], [], []
    for s in a.sessions:
        meta = json.load(open(os.path.join(s, "index.json")))
        gain = meta["target_rms"] / max(
            np.median([c["rms"] for c in meta["clips"]]) if meta["clips"] else 1.0, 1e-9)
        for c in meta["clips"]:
            if c["word"] not in LABELS12:
                skipped.append(c["word"])
                continue
            x, sr = sf.read(os.path.join(s, c["file"]), dtype="float32")
            assert sr == SR, f"{c['file']}: {sr} Hz"
            w = np.zeros(SR, np.float32)
            w[: min(len(x), SR)] = x[:SR]
            audio.append(np.clip(w * gain * 32768, -32768, 32767).astype(np.int16))
            labels.append(LABELS12.index(c["word"]))
            speakers.append(meta["speaker"])
            dists.append(meta["distance_m"])
            snrs.append(c["snr_db"])
        print(f"{s}: {len(meta['clips'])} clips, speaker {meta['speaker']}, "
              f"{meta['distance_m']} m, gain x{gain:.2f}")

    uniq = sorted(set(speakers))
    torch.save({"audio": torch.from_numpy(np.stack(audio)),
                "label": torch.tensor(labels),
                "speaker": speakers, "distance_m": torch.tensor(dists),
                "snr_db": torch.tensor(snrs, dtype=torch.float32),
                "labels": LABELS12,
                "meta": {"kind": "live", "sessions": a.sessions, "speakers": uniq,
                         "n_speakers": len(uniq)}}, a.out)
    print(f"\n{a.out}: {len(audio)} clips, {len(uniq)} speaker(s) {uniq}")
    if skipped:
        print(f"skipped words outside the 12-class set: {sorted(set(skipped))}")
    print(f"REPORT THE SPEAKER COUNT ({len(uniq)}) with any number from this set.")


if __name__ == "__main__":
    main()
