"""Record the ESP32 I2S stream to a 32-bit WAV.

Reads the framed block stream produced by rig/firmware/i2s_stream, checks the
sequence numbers, and writes the samples unmodified (24 valid bits in the top of
each int32). Dropped blocks are filled with zeros so the file stays sample-exact
in time -- alignment in align.py depends on that -- and are listed in the sidecar
<out>.json together with the measured level, so a bad take is obvious before the
microphone is moved.

usage: python rig/capture.py rig/out/quiet_1m.wav --port /dev/ttyUSB0 --seconds 600
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

from rig.stream import SR, Stream

SUBTYPE = "PCM_32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--port", default="/dev/ttyUSB0",
                    help="ESP32 via USB-UART bridge; an ESP32-S3 native-USB port is /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=2000000)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--note", default="", help="condition description stored in the sidecar")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    st = Stream(a.port, a.baud)
    t0 = time.time()

    def show(done):
        print(f"\r{done:7.1f} / {a.seconds:.0f} s  ({time.time() - t0:.0f} s wall, "
              f"{st.lost} blocks lost)", end="", file=sys.stderr, flush=True)

    try:
        x, drops = st.read(a.seconds, progress=show)
    finally:
        st.close()

    sf.write(a.out, (x * 2**31).astype(np.int32), SR, subtype=SUBTYPE)
    meta = {
        "seconds": len(x) / SR,
        "note": a.note,
        "dropped_blocks": int(sum(d["lost_blocks"] for d in drops)),
        "drops": drops[:200],
        "peak_dbfs": float(20 * np.log10(max(np.abs(x).max(), 1e-12))),
        "rms_dbfs": float(20 * np.log10(max(np.sqrt((x**2).mean()), 1e-12))),
        "dc_offset": float(x.mean()),
    }
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n{a.out}: {meta['seconds']:.1f} s, peak {meta['peak_dbfs']:.1f} dBFS, "
          f"rms {meta['rms_dbfs']:.1f} dBFS, {meta['dropped_blocks']} blocks lost")
    if meta["peak_dbfs"] > -1:
        print("WARNING: clipping; move the speaker back or lower its volume")
    if meta["peak_dbfs"] < -40:
        print("WARNING: very quiet; check L/R is grounded and SD is on the right GPIO")
    if meta["dropped_blocks"]:
        print("WARNING: blocks were lost; lower --baud or close other serial programs")


if __name__ == "__main__":
    main()
