"""Record the ESP32-S3 I2S stream to a 32-bit WAV.

Reads the framed block stream produced by rig/firmware/i2s_stream, checks the
sequence numbers, and writes the samples unmodified (24 valid bits in the top of
each int32). Dropped blocks are filled with zeros so the file stays sample-exact
in time -- alignment in align.py depends on that -- and are listed in the sidecar
<out>.json together with the measured level, so a bad take is obvious before the
microphone is moved.

usage: python rig/capture.py out/quiet_1m.wav --port /dev/ttyACM0 --seconds 600
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import serial
import soundfile as sf

SR = 16000
BLOCK = 256
MAGIC = 0xA5A5A5A5
HDR = struct.Struct("<II")
FRAME = HDR.size + BLOCK * 4


def resync(ser):
    """Byte-align to the next magic word."""
    window = b""
    while len(window) < 4 or struct.unpack("<I", window[-4:])[0] != MAGIC:
        b = ser.read(1)
        if not b:
            raise RuntimeError("no data from the serial port; is the firmware running?")
        window = (window + b)[-4:]
    return struct.unpack("<I", ser.read(4))[0]  # the sequence number that follows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--port", default="/dev/ttyUSB0", help="ESP32 via USB-UART bridge; an ESP32-S3 native-USB port is /dev/ttyACM0")
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--note", default="", help="condition description stored in the sidecar")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ser = serial.Serial(a.port, 2000000, timeout=2)
    ser.reset_input_buffer()

    want = int(a.seconds * SR / BLOCK)
    blocks, drops = [], []
    expect = resync(ser)
    t0 = time.time()
    for i in range(want):
        raw = ser.read(FRAME)
        if len(raw) < FRAME:
            raise RuntimeError(f"short read after {len(blocks) * BLOCK / SR:.1f} s")
        magic, seq = HDR.unpack_from(raw)
        if magic != MAGIC:
            expect = resync(ser)
            drops.append({"at_block": len(blocks), "reason": "desync"})
            continue
        gap = (seq - expect) & 0xFFFFFFFF
        if gap:
            # keep the timeline honest: insert the blocks the device sent but we lost
            drops.append({"at_block": len(blocks), "lost_blocks": gap})
            blocks.extend([np.zeros(BLOCK, np.int32)] * min(gap, want - len(blocks)))
        blocks.append(np.frombuffer(raw, np.int32, BLOCK, HDR.size).copy())
        expect = seq + 1
        if i % (SR // BLOCK * 10) == 0:
            el = time.time() - t0
            print(f"\r{len(blocks) * BLOCK / SR:7.1f} s  ({el:.0f} s wall, {len(drops)} drops)",
                  end="", file=sys.stderr, flush=True)
    ser.close()

    x = np.concatenate(blocks)[: want * BLOCK]
    sf.write(a.out, x, SR, subtype="PCM_32")

    v = x.astype(np.float64) / 2**31
    meta = {
        "seconds": len(x) / SR,
        "note": a.note,
        "dropped_blocks": sum(d.get("lost_blocks", 0) for d in drops),
        "drops": drops[:200],
        "peak_dbfs": float(20 * np.log10(max(np.abs(v).max(), 1e-12))),
        "rms_dbfs": float(20 * np.log10(max(np.sqrt((v**2).mean()), 1e-12))),
        "dc_offset": float(v.mean()),
    }
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n{a.out}: {meta['seconds']:.1f} s, peak {meta['peak_dbfs']:.1f} dBFS, "
          f"rms {meta['rms_dbfs']:.1f} dBFS, {meta['dropped_blocks']} blocks lost")
    if meta["peak_dbfs"] > -1:
        print("WARNING: clipping; move the speaker back or lower its volume")
    if meta["peak_dbfs"] < -40:
        print("WARNING: very quiet; check the L/R pin is grounded and the mic is wired to GPIO 6")


if __name__ == "__main__":
    main()
