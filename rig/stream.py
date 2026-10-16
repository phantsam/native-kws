"""Framed block reader for the ESP32 I2S stream.

The firmware tags every 256-sample block with a magic word and a sequence number.
Reading through this class means a dropped block is reported and zero-filled rather
than silently shifting every later sample in time, which is what alignment and
labelling depend on.

Needs only pyserial and numpy, so it runs on the laptop the board is plugged into.
"""
import struct

import numpy as np
import serial

SR = 16000
BLOCK = 256
MAGIC = 0xA5A5A5A5
HDR = struct.Struct("<II")
FRAME = HDR.size + BLOCK * 4


class Stream:
    def __init__(self, port="/dev/ttyUSB0", baud=2000000, timeout=2.0):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.ser.reset_input_buffer()
        self.expect = None
        self.lost = 0

    def resync(self):
        """Scan to the next magic word and return that whole block.

        Returns a complete block, not just the header: the reader must end up on a
        frame boundary, or every later read is offset by the 8 header bytes.
        """
        window = b""
        while len(window) < 4 or struct.unpack("<I", window[-4:])[0] != MAGIC:
            b = self._read(1)
            window = (window + b)[-4:]
        seq = struct.unpack("<I", self._read(4))[0]
        payload = self._read(BLOCK * 4)
        self.expect = seq + 1
        return np.frombuffer(payload, np.int32).copy()

    def _read(self, n):
        try:
            raw = self.ser.read(n)
        except serial.SerialException as e:
            raise RuntimeError(f"serial read failed ({e}); is the board still plugged in?")
        if len(raw) < n:
            raise RuntimeError("the device stopped sending; check the cable and the firmware")
        return raw

    def block(self):
        """Next block as (samples, lost_before). Lost blocks are returned as zeros."""
        if self.expect is None:
            return self.resync(), 0
        raw = self._read(FRAME)
        magic, seq = HDR.unpack_from(raw)
        if magic != MAGIC:
            return self.resync(), 1
        gap = (seq - self.expect) & 0xFFFFFFFF
        if gap > 1 << 20:  # the device restarted; treat as a fresh start
            gap = 0
        self.expect = seq + 1
        self.lost += gap
        return np.frombuffer(raw, np.int32, BLOCK, HDR.size).copy(), gap

    def read(self, seconds, progress=None):
        """Read `seconds` of audio as float64 in [-1, 1), with dropped blocks zero-filled."""
        want = max(1, int(seconds * SR / BLOCK))
        out, drops = [], []
        while len(out) < want:
            samples, gap = self.block()
            if gap:
                drops.append({"at_block": len(out), "lost_blocks": int(gap)})
                out.extend([np.zeros(BLOCK, np.int32)] * min(gap, want - len(out)))
            if len(out) < want:
                out.append(samples)
            if progress and len(out) % 125 == 0:
                progress(len(out) * BLOCK / SR)
        return np.concatenate(out)[: want * BLOCK].astype(np.float64) / 2**31, drops

    def flush(self):
        self.ser.reset_input_buffer()
        self.expect = None

    def close(self):
        self.ser.close()


def dbfs(x):
    return 20 * np.log10(max(float(np.abs(x).max()), 1e-12))
