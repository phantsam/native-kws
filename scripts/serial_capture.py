"""Reset the board and capture what it prints, for firmware that just runs and reports
(deploy/esp32_kws/main/app_main.cpp). Replaces `idf.py monitor`, which needs a terminal.

DTR and RTS are released before the port opens and RTS is pulsed, which on the usual
auto-reset circuit taps EN with IO0 high: a normal boot, not download mode.

  python scripts/serial_capture.py --port COM5 --seconds 15 --out device_output.txt
"""
import argparse
import re
import time

import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    s = serial.Serial()
    s.port, s.baudrate, s.timeout = a.port, a.baud, 0.2
    s.dtr = False
    s.rts = False
    s.open()
    s.rts = True
    time.sleep(0.1)
    s.rts = False
    buf = bytearray()
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        buf += s.read(4096)
    s.close()
    text = re.sub(r"\x1b\[[0-9;]*m", "", buf.decode("ascii", "replace")).replace("\r", "")
    open(a.out, "w", encoding="utf-8").write(text)
    print(f"{len(buf)} bytes -> {a.out}")


if __name__ == "__main__":
    main()
