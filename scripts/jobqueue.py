"""Minimal GPU job queue for training runs.

Each non-empty, non-# line of the jobs file is:  <run_name> <kws.train arguments ...>
(`--name <run_name>` is added automatically). A job is skipped if runs/<name>/summary.json
exists or a process with that run name is already running. Up to `slots` training processes
run per GPU; the count includes runs started outside the queue (read from /proc environ).
The jobs file is re-read on every poll, so jobs can be appended while the queue runs.

usage: python scripts/jobqueue.py jobs/wave2.txt --slots 0:4 1:4 [--poll 60]
"""
import argparse
import os
import subprocess
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    line = f"{datetime.now():%F %T} {msg}"
    print(line, flush=True)
    with open(os.path.join(ROOT, "runs/logs/queue.log"), "a") as f:
        f.write(line + "\n")


def running_trainers():
    """{run_name: gpu} for every live `kws.train` process of this user."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            if b"kws.train" not in cmd:
                continue
            env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
        except (OSError, PermissionError):
            continue
        args = [c.decode() for c in cmd if c]
        name = args[args.index("--name") + 1] if "--name" in args else f"pid{pid}"
        gpu = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"CUDA_VISIBLE_DEVICES=")), "?")
        out[name] = gpu
    return out


def free_memory_gb():
    """{gpu_index: free GiB} from nvidia-smi (includes other users' usage)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {line.split(",")[0].strip(): int(line.split(",")[1]) / 1024 for line in out.strip().splitlines()}


def teachers_ready(args):
    """A job with --teacher dirs waits until every teacher has finished (last_ema.pt exists)."""
    if "--teacher" not in args:
        return True
    i = args.index("--teacher") + 1
    while i < len(args) and not args[i].startswith("--"):
        if not os.path.exists(os.path.join(ROOT, args[i], "last_ema.pt")):
            return False
        i += 1
    return True


def read_jobs(path):
    jobs = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            name, *args = line.split()
            jobs.append((name, args))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--slots", nargs="+", required=True, help="gpu:max_concurrent, e.g. 0:4 1:4")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--min_free_gb", type=float, default=32, help="launch on a GPU only if this much memory is free")
    a = ap.parse_args()
    slots = {s.split(":")[0]: int(s.split(":")[1]) for s in a.slots}
    launched = set()
    log(f"queue start: {a.jobs} slots={slots}")
    while True:
        running = running_trainers()
        jobs = read_jobs(a.jobs)
        file_jobs_running = any(n in running for n, _ in jobs)  # includes jobs started by an earlier queue
        todo = [(n, args) for n, args in jobs
                if not os.path.exists(os.path.join(ROOT, "runs", n, "summary.json"))
                and not os.path.exists(os.path.join(ROOT, "runs", n, "STOPPED"))
                and n not in running and n not in launched]  # a launched job that died is not retried
        ready = [(n, args) for n, args in todo if teachers_ready(args)]
        blocked = [n for n, args in todo if not teachers_ready(args)]
        if not todo and not file_jobs_running:
            log("queue empty, exiting")
            return
        if blocked and not ready and not file_jobs_running:
            log(f"only jobs with unfinished teachers remain and nothing is running: {blocked}; exiting")
            return
        mem = free_memory_gb()
        for name, args in ready:
            load = {g: sum(1 for v in running.values() if v == g) for g in slots}
            free = [g for g in slots if load[g] < slots[g] and mem.get(g, 0) >= a.min_free_gb]
            if not free:
                break
            gpu = min(free, key=lambda g: load[g] / slots[g])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            logf = open(os.path.join(ROOT, "runs/logs", f"{name}.log"), "a")
            subprocess.Popen([os.path.join(ROOT, ".venv/bin/python"), "-m", "kws.train", *args, "--resume", "--name", name],
                             cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
            launched.add(name)
            running[name] = gpu
            mem[gpu] = mem.get(gpu, 0) - a.min_free_gb  # reserve until the new job allocates
            log(f"launched {name} on GPU {gpu}: {' '.join(args)}")
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
