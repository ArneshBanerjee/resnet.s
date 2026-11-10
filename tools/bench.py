"""Time ./resnet against PyTorch on the CPU (fp32, one image at a time).

./resnet -b N runs the forward pass N times in one process. PyTorch runs it
N times after a warm-up, with 1 thread and with its default thread count.
Each result is the median over 3 images.

usage: uv run python tools/bench.py [N]
"""
import statistics
import subprocess
import sys
import time

import torch

from ref import ROOT, load_model, normalize

IMAGES = ["n01440764_0.ppm", "n03417042_2.ppm", "n03888257_0.ppm"]


def resnet_ms(ppm, n):
    r = subprocess.run([ROOT / "resnet", "-b", str(n), ppm], capture_output=True,
                       text=True, check=True, cwd=ROOT)
    # [resnet.s] load 12.00 ms, forward 49.90 47.89 ... ms
    load, fwd = r.stderr.split("load")[1].split("ms, forward")
    return float(load), [float(t) for t in fwd.split()[:-1]]


def torch_ms(model, x, n, threads):
    torch.set_num_threads(threads)
    times = []
    with torch.inference_mode():
        for _ in range(5):
            model(x)
        for _ in range(n):
            t = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t) * 1000)
    return times


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    default_threads = torch.get_num_threads()
    model = load_model()
    rows = {"resnet.s (1 thread)": [], "PyTorch, 1 thread": [],
            f"PyTorch, {default_threads} threads (default)": []}
    loads, firsts = [], []
    for name in IMAGES:
        ppm = ROOT / "images" / name
        load, fwd = resnet_ms(ppm, n)
        loads.append(load)
        firsts.append(fwd[0])
        rows["resnet.s (1 thread)"] += fwd
        x = normalize(ppm)
        rows["PyTorch, 1 thread"] += torch_ms(model, x, n, 1)
        rows[f"PyTorch, {default_threads} threads (default)"] += torch_ms(model, x, n, default_threads)
    print(f"median of {n} runs x {len(IMAGES)} images, forward pass only")
    for k, v in rows.items():
        print(f"  {k:32s} {statistics.median(v):7.2f} ms   (min {min(v):.2f})")
    print(f"resnet.s load (parse, BN fold, repack): median {statistics.median(loads):.2f} ms")
    print(f"resnet.s first forward pass in a new process: median {statistics.median(firsts):.2f} ms")


if __name__ == "__main__":
    main()
