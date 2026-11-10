"""Check ./resnet against timm resnet18.tv_in1k in PyTorch (fp32).

For each image in images/*.ppm:
1. The top-5 class indices from ./resnet -d 12 (the logits) must match
   PyTorch exactly, in the same order. The largest logit difference is shown.
2. The 5 lines that ./resnet prints must name the same classes, with the
   same probabilities to 0.01%.

usage: uv run python tools/verify.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ref import ROOT, load_model, normalize

TOL_PCT = 0.01


def check(model, labels, ppm):
    with torch.no_grad():
        ref = model(normalize(ppm))[0]
    ref_top = torch.topk(ref, 5).indices.tolist()
    ref_pct = torch.softmax(ref, 0) * 100

    r = subprocess.run([ROOT / "resnet", "-d", "12", ppm], capture_output=True, check=True, cwd=ROOT)
    got = np.frombuffer(r.stdout, np.float32)
    got_top = np.argsort(-got, kind="stable")[:5].tolist()
    diff = np.abs(got - ref.numpy()).max()

    out = subprocess.run([ROOT / "resnet", ppm], capture_output=True, check=True, text=True, cwd=ROOT)
    lines = out.stdout.splitlines()
    ok = got_top == ref_top and len(lines) == 5
    for line, i in zip(lines, ref_top):
        pct, name = line.split("%  ", 1)
        ok &= name == labels[i] and abs(float(pct) - ref_pct[i].item()) <= TOL_PCT
    return ok, ref_top, diff


def main():
    model = load_model()
    labels = (ROOT / "labels.txt").read_text().splitlines()
    images = sorted((ROOT / "images").glob("*.ppm"))
    if not images:
        sys.exit("no images: run uv run python tools/fetch_images.py")
    fails = 0
    worst = 0.0
    for ppm in images:
        ok, top, diff = check(model, labels, ppm)
        fails += not ok
        worst = max(worst, diff)
        print(f"{'PASS' if ok else 'FAIL'}  {ppm.name:22s} top-5 {top}  "
              f"max logit diff {diff:.2e}  ({labels[top[0]]})")
    print(f"\nlargest logit difference: {worst:.2e}")
    print("ALL PASS" if fails == 0 else f"{fails} of {len(images)} FAIL")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
