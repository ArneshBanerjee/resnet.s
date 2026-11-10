"""Run timm resnet18.tv_in1k in PyTorch (fp32) on a PPM and dump the
activations at each debug point, in the same layout as ./resnet -d N
(raw little-endian float32, NHWC).

usage: uv run python tools/ref.py image.ppm [--check]
  Writes ref/<image>/<N>.bin for N = 0..12.
  --check also runs ./resnet -d N for each point and prints the differences.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent.parent
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
POINTS = ["input", "stem", "maxpool",
          "layer1.0", "layer1.1", "layer2.0", "layer2.1",
          "layer3.0", "layer3.1", "layer4.0", "layer4.1",
          "avgpool", "logits"]


def load_model():
    """The same model.safetensors that ./resnet reads."""
    model = timm.create_model("resnet18.tv_in1k", pretrained=False)
    model.load_state_dict(load_file(ROOT / "model.safetensors"))
    return model.eval()


def normalize(ppm):
    """The standard torchvision steps: to_tensor (divide by 255), then normalize."""
    img = Image.open(ppm).convert("RGB")
    return TF.normalize(TF.to_tensor(img), MEAN, STD).unsqueeze(0)


def nhwc(t):
    t = t[0]
    return (t.permute(1, 2, 0) if t.dim() == 3 else t).contiguous().numpy()


def activations(model, x):
    """All 13 debug points, as NHWC numpy arrays."""
    outs = [x]
    mods = [model.act1, model.maxpool, *model.layer1, *model.layer2,
            *model.layer3, *model.layer4, model.global_pool, model.fc]
    hooks = [m.register_forward_hook(lambda m, i, o: outs.append(o.detach().clone()))
             for m in mods]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return [nhwc(o) for o in outs]


def check(ppm, acts):
    for n, ref in enumerate(acts):
        r = subprocess.run([ROOT / "resnet", "-d", str(n), ppm], capture_output=True, cwd=ROOT)
        got = np.frombuffer(r.stdout, np.float32)
        if r.returncode != 0 or got.size != ref.size:
            print(f"{n:2d} {POINTS[n]:9s} no match: exit {r.returncode}, "
                  f"{got.size} floats, want {ref.size} {r.stderr.decode().strip()}")
            continue
        diff = np.abs(got - ref.ravel()).max()
        print(f"{n:2d} {POINTS[n]:9s} max abs diff {diff:.2e}   "
              f"(max abs value {np.abs(ref).max():.2e})")


def main():
    args = [a for a in sys.argv[1:] if a != "--check"]
    if len(args) != 1:
        sys.exit("usage: uv run python tools/ref.py image.ppm [--check]")
    ppm = Path(args[0]).resolve()
    acts = activations(load_model(), normalize(ppm))
    out = ROOT / "ref" / ppm.stem
    out.mkdir(parents=True, exist_ok=True)
    for n, a in enumerate(acts):
        a.astype(np.float32).tofile(out / f"{n}.bin")
    if "--check" in sys.argv:
        check(ppm, acts)


if __name__ == "__main__":
    main()
