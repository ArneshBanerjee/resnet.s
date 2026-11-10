"""Convert any image to the 224x224 binary PPM that ./resnet reads.

The steps are the standard torchvision ones for ResNet-18: resize the
shorter side to 256 (bilinear), then center crop to 224. ./resnet does the
normalization itself.

usage: uv run python tools/prep.py image.jpg image.ppm
"""
import sys

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def prep(img, dst):
    img = img.convert("RGB")
    img = F.resize(img, 256, interpolation=InterpolationMode.BILINEAR)
    img = F.center_crop(img, 224)
    with open(dst, "wb") as f:
        f.write(b"P6\n224 224\n255\n" + img.tobytes())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[-1])
    prep(Image.open(sys.argv[1]), sys.argv[2])
