"""Download 30 test images (3 per class) from the Imagenette validation set.

Imagenette is a 10-class subset of ImageNet. The script streams the
imagenette2-320 archive (about 340 MB, not kept on disk) and writes each
picked image as a 224x224 PPM with tools/prep.py to images/<class>_<n>.ppm.

usage: uv run python tools/fetch_images.py
"""
import io
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image

from prep import prep

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
PER_CLASS = 3
N_CLASSES = 10


def main():
    out = Path(__file__).resolve().parent.parent / "images"
    out.mkdir(exist_ok=True)
    count = {}
    with urllib.request.urlopen(URL) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            # names look like imagenette2-320/val/n01440764/ILSVRC2012_val_00009111.JPEG
            parts = m.name.split("/")
            if not m.isfile() or len(parts) != 4 or parts[1] != "val":
                continue
            wnid = parts[2]
            n = count.get(wnid, 0)
            if n == PER_CLASS:
                continue
            img = Image.open(io.BytesIO(tar.extractfile(m).read()))
            dst = out / f"{wnid}_{n}.ppm"
            prep(img, dst)
            count[wnid] = n + 1
            print(dst.relative_to(out.parent))
            if len(count) == N_CLASSES and all(c == PER_CLASS for c in count.values()):
                break
    print(f"{sum(count.values())} images in {out}")


if __name__ == "__main__":
    main()
