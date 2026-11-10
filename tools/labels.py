"""Write labels.txt: the 1000 ImageNet class names, one per line, in index order.

The names come from timm (for example "tench, Tinca tinca").

usage: uv run python tools/labels.py
"""
from pathlib import Path

from timm.data import ImageNetInfo

info = ImageNetInfo()
names = [info.index_to_description(i) for i in range(1000)]
out = Path(__file__).resolve().parent.parent / "labels.txt"
out.write_text("".join(n + "\n" for n in names))
print(f"{len(names)} labels in {out}")
