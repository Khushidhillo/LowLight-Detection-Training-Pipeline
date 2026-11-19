"""Visual label inspection tool to verify bounding boxes before training."""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from lowlight.data import NAMES, list_images


def main():
    p = argparse.ArgumentParser(description="Render ground-truth YOLO boxes for visual inspection")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--count", type=int, default=24)
    a = p.parse_args()

    root = Path(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for path in list_images(root / "images" / a.split)[: a.count]:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        draw = ImageDraw.Draw(im)
        label_file = root / "labels" / a.split / path.with_suffix(".txt").name
        if label_file.exists():
            for row in label_file.read_text(encoding="utf-8-sig").splitlines():
                if not row.strip():
                    continue
                c, x, y, bw, bh = map(float, row.split())
                box = (
                    (x - bw / 2) * w,
                    (y - bh / 2) * h,
                    (x + bw / 2) * w,
                    (y + bh / 2) * h,
                )
                draw.rectangle(box, outline="lime", width=2)
                draw.text(box[:2], NAMES[int(c)], fill="yellow")
        im.save(out / path.with_suffix(".png").name)

    print(out.resolve())


if __name__ == "__main__":
    main()
