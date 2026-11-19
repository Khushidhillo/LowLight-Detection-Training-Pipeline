"""Dataset preparation, YOLO formatting, and PyTorch data loading for ExDark."""

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import yaml

NAMES = [
    "Bicycle",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cup",
    "Dog",
    "Motorbike",
    "People",
    "Table",
]

SPLITS = {1: "train", 2: "val", 3: "test"}

LIGHTING_NAMES = {
    1: "Low",
    2: "Ambient",
    3: "Object",
    4: "Single",
    5: "Weak",
    6: "Strong",
    7: "Screen",
    8: "Window",
    9: "Shadow",
    10: "Twilight",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(directory: Path) -> List[Path]:
    """Return sorted image files with supported extensions inside ``directory``."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def convert_box(
    left: float,
    top: float,
    width: float,
    height: float,
    iw: int,
    ih: int,
    origin: int,
) -> Tuple[float, float, float, float]:
    """Convert ExDark (left, top, width, height) pixel box to normalized YOLO xywh."""
    x1, y1 = left - origin, top - origin
    x2, y2 = x1 + width, y1 + height
    if width <= 0 or height <= 0 or x1 < 0 or y1 < 0 or x2 > iw or y2 > ih:
        raise ValueError(
            f"Invalid box {(left, top, width, height)} for {(iw, ih)}; inspect coordinate origin"
        )
    return (
        (x1 + x2) / (2 * iw),
        (y1 + y2) / (2 * ih),
        width / iw,
        height / ih,
    )


def prepare(images, annotations, metadata, output, origin: int) -> None:
    """Convert raw ExDark images/annotations/metadata into a clean YOLO dataset."""
    images, annotations, output = map(Path, (images, annotations, output))
    if output.exists():
        raise FileExistsError(f"Use a NEW output folder: {output}")

    image_index: Dict[str, Path] = {}
    for p in images.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if p.name in image_index:
                raise ValueError(f"Duplicate image name: {p.name}")
            image_index[p.name] = p

    ann_index: Dict[str, Path] = {}
    for p in annotations.rglob("*.txt"):
        if p.name in ann_index:
            raise ValueError(f"Duplicate annotation name: {p.name}")
        ann_index[p.name] = p

    records = []
    seen = set()
    hashes: Dict[str, str] = {}

    for line in Path(metadata).read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if (
            not fields
            or fields[0].startswith("#")
            or fields[0].lower() in {"image", "image_name", "image_name,"}
        ):
            continue
        if len(fields) != 5:
            raise ValueError(f"Expected 5 metadata columns: {line}")

        name, _category, lighting, scene, split_id = fields
        split = SPLITS[int(split_id)]
        if name in seen:
            raise ValueError(f"Duplicate metadata row: {name}")
        seen.add(name)

        if name not in image_index:
            raise FileNotFoundError(f"Image not found in {images}: {name}")
        src = image_index[name]

        ann = ann_index.get(name + ".txt", ann_index.get(Path(name).stem + ".txt"))
        if ann is None:
            raise FileNotFoundError(f"No annotation for {name}")

        with Image.open(src) as im:
            im = im.convert("RGB")
            iw, ih = im.size
            digest = hashlib.sha256(im.tobytes() + str(im.size).encode()).hexdigest()

        if digest in hashes and hashes[digest] != split:
            raise ValueError(f"Identical image crosses splits: {name}")
        hashes[digest] = split

        labels = []
        for row in ann.read_text(encoding="utf-8-sig").splitlines():
            vals = row.split()
            if not vals or vals[0].startswith("%") or vals[0].startswith("#"):
                continue
            if vals[0] not in NAMES:
                raise ValueError(f"Unknown class/header in {ann}: {row}")
            box = convert_box(*map(float, vals[1:5]), iw, ih, origin)
            labels.append([NAMES.index(vals[0]), *box])

        records.append(
            {
                "source": str(src),
                "name": Path(name).stem + ".png",
                "split": split,
                "lighting": int(lighting),
                "lighting_name": LIGHTING_NAMES.get(int(lighting), "Unknown"),
                "scene": int(scene),
                "labels": labels,
                "sha256": digest,
            }
        )

    if not records or {r["split"] for r in records} != {"train", "val", "test"}:
        raise ValueError("Metadata must include all three nonempty splits")

    output.mkdir(parents=True)
    for r in records:
        imgdir = output / "images" / r["split"]
        labdir = output / "labels" / r["split"]
        imgdir.mkdir(parents=True, exist_ok=True)
        labdir.mkdir(parents=True, exist_ok=True)
        dst = imgdir / r["name"]
        if dst.exists():
            raise ValueError(f"Output stem collision: {dst}")
        with Image.open(r["source"]) as im:
            im.convert("RGB").save(dst)
        label_lines = [
            f"{v[0]} " + " ".join(f"{x:.9f}" for x in v[1:]) for v in r["labels"]
        ]
        (labdir / Path(r["name"]).with_suffix(".txt")).write_text(
            "\n".join(label_lines), encoding="utf-8"
        )

    write_yaml(output)
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print({s: sum(r["split"] == s for r in records) for s in SPLITS.values()})


def write_yaml(root) -> None:
    """Write Ultralytics-compatible ``data.yaml`` with absolute dataset path."""
    root = Path(root)
    payload = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": NAMES,
    }
    (root / "data.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def image_tensor(path, size: Optional[int] = None):
    """Load an image as a float32 CHW tensor in [0, 1]."""
    import torch

    with Image.open(path) as im:
        im = im.convert("RGB")
        if size:
            im = im.resize((size, size), Image.Resampling.BILINEAR)
        arr = np.array(im, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


class DetectionDataset:
    """YOLO-format dataset for enhancer and joint training.

    Square resize preserves normalized xywh labels without cropping.
    Optional ``augment=True`` applies label-safe random horizontal flipping
    (``x_center -> 1.0 - x_center``), preserving lighting physics while
    doubling effective training diversity.
    """

    def __init__(
        self,
        root,
        split: str = "train",
        size: int = 640,
        augment: bool = False,
        flip_prob: float = 0.5,
    ):
        self.root = Path(root)
        self.split = split
        self.size = size
        self.augment = bool(augment)
        self.flip_prob = float(flip_prob)
        self.files = list_images(self.root / "images" / split)
        if not self.files:
            raise ValueError(f"No images in {root}/images/{split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        import torch

        path = self.files[i]
        label_path = self.root / "labels" / self.split / path.with_suffix(".txt").name
        text = label_path.read_text(encoding="utf-8-sig") if label_path.exists() else ""
        rows = [list(map(float, s.split())) for s in text.splitlines() if s.strip()]
        labels = torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)
        img = image_tensor(path, self.size)

        if self.augment and random.random() < self.flip_prob:
            img = torch.flip(img, dims=[2])
            if labels.numel() > 0:
                labels = labels.clone()
                labels[:, 1] = 1.0 - labels[:, 1]

        return img, labels


def collate(items):
    """Collate variable-length YOLO label tensors into an Ultralytics batch dict."""
    import torch

    images, labels = zip(*items)
    return {
        "img": torch.stack(images),
        "cls": torch.cat([x[:, :1] for x in labels]),
        "bboxes": torch.cat([x[:, 1:] for x in labels]),
        "batch_idx": torch.cat(
            [torch.full((len(x),), i, dtype=torch.float32) for i, x in enumerate(labels)]
        ),
    }
