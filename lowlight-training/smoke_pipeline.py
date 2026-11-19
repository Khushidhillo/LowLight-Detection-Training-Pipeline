"""Small offline integration run. Output is NOT a trained real-world model."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image, ImageDraw

from lowlight.data import write_yaml
from lowlight.engine import (
    baseline,
    benchmark,
    evaluate,
    export_images,
    predict,
    train_enhancer,
)


def main():
    torch.set_num_threads(1)
    root = Path("smoke_artifacts/data")
    for si, split in enumerate(["train", "val", "test"]):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(4):
            im = Image.new("RGB", (64, 64), (10 + si + i, 10, 10))
            ImageDraw.Draw(im).rectangle((16, 16, 48, 48), fill=(70 + i, 50, 30))
            im.save(root / "images" / split / f"{i}.png")
            (root / "labels" / split / f"{i}.txt").write_text("4 0.5 0.5 0.5 0.5\n")
    write_yaml(root)

    baseline(
        SimpleNamespace(
            weights="yolov8n.yaml",
            data=str(root),
            out="smoke_artifacts/baseline",
            epochs=1,
            batch=2,
            size=64,
            device="cpu",
            seed=42,
        )
    )

    common = dict(
        data=str(root),
        detector="smoke_artifacts/baseline/weights/best.pt",
        epochs=1,
        batch=2,
        size=64,
        device="cpu",
        seed=42,
        lr=1e-4,
        det_weight=0.1,
        val_every=1,
        resume=None,
    )
    train_enhancer(SimpleNamespace(**common, command="pretrain", init=None, out="smoke_artifacts/enhancer"))
    train_enhancer(
        SimpleNamespace(
            **common,
            command="guided",
            init="smoke_artifacts/enhancer/best.pt",
            out="smoke_artifacts/guided",
        )
    )
    train_enhancer(
        SimpleNamespace(
            **common,
            command="joint",
            init="smoke_artifacts/enhancer/best.pt",
            out="smoke_artifacts/joint",
            det_lr=1e-5,
        )
    )

    # Export guided and joint test splits using their respective enhancers
    export_images(
        root,
        "smoke_artifacts/enhanced_guided",
        "learned",
        "smoke_artifacts/guided/best.pt",
        splits=("test",),
    )
    export_images(
        root,
        "smoke_artifacts/enhanced_joint",
        "learned",
        "smoke_artifacts/joint/best.pt",
        splits=("test",),
    )

    evaluate(common["detector"], "smoke_artifacts/enhanced_guided", "smoke_artifacts/final_guided", split="test", size=64)
    evaluate(
        "smoke_artifacts/joint/detector_best.pt",
        "smoke_artifacts/enhanced_joint",
        "smoke_artifacts/final_joint",
        split="test",
        size=64,
    )

    predict(
        SimpleNamespace(
            image=str(root / "images" / "test" / "0.png"),
            enhancer="smoke_artifacts/guided/best.pt",
            detector=common["detector"],
            out="smoke_artifacts/predict_demo",
            device="cpu",
            size=64,
            hq=True,
        )
    )

    benchmark(
        SimpleNamespace(
            enhancer="smoke_artifacts/guided/best.pt",
            detector=common["detector"],
            out="smoke_artifacts/benchmark.json",
            device="cpu",
            size=64,
            runs=3,
            warmup=1,
        )
    )
    print("SMOKE PIPELINE PASSED; synthetic metrics have no research meaning.")


if __name__ == "__main__":
    main()
