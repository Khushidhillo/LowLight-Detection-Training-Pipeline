"""Command-line interface for the ExDark low-light detection training pipeline."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ExDark low-light detection training pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    # prepare
    c = sub.add_parser("prepare", help="Convert raw ExDark dataset into YOLO format")
    for key in ["images", "annotations", "metadata", "out"]:
        c.add_argument("--" + key, required=True)
    c.add_argument(
        "--origin",
        type=int,
        choices=[0, 1],
        required=True,
        help="Box coordinate origin; inspect native annotations",
    )

    # baseline, pretrain, guided, joint
    for command in ["baseline", "pretrain", "guided", "joint"]:
        c = sub.add_parser(command, help=f"Run {command} training stage")
        c.add_argument("--data", required=True)
        c.add_argument("--out", required=True)
        c.add_argument("--epochs", type=int, default=50)
        c.add_argument("--batch", type=int, default=4)
        c.add_argument("--size", type=int, default=640)
        c.add_argument("--device", default="cpu")
        c.add_argument("--seed", type=int, default=42)
        c.add_argument("--workers", type=int, default=0)
        c.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision")

        if command == "baseline":
            c.add_argument("--weights", default="yolov8n.pt")
        else:
            c.add_argument("--detector", required=True)
            c.add_argument("--init")
            c.add_argument("--resume")
            c.add_argument("--lr", type=float, default=1e-4)
            c.add_argument(
                "--det-lr",
                type=float,
                default=None,
                help="Detector learning rate in joint fine-tuning mode",
            )
            c.add_argument("--det-weight", type=float, default=0.1)
            c.add_argument("--val-every", type=int, default=5)
            c.add_argument("--val-batch", type=int, default=None, help="Validation batch size")
            c.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
            c.add_argument(
                "--scheduler",
                choices=["cosine", "none"],
                default="cosine",
                help="Learning rate schedule",
            )
            c.add_argument("--warmup-epochs", type=int, default=0, help="Linear LR warmup epochs")
            c.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay")
            c.add_argument(
                "--augment",
                action="store_true",
                help="Enable label-safe random horizontal flip augmentation",
            )
            if command == "joint":
                c.add_argument(
                    "--no-freeze-bn",
                    dest="freeze_bn",
                    action="store_false",
                    default=True,
                    help="Allow detector BatchNorm running stats to update in joint mode",
                )
            # Enhancer architecture & loss weights
            c.add_argument("--width", type=int, default=32, help="CurveEnhancer channel width")
            c.add_argument("--iterations", type=int, default=8, help="Curve enhancement iterations")
            c.add_argument(
                "--scale-factor",
                type=int,
                default=None,
                help="Zero-DCE++ curve estimation downsample factor (e.g. 1, 4, 12)",
            )
            c.add_argument("--exposure", type=float, default=0.6, help="Target patch exposure")
            c.add_argument("--w-spatial", type=float, default=1.0)
            c.add_argument("--w-exp", type=float, default=10.0)
            c.add_argument("--w-color", type=float, default=5.0)
            c.add_argument("--w-tv", type=float, default=200.0)

    # export
    c = sub.add_parser("export", help="Export enhanced dataset split images")
    c.add_argument("--data", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--mode", choices=["learned", "hq", "gamma", "clahe"], default="learned")
    c.add_argument("--enhancer")
    c.add_argument("--device", default="cpu")
    c.add_argument("--scale-factor", type=int, default=None)
    c.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"])

    # evaluate
    c = sub.add_parser("evaluate", help="Evaluate YOLO detector on val or test split")
    c.add_argument("--data", required=True)
    c.add_argument("--detector", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--split", choices=["val", "test"], default="val")
    c.add_argument("--device", default="cpu")
    c.add_argument("--size", type=int, default=640)
    c.add_argument("--batch", type=int, default=1)

    # predict
    c = sub.add_parser("predict", help="Run single-image enhancement and detection")
    for key in ["image", "enhancer", "detector", "out"]:
        c.add_argument("--" + key, required=True)
    c.add_argument("--device", default="cpu")
    c.add_argument("--size", type=int, default=640)
    c.add_argument("--conf", type=float, default=0.25)
    c.add_argument("--hq", action="store_true", help="Use high-quality post-processing pipeline")
    c.add_argument("--scale-factor", type=int, default=None)

    # benchmark
    c = sub.add_parser("benchmark", help="Benchmark warmed enhancer, detector, and end-to-end latency")
    c.add_argument("--detector", required=True)
    c.add_argument("--enhancer", default=None)
    c.add_argument("--out", default=None, help="Optional JSON path to save benchmark report")
    c.add_argument("--device", default="cpu")
    c.add_argument("--size", type=int, default=640)
    c.add_argument("--runs", type=int, default=20)
    c.add_argument("--warmup", type=int, default=5)
    c.add_argument("--scale-factor", type=int, default=None)

    return p


def main() -> None:
    p = build_parser()
    a = p.parse_args()

    if hasattr(a, "size") and (a.size < 32 or a.size % 32):
        p.error("--size must be a positive multiple of 32 >=32")
    for key in ["epochs", "batch", "val_every", "grad_accum", "runs", "width", "iterations"]:
        if hasattr(a, key) and getattr(a, key) is not None and getattr(a, key) <= 0:
            p.error(f"--{key} must be positive")
    if getattr(a, "scale_factor", None) is not None and a.scale_factor <= 0:
        p.error("--scale-factor must be positive")

    if a.command == "prepare":
        from .data import prepare

        prepare(a.images, a.annotations, a.metadata, a.out, a.origin)
    else:
        from . import engine

        if a.command == "baseline":
            engine.baseline(a)
        elif a.command in ["pretrain", "guided", "joint"]:
            engine.train_enhancer(a)
        elif a.command == "export":
            if a.mode in ("learned", "hq") and not a.enhancer:
                p.error(f"--enhancer required for {a.mode} export")
            if Path(a.out).exists():
                p.error("Use a new export output folder")
            engine.export_images(
                a.data,
                a.out,
                a.mode,
                a.enhancer,
                a.device,
                tuple(a.splits),
                scale_factor=a.scale_factor,
            )
        elif a.command == "evaluate":
            engine.evaluate(
                a.detector, a.data, a.out, a.split, a.device, a.size, batch=a.batch
            )
        elif a.command == "predict":
            engine.predict(a)
        elif a.command == "benchmark":
            engine.benchmark(a)


if __name__ == "__main__":
    main()
