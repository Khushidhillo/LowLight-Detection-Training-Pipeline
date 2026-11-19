"""Core training, export, evaluation, benchmarking, and inference engine."""

import copy
import csv
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .data import DetectionDataset, collate, image_tensor, list_images, write_yaml
from .enhancer import CurveEnhancer, enhancement_loss
from .hq_enhance import enhance_high_quality


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def load_enhancer(
    checkpoint: Optional[str],
    device,
    width: int = 32,
    iterations: int = 8,
    scale_factor: Optional[int] = None,
) -> CurveEnhancer:
    """Instantiate a CurveEnhancer and optionally load weights from a checkpoint.

    Automatically reads ``arch`` metadata from newer checkpoints while remaining
    backward-compatible with older checkpoints trained with default settings.
    """
    ck = None
    if checkpoint:
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        arch = ck.get("arch", {}) if isinstance(ck, dict) else {}
        width = int(arch.get("width", width))
        iterations = int(arch.get("iterations", iterations))
        ckpt_sf = int(arch.get("scale_factor", 1))
        sf = int(scale_factor) if scale_factor is not None else ckpt_sf
    else:
        sf = int(scale_factor) if scale_factor is not None else 1

    model = CurveEnhancer(width=width, iterations=iterations, scale_factor=sf).to(device)
    if ck is not None:
        model.load_state_dict(ck["enhancer"])
    return model


def _set_bn_eval(module: torch.nn.Module) -> None:
    """Freeze BatchNorm running statistics while leaving Conv weights trainable."""
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def _save_detector_snapshot(detector: torch.nn.Module, path: Path) -> None:
    """Save an eval-mode deepcopy of the detector without mutating live training state."""
    snap = copy.deepcopy(detector).eval()
    for p in snap.parameters():
        p.requires_grad_(False)
    torch.save({"model": snap}, path)


def _plot_training_curves(history_csv: Path, out_png: Path) -> None:
    """Generate a clean multi-panel training curve figure from ``history.csv``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = list(csv.DictReader(history_csv.open("r", encoding="utf-8")))
        if not rows:
            return

        epochs = [int(r["epoch"]) for r in rows]
        total_loss = [float(r["loss"]) for r in rows]
        enh_loss = [float(r["enhancement"]) for r in rows]
        det_loss = [float(r["detection"]) for r in rows]
        val_points = [
            (int(r["epoch"]), float(r["val_map"]))
            for r in rows
            if r.get("val_map") not in (None, "", "None")
        ]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)

        axes[0].plot(epochs, total_loss, label="Total Loss", linewidth=2.0)
        axes[0].plot(epochs, enh_loss, label="Enhancement Loss", linestyle="--")
        if any(v > 0 for v in det_loss):
            axes[0].plot(epochs, det_loss, label="Detection Loss", linestyle=":")
        axes[0].set_title("Training Losses")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        if val_points:
            ve, vm = zip(*val_points)
            axes[1].plot(ve, vm, marker="o", linewidth=2.0, color="#2563eb")
            best_idx = int(np.argmax(vm))
            axes[1].scatter([ve[best_idx]], [vm[best_idx]], color="#dc2626", zorder=5, label=f"Best: {vm[best_idx]:.4f}")
            axes[1].legend()
        axes[1].set_title("Validation mAP50-95")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("mAP50-95")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
    except Exception:
        # Plotting is best-effort and must never fail a training run.
        pass


def export_images(
    root,
    output,
    mode: str,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    splits: Tuple[str, ...] = ("train", "val", "test"),
    model: Optional[CurveEnhancer] = None,
    scale_factor: Optional[int] = None,
) -> None:
    """Export an enhanced copy of a YOLO dataset using learned, HQ, gamma, or CLAHE mode."""
    import cv2

    root, output = Path(root), Path(output)
    if "test" in splits and "val" not in splits:
        # Ultralytics checks the val directory even during test split evaluation.
        splits = (*splits, "val")
    if root.resolve() == output.resolve():
        raise ValueError("Never overwrite source dataset")

    output.mkdir(parents=True, exist_ok=True)
    if mode in ("learned", "hq"):
        model = model or load_enhancer(checkpoint, device, scale_factor=scale_factor)
        model.eval()

    for split in splits:
        target = output / "images" / split
        target.mkdir(parents=True, exist_ok=True)
        labels = output / "labels" / split
        src_labels = root / "labels" / split
        if src_labels.exists():
            shutil.copytree(src_labels, labels, dirs_exist_ok=True)

        for p in list_images(root / "images" / split):
            out_name = p.with_suffix(".png").name
            if mode == "learned":
                with torch.no_grad():
                    x = image_tensor(p).unsqueeze(0).to(device)
                    out_arr = model(x)[0][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                img_u8 = np.round(out_arr * 255.0).astype(np.uint8)
                Image.fromarray(img_u8).save(target / out_name)
            elif mode == "hq":
                with Image.open(p) as im:
                    enhanced = enhance_high_quality(model, im.convert("RGB"), device=str(device))
                enhanced.save(target / out_name)
            else:
                with Image.open(p) as im:
                    arr = np.array(im.convert("RGB"))
                if mode == "gamma":
                    arr = np.round(255.0 * (arr / 255.0) ** 0.5).astype(np.uint8)
                elif mode == "clahe":
                    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
                    lab[:, :, 0] = cv2.createCLAHE(
                        clipLimit=2.0, tileGridSize=(8, 8)
                    ).apply(lab[:, :, 0])
                    arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                else:
                    raise ValueError(f"Unsupported export mode: {mode}")
                Image.fromarray(arr).save(target / out_name)

    write_yaml(output)
    (output / "export.json").write_text(
        json.dumps(
            {"mode": mode, "checkpoint": str(checkpoint), "splits": list(splits)},
            indent=2,
        ),
        encoding="utf-8",
    )


def evaluate(
    weights,
    root,
    output,
    split: str = "val",
    device: str = "cpu",
    size: int = 640,
    batch: int = 1,
) -> float:
    """Evaluate a YOLO checkpoint on a dataset split and persist metrics + plots."""
    from ultralytics import YOLO

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    m = YOLO(str(weights)).val(
        data=str(Path(root) / "data.yaml"),
        split=split,
        imgsz=size,
        batch=batch,
        device=device,
        workers=0,
        plots=True,
        project=str(out),
        name="plots",
        exist_ok=True,
    )
    result = {k: float(v) for k, v in m.results_dict.items()}
    present = {int(i) for i in m.box.ap_class_index}
    result["per_class_mAP50_95"] = {
        str(m.names[i]): float(v) if i in present else None
        for i, v in enumerate(m.box.maps)
    }
    result["detector_speed_ms"] = m.speed
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return float(m.box.map)


def baseline(args) -> None:
    """Train a raw-image YOLOv8 baseline detector on the prepared dataset."""
    from ultralytics import YOLO

    if Path(args.out).exists():
        raise FileExistsError("Use a new baseline output directory")
    model = YOLO(args.weights)
    model.train(
        data=str(Path(args.data) / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.size,
        batch=args.batch,
        device=args.device,
        workers=getattr(args, "workers", 0),
        seed=args.seed,
        deterministic=True,
        project=str(Path(args.out).parent),
        name=Path(args.out).name,
        exist_ok=False,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        amp=bool(getattr(args, "amp", False)),
    )


def _build_scheduler(optimizer, scheduler_name: str, epochs: int, warmup_epochs: int):
    """Create a cosine-with-optional-warmup LR scheduler or None."""
    if scheduler_name == "none" or epochs <= 1:
        return None

    warmup = max(0, min(int(warmup_epochs), epochs - 1))

    def lr_lambda(epoch_idx: int) -> float:
        if warmup > 0 and epoch_idx < warmup:
            return float(epoch_idx + 1) / float(warmup)
        progress = float(epoch_idx - warmup) / float(max(1, epochs - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_enhancer(args) -> None:
    """Train the curve enhancer in pretrain, detection-guided, or joint JE-YOLO mode."""
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8DetectionLoss

    seed_all(args.seed)
    dev = torch.device(args.device)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Use a new run folder; previous results are protected")
    out.mkdir(parents=True)

    # Read optional advanced hyperparameters with safe backward-compatible defaults
    width = int(getattr(args, "width", 32))
    iterations = int(getattr(args, "iterations", 8))
    scale_factor = getattr(args, "scale_factor", None)
    exposure = float(getattr(args, "exposure", 0.6))
    w_spatial = float(getattr(args, "w_spatial", 1.0))
    w_exp = float(getattr(args, "w_exp", 10.0))
    w_color = float(getattr(args, "w_color", 5.0))
    w_tv = float(getattr(args, "w_tv", 200.0))
    grad_accum = max(1, int(getattr(args, "grad_accum", 1)))
    scheduler_name = getattr(args, "scheduler", "cosine")
    warmup_epochs = int(getattr(args, "warmup_epochs", 0))
    augment = bool(getattr(args, "augment", False))
    num_workers = int(getattr(args, "workers", 0))
    use_amp = bool(getattr(args, "amp", False)) and dev.type == "cuda"
    val_batch = int(getattr(args, "val_batch", None) or (4 if dev.type == "cuda" else 1))

    guided = args.command == "guided"
    joint = args.command == "joint"
    guided_or_joint = guided or joint

    freeze_bn = bool(getattr(args, "freeze_bn", True)) if joint else True
    weight_decay = getattr(args, "weight_decay", None)
    if weight_decay is None:
        weight_decay = 1e-4 if joint else 0.0

    if guided_or_joint and not args.init and not args.resume:
        raise ValueError(f"{args.command.capitalize()} training requires --init from pretraining")
    if not args.detector:
        raise ValueError("--detector is required for validation mAP selection")

    model = load_enhancer(
        args.init,
        dev,
        width=width,
        iterations=iterations,
        scale_factor=scale_factor,
    )

    config_snapshot = {**vars(args), "arch": model.get_config(), "amp_active": use_amp, "freeze_bn": freeze_bn}
    (out / "config.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    detector_wrapper = YOLO(args.detector)
    detector = detector_wrapper.model.to(dev).float()
    if detector.model[-1].nc != 12:
        raise ValueError("Use the 12-class ExDark baseline checkpoint")
    detector.args = get_cfg(overrides={"box": 7.5, "cls": 0.5, "dfl": 1.5})

    if joint:
        detector.train()
        for p in detector.parameters():
            p.requires_grad_(True)
        if freeze_bn:
            _set_bn_eval(detector)
    else:
        detector.eval()
        for p in detector.parameters():
            p.requires_grad_(False)

    criterion = v8DetectionLoss(detector)
    dataset = DetectionDataset(args.data, split="train", size=args.size, augment=augment)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(dev.type == "cuda"),
    )

    det_lr = getattr(args, "det_lr", None)
    if det_lr is None:
        det_lr = args.lr if joint else 0.0

    if joint:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(), "lr": args.lr},
                {"params": detector.parameters(), "lr": det_lr},
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=weight_decay
        )

    scheduler = _build_scheduler(optimizer, scheduler_name, args.epochs, warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if dev.type == "cuda" else None

    start = 0
    best = -1.0
    prior_history_rows: List[Dict[str, str]] = []

    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        stage_match = ck.get("command", "guided" if ck.get("guided") else "pretrain")
        if stage_match != args.command:
            raise ValueError("Resume stage mismatch")
        model.load_state_dict(ck["enhancer"])
        if joint and ck.get("detector") is not None:
            detector.load_state_dict(ck["detector"])
        optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None and ck.get("scheduler") is not None:
            scheduler.load_state_dict(ck["scheduler"])
        if scaler is not None and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])

        start = int(ck["epoch"]) + 1
        best = float(ck["best_map"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng"):
            torch.cuda.set_rng_state_all(ck["cuda_rng"])

        resume_dir = Path(args.resume).parent
        previous_best = resume_dir / "best.pt"
        if previous_best.exists():
            shutil.copy2(previous_best, out / "best.pt")
            if joint and (resume_dir / "detector_best.pt").exists():
                shutil.copy2(resume_dir / "detector_best.pt", out / "detector_best.pt")
        else:
            best = -1.0  # No prior best checkpoint available.

        prev_hist = resume_dir / "history.csv"
        if prev_hist.exists():
            with prev_hist.open("r", encoding="utf-8") as hf:
                for row in csv.DictReader(hf):
                    if int(row["epoch"]) <= start:
                        prior_history_rows.append(row)

    fieldnames = ["epoch", "loss", "enhancement", "detection", "lr", "val_map", "seconds"]
    history_path = out / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for prev_row in prior_history_rows:
            writer.writerow({k: prev_row.get(k, "") for k in fieldnames})
        f.flush()

        for epoch in range(start, args.epochs):
            model.train()
            if joint:
                detector.train()
                if freeze_bn:
                    _set_bn_eval(detector)

            totals = np.zeros(3, dtype=np.float64)
            n = 0
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            num_batches = len(loader)
            for step_idx, batch in enumerate(loader):
                batch = {k: v.to(dev, non_blocking=(dev.type == "cuda")) for k, v in batch.items()}
                x = batch["img"]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    enhanced, curve = model(x)
                    enh, _ = enhancement_loss(
                        x,
                        enhanced,
                        curve,
                        exposure=exposure,
                        w_spatial=w_spatial,
                        w_exp=w_exp,
                        w_color=w_color,
                        w_tv=w_tv,
                    )
                    det = torch.zeros((), device=dev)
                    if guided_or_joint:
                        preds = detector(enhanced)
                        det, _ = criterion(preds, {**batch, "img": enhanced})
                        det = (det.sum() if getattr(det, "ndim", 0) > 0 else det) / x.shape[0]
                    loss = enh + args.det_weight * det

                if not torch.isfinite(loss).all():
                    raise FloatingPointError("Nonfinite training loss")

                # Determine effective accumulation window (handles final partial window)
                window = grad_accum if (step_idx // grad_accum + 1) * grad_accum <= num_batches else (num_batches - (step_idx // grad_accum) * grad_accum)
                scaled_loss = loss / float(max(1, window))

                if scaler is not None and use_amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                is_step = ((step_idx + 1) % grad_accum == 0) or ((step_idx + 1) == num_batches)
                if is_step:
                    if scaler is not None and use_amp:
                        scaler.unscale_(optimizer)

                    if guided_or_joint and not any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.parameters()
                    ):
                        raise RuntimeError("Enhancer gradient is missing")
                    if joint and not any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in detector.parameters()
                    ):
                        raise RuntimeError("Detector gradient is missing")

                    params_to_clip = (
                        list(model.parameters()) + list(detector.parameters())
                        if joint
                        else model.parameters()
                    )
                    torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)

                    if scaler is not None and use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                totals += np.array([loss.item(), enh.item(), det.item()], dtype=np.float64) * len(x)
                n += len(x)

            current_lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                scheduler.step()

            score = None
            if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
                cache = out / "validation_images"
                export_images(args.data, cache, "learned", device=dev, splits=("val",), model=model)
                if joint:
                    val_det_path = out / "detector_last.pt"
                    _save_detector_snapshot(detector, val_det_path)
                    eval_detector = str(val_det_path)
                else:
                    eval_detector = args.detector
                score = evaluate(
                    eval_detector,
                    cache,
                    out / "validation" / f"epoch_{epoch + 1}",
                    device=args.device,
                    size=args.size,
                    batch=val_batch,
                )

            improved = score is not None and score > best
            if improved:
                best = score

            ck = {
                "enhancer": model.state_dict(),
                "arch": model.get_config(),
                "detector": detector.state_dict() if joint else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if (scaler is not None and use_amp) else None,
                "epoch": epoch,
                "best_map": best,
                "guided": guided,
                "command": args.command,
                "config": config_snapshot,
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }
            torch.save(ck, out / "last.pt")
            if joint:
                _save_detector_snapshot(detector, out / "detector_last.pt")
            if improved:
                torch.save(ck, out / "best.pt")
                if joint:
                    _save_detector_snapshot(detector, out / "detector_best.pt")

            row = dict(zip(["loss", "enhancement", "detection"], totals / max(1, n)))
            row.update(
                epoch=epoch + 1,
                lr=current_lr,
                val_map=score,
                seconds=time.perf_counter() - began,
            )
            writer.writerow(row)
            f.flush()
            print(row, flush=True)

    _plot_training_curves(history_path, out / "training_curves.png")


def _make_comparison_grid(
    orig: Image.Image,
    enhanced: Image.Image,
    orig_det: Image.Image,
    enh_det: Image.Image,
) -> Image.Image:
    """Assemble a clean 2x2 titled visual comparison image."""
    w, h = orig.size
    header = 32
    canvas = Image.new("RGB", (w * 2, (h + header) * 2), color=(20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    panels = [
        (orig.resize((w, h)), "Original", 0, 0),
        (enhanced.resize((w, h)), "Enhanced", w, 0),
        (orig_det.resize((w, h)), "Detections (Original)", 0, h + header),
        (enh_det.resize((w, h)), "Detections (Enhanced)", w, h + header),
    ]
    for img, title, ox, oy in panels:
        draw.text((ox + 10, oy + 8), title, fill=(240, 240, 245))
        canvas.paste(img, (ox, oy + header))
    return canvas


def predict(args) -> None:
    """Run single-image enhancement + YOLO detection and save visual comparisons."""
    from ultralytics import YOLO

    dev = torch.device(args.device)
    scale_factor = getattr(args, "scale_factor", None)
    use_hq = bool(getattr(args, "hq", False))
    conf = float(getattr(args, "conf", 0.25))

    model = load_enhancer(args.enhancer, dev, scale_factor=scale_factor).eval()
    detector = YOLO(args.detector)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    source = Image.open(args.image).convert("RGB")
    x = image_tensor(args.image).unsqueeze(0).to(dev)

    # Warmup enhancer
    for _ in range(2):
        with torch.no_grad():
            model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    if use_hq:
        pil, hq_meta = enhance_high_quality(model, source, device=str(dev), return_meta=True)
    else:
        with torch.no_grad():
            enhanced = model(x)[0][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        pil = Image.fromarray(np.round(enhanced * 255.0).astype(np.uint8))
        hq_meta = {"method": "learned_curve"}
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    res_enh = detector.predict(pil, device=args.device, imgsz=args.size, conf=conf, verbose=False)[0]
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    res_orig = detector.predict(source, device=args.device, imgsz=args.size, conf=conf, verbose=False)[0]

    enhancer_ms = (t1 - t0) * 1000.0
    detector_ms = (t2 - t1) * 1000.0
    total_ms = (t2 - t0) * 1000.0

    pil.save(out / "enhanced.png")
    res_enh.save(filename=str(out / "enhanced_detection.jpg"))
    res_orig.save(filename=str(out / "original_detection.jpg"))

    orig_det_pil = Image.fromarray(res_orig.plot()[..., ::-1])
    enh_det_pil = Image.fromarray(res_enh.plot()[..., ::-1])
    grid = _make_comparison_grid(source, pil, orig_det_pil, enh_det_pil)
    grid.save(out / "comparison.jpg", quality=92)

    orig_lum = float(np.array(source.convert("L")).mean())
    enh_lum = float(np.array(pil.convert("L")).mean())
    summary = {
        "enhancer_ms": round(enhancer_ms, 2),
        "detector_ms": round(detector_ms, 2),
        "single_image_pipeline_ms": round(total_ms, 2),
        "mode": hq_meta.get("method", "learned_curve"),
        "original_brightness": round(orig_lum, 2),
        "enhanced_brightness": round(enh_lum, 2),
        "detections_original": int(len(res_orig.boxes)),
        "detections_enhanced": int(len(res_enh.boxes)),
        "note": "Illustrative single-image run; use `python -m lowlight benchmark` for warmed repeated trials.",
    }
    (out / "timing.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def benchmark(args) -> Dict[str, object]:
    """Benchmark warmed enhancer, detector, and end-to-end pipeline latency & model size."""
    from ultralytics import YOLO

    dev = torch.device(args.device)
    size = int(args.size)
    runs = int(getattr(args, "runs", 20))
    warmup = int(getattr(args, "warmup", 5))
    scale_factor = getattr(args, "scale_factor", None)

    enhancer = load_enhancer(args.enhancer, dev, scale_factor=scale_factor).eval()
    detector_wrapper = YOLO(args.detector)
    detector = detector_wrapper.model.to(dev).eval().float()

    enh_params = enhancer.param_count()
    det_params = sum(p.numel() for p in detector.parameters())
    total_params = enh_params + det_params

    enh_mb = round(Path(args.enhancer).stat().st_size / (1024 * 1024), 3) if args.enhancer and Path(args.enhancer).exists() else None
    det_mb = round(Path(args.detector).stat().st_size / (1024 * 1024), 3) if Path(args.detector).exists() else None

    dummy = torch.rand(1, 3, size, size, device=dev) * 0.2

    def _sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            y, _ = enhancer(dummy)
            _ = detector(y)
    _sync()

    enh_times: List[float] = []
    det_times: List[float] = []
    e2e_times: List[float] = []

    with torch.no_grad():
        for _ in range(runs):
            _sync()
            t0 = time.perf_counter()
            y, _ = enhancer(dummy)
            _sync()
            t1 = time.perf_counter()
            _ = detector(y)
            _sync()
            t2 = time.perf_counter()

            enh_times.append((t1 - t0) * 1000.0)
            det_times.append((t2 - t1) * 1000.0)
            e2e_times.append((t2 - t0) * 1000.0)

    def _stats(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals, dtype=np.float64)
        mean_ms = float(arr.mean())
        return {
            "mean_ms": round(mean_ms, 3),
            "std_ms": round(float(arr.std()), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "fps": round(1000.0 / max(mean_ms, 1e-6), 2),
        }

    report: Dict[str, object] = {
        "device": str(dev),
        "input_size": [size, size],
        "enhancer_arch": enhancer.get_config(),
        "runs": runs,
        "warmup": warmup,
        "parameters": {
            "enhancer": enh_params,
            "detector": det_params,
            "combined": total_params,
        },
        "checkpoint_mb": {
            "enhancer": enh_mb,
            "detector": det_mb,
        },
        "latency": {
            "enhancer": _stats(enh_times),
            "detector_forward": _stats(det_times),
            "end_to_end_forward": _stats(e2e_times),
        },
    }

    if getattr(args, "out", None):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return report
