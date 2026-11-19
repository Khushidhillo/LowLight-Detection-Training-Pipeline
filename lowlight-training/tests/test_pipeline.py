import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from lowlight.data import DetectionDataset, convert_box, prepare, write_yaml


def test_coordinates():
    assert convert_box(1, 1, 20, 10, 100, 100, 1) == (0.1, 0.05, 0.2, 0.1)
    with pytest.raises(ValueError):
        convert_box(-1, 0, 20, 10, 100, 100, 0)


def test_conversion_and_leakage(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    anns = tmp_path / "annotations"
    anns.mkdir()
    rows = []
    for i in range(1, 4):
        Image.new("RGB", (40, 30), (i * 20, 0, 0)).save(images / f"{i}.png")
        (anns / f"{i}.png.txt").write_text("% bbGt version=3\nCar 1 1 20 10 0 0 0 0 0 0 0\n")
        rows.append(f"{i}.png 5 1 1 {i}")
    meta = tmp_path / "meta.txt"
    meta.write_text("\n".join(rows))

    prepare(images, anns, meta, tmp_path / "converted", 1)
    manifest = json.loads((tmp_path / "converted" / "manifest.json").read_text())
    assert len(manifest) == 3
    assert manifest[0]["lighting_name"] == "Low"
    assert (tmp_path / "converted" / "labels" / "train" / "1.txt").read_text().startswith("4 ")

    Image.open(images / "1.png").save(images / "2.png")
    with pytest.raises(ValueError, match="crosses splits"):
        prepare(images, anns, meta, tmp_path / "bad", 1)


def test_enhancer_and_detection_gradient():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss
    from lowlight.enhancer import CurveEnhancer, enhancement_loss

    torch.set_num_threads(1)
    enhancer = CurveEnhancer()
    detector = DetectionModel("yolov8n.yaml", nc=12, verbose=False).eval()
    detector.args = get_cfg(overrides={"box": 7.5, "cls": 0.5, "dfl": 1.5})
    for p in detector.parameters():
        p.requires_grad_(False)

    x = torch.rand(2, 3, 64, 64) * 0.2
    y, curve = enhancer(x)
    assert torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1

    labels = {
        "img": y,
        "cls": torch.tensor([[4.0], [10.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.4, 0.4, 0.3, 0.3]]),
        "batch_idx": torch.tensor([0.0, 1.0]),
    }
    loss, _ = v8DetectionLoss(detector)(detector(y), labels)
    if loss.ndim > 0:
        loss = loss.sum()
    loss.backward()
    assert sum(p.grad.abs().sum().item() for p in enhancer.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in detector.parameters())

    enhancer.zero_grad()
    y, curve = enhancer(x)
    l, parts = enhancement_loss(x, y, curve)
    l.backward()
    assert torch.isfinite(l)
    assert set(parts.keys()) == {"spatial", "exposure", "color", "tv"}


def test_joint_fine_tuning_gradient():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss
    from lowlight.enhancer import CurveEnhancer
    from lowlight.engine import _set_bn_eval

    torch.set_num_threads(1)
    enhancer = CurveEnhancer()
    detector = DetectionModel("yolov8n.yaml", nc=12, verbose=False).train()
    detector.args = get_cfg(overrides={"box": 7.5, "cls": 0.5, "dfl": 1.5})
    for p in detector.parameters():
        p.requires_grad_(True)
    _set_bn_eval(detector)

    # Ensure BatchNorm layers stay in eval mode while conv weights still receive gradients
    bns = [m for m in detector.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert bns and all(not bn.training for bn in bns)

    x = torch.rand(2, 3, 64, 64) * 0.2
    y, curve = enhancer(x)
    labels = {
        "img": y,
        "cls": torch.tensor([[4.0], [10.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.4, 0.4, 0.3, 0.3]]),
        "batch_idx": torch.tensor([0.0, 1.0]),
    }
    loss, _ = v8DetectionLoss(detector)(detector(y), labels)
    if loss.ndim > 0:
        loss = loss.sum()
    loss.backward()
    assert sum(p.grad.abs().sum().item() for p in enhancer.parameters() if p.grad is not None) > 0
    assert sum(p.grad.abs().sum().item() for p in detector.parameters() if p.grad is not None) > 0


def test_enhancer_multiscale_and_shapes():
    torch = pytest.importorskip("torch")
    from lowlight.enhancer import CurveEnhancer, enhancement_loss

    for sf in (1, 4, 8):
        m = CurveEnhancer(width=16, iterations=6, scale_factor=sf)
        assert m.get_config() == {"width": 16, "iterations": 6, "scale_factor": sf}
        assert m.param_count() > 0

        # Odd non-multiple-of-4 spatial shape
        x = torch.rand(1, 3, 53, 77) * 0.25
        y, curve = m(x)
        assert y.shape == x.shape
        assert curve.shape == x.shape
        assert torch.isfinite(y).all() and y.min() >= 0.0 and y.max() <= 1.0

        loss, _ = enhancement_loss(x, y, curve, exposure=0.55)
        assert torch.isfinite(loss)


def test_hq_enhance_preserves_contrast_and_brightens():
    pytest.importorskip("torch")
    from lowlight.enhancer import CurveEnhancer
    from lowlight.hq_enhance import _local_contrast_boost, enhance_high_quality, render_curve_map

    # 1. Regression test for fine-texture contrast preservation
    yy, xx = np.mgrid[0:128, 0:128]
    tex = 0.4 + 0.3 * np.sin(xx * 0.8) * np.sin(yy * 0.8)
    tex_rgb = np.stack([tex] * 3, axis=-1).astype(np.float32)
    boosted = _local_contrast_boost(tex_rgb, strength=0.35)
    # Previously crashed std from 0.149 -> 0.0079 (95% loss). Now contrast must stay >= original.
    assert float(boosted.std()) >= float(tex_rgb.std()) * 0.95

    # 2. Dark image brightening via HQ pipeline
    dark = Image.new("RGB", (65, 51), (18, 22, 28))
    enh_dark, meta = enhance_high_quality(
        CurveEnhancer().eval(), dark, device="cpu", return_meta=True
    )
    assert enh_dark.size == dark.size
    assert meta["enhanced_luminance"] > meta["orig_luminance"] * 1.5
    assert render_curve_map(meta["curve_map"]) is not None

    # 3. Already bright image must NOT be blown out to pure white
    bright = Image.new("RGB", (64, 64), (210, 205, 200))
    enh_bright, meta_b = enhance_high_quality(
        None, bright, device="cpu", return_meta=True
    )
    assert meta_b["gamma_used"] == pytest.approx(1.0)
    assert np.array(enh_bright).mean() < 245.0


def test_dataset_horizontal_flip_augmentation(tmp_path):
    pytest.importorskip("torch")
    img_dir = tmp_path / "images" / "train"
    lab_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)

    # Left half dark, right half bright, saved as JPEG to verify non-PNG support
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, 16:] = 200
    Image.fromarray(arr).save(img_dir / "sample.jpg")
    (lab_dir / "sample.txt").write_text("4 0.25 0.50 0.20 0.40\n")

    # Empty-label image
    Image.new("RGB", (32, 32), (10, 10, 10)).save(img_dir / "empty.png")
    (lab_dir / "empty.txt").write_text("")

    ds_no_flip = DetectionDataset(tmp_path, split="train", size=32, augment=False)
    ds_flip = DetectionDataset(tmp_path, split="train", size=32, augment=True, flip_prob=1.0)
    assert len(ds_no_flip) == 2

    # sample.jpg is sorted after empty.png
    img_orig, lab_orig = ds_no_flip[1]
    img_flipped, lab_flipped = ds_flip[1]
    assert lab_orig[0, 1].item() == pytest.approx(0.25)
    assert lab_flipped[0, 1].item() == pytest.approx(0.75)
    # Bright side moved from right half to left half
    assert img_orig[:, :, :16].mean() < img_orig[:, :, 16:].mean()
    assert img_flipped[:, :, :16].mean() > img_flipped[:, :, 16:].mean()


def test_export_modes_and_test_only_val(tmp_path):
    pytest.importorskip("torch")
    from lowlight.enhancer import CurveEnhancer
    from lowlight.engine import export_images

    data = tmp_path / "data"
    for split in ("val", "test"):
        (data / "images" / split).mkdir(parents=True)
        (data / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (32, 32), (20, 25, 30)).save(data / "images" / split / "0.png")
        (data / "labels" / split / "0.txt").write_text("4 0.5 0.5 0.4 0.4\n")
    write_yaml(data)

    model = CurveEnhancer().eval()
    for mode in ("learned", "hq", "gamma", "clahe"):
        out = tmp_path / f"exp_{mode}"
        export_images(
            data,
            out,
            mode=mode,
            device="cpu",
            splits=("test",),
            model=model if mode in ("learned", "hq") else None,
        )
        assert (out / "images" / "test" / "0.png").exists()
        assert (out / "images" / "val" / "0.png").exists()
        assert (out / "data.yaml").exists()


def test_resume_training_continuity(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel
    from lowlight.engine import load_enhancer, train_enhancer

    torch.set_num_threads(1)
    data = tmp_path / "data"
    for split in ("train", "val", "test"):
        (data / "images" / split).mkdir(parents=True)
        (data / "labels" / split).mkdir(parents=True)
        for i in range(2):
            im = Image.new("RGB", (64, 64), (15 + i, 15, 15))
            ImageDraw.Draw(im).rectangle((16, 16, 48, 48), fill=(80, 60, 40))
            im.save(data / "images" / split / f"{i}.png")
            (data / "labels" / split / f"{i}.txt").write_text("4 0.5 0.5 0.5 0.5\n")
    write_yaml(data)

    det_path = tmp_path / "det.pt"
    det = DetectionModel("yolov8n.yaml", nc=12, verbose=False).eval()
    torch.save({"model": det}, det_path)

    run1 = tmp_path / "run1"
    args1 = SimpleNamespace(
        command="pretrain",
        data=str(data),
        detector=str(det_path),
        init=None,
        resume=None,
        out=str(run1),
        epochs=1,
        batch=2,
        size=64,
        device="cpu",
        seed=42,
        lr=1e-4,
        det_lr=None,
        det_weight=0.1,
        val_every=1,
        width=16,
        iterations=4,
        scale_factor=2,
        grad_accum=2,
        augment=True,
    )
    train_enhancer(args1)
    assert (run1 / "last.pt").exists()
    assert (run1 / "best.pt").exists()
    assert (run1 / "training_curves.png").exists()

    # Verify load_enhancer automatically restores custom arch metadata
    reloaded = load_enhancer(str(run1 / "best.pt"), torch.device("cpu"))
    assert reloaded.get_config() == {"width": 16, "iterations": 4, "scale_factor": 2}

    # Resume into run2 for total epochs=2
    run2 = tmp_path / "run2"
    args2 = SimpleNamespace(**{**vars(args1), "epochs": 2, "out": str(run2), "resume": str(run1 / "last.pt")})
    train_enhancer(args2)
    rows = list(csv.DictReader((run2 / "history.csv").open("r", encoding="utf-8")))
    assert [int(r["epoch"]) for r in rows] == [1, 2]


def test_benchmark_and_compare(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel
    from lowlight.engine import benchmark
    from compare_results import compare

    torch.set_num_threads(1)
    det_path = tmp_path / "det.pt"
    torch.save({"model": DetectionModel("yolov8n.yaml", nc=12, verbose=False).eval()}, det_path)

    bench_json = tmp_path / "bench.json"
    rep = benchmark(
        SimpleNamespace(
            detector=str(det_path),
            enhancer=None,
            out=str(bench_json),
            device="cpu",
            size=64,
            runs=2,
            warmup=1,
            scale_factor=2,
        )
    )
    assert bench_json.exists()
    assert rep["parameters"]["combined"] == rep["parameters"]["enhancer"] + rep["parameters"]["detector"]
    assert rep["latency"]["end_to_end_forward"]["mean_ms"] > 0

    m1 = tmp_path / "raw" / "metrics.json"
    m2 = tmp_path / "enh" / "metrics.json"
    m1.parent.mkdir()
    m2.parent.mkdir()
    m1.write_text(json.dumps({
        "metrics/precision(B)": 0.5,
        "metrics/recall(B)": 0.6,
        "metrics/mAP50(B)": 0.55,
        "metrics/mAP50-95(B)": 0.30,
        "per_class_mAP50_95": {"Car": 0.4, "People": 0.2},
        "detector_speed_ms": {"inference": 12.5},
    }))
    m2.write_text(json.dumps({
        "metrics/precision(B)": 0.55,
        "metrics/recall(B)": 0.65,
        "metrics/mAP50(B)": 0.62,
        "metrics/mAP50-95(B)": 0.35,
        "per_class_mAP50_95": {"Car": 0.45, "People": 0.25},
        "detector_speed_ms": {"inference": 12.1},
    }))
    rows = compare([str(m1), str(m2)], str(tmp_path / "cmp.csv"), str(tmp_path / "pc.csv"))
    assert rows[1]["delta_mAP50"] == pytest.approx(0.07)
    assert rows[1]["delta_mAP50_95"] == pytest.approx(0.05)
    assert (tmp_path / "pc.csv").exists()
