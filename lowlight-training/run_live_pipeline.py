"""Run full training pipeline on data/exdark with real-time status output."""
import os, sys, time, json
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)
from pathlib import Path
from types import SimpleNamespace
from lowlight.engine import baseline, train_enhancer, export_images, evaluate, predict, benchmark
from compare_results import compare

STATUS_FILE = Path("live_status.json")

def update_status(stage, progress, status, details=None):
    data = {
        "stage": stage,
        "progress": progress,
        "status": status,
        "timestamp": time.time(),
        "details": details or {}
    }
    STATUS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] [{progress:3d}%] {stage}: {status}", flush=True)

start_time = time.perf_counter()
data_dir = "data/exdark"
runs_dir = Path("runs_live")
runs_dir.mkdir(parents=True, exist_ok=True)

# Stage 1: Baseline Detector
update_status("Baseline", 10, "Training YOLOv8n detector on raw low-light data (3 epochs)...")
t0 = time.perf_counter()
baseline(SimpleNamespace(
    weights="yolov8n.pt",
    data=data_dir,
    out="runs_live/baseline",
    epochs=3,
    batch=2,
    size=320,
    device="cpu",
    seed=42,
    amp=False
))
update_status("Baseline", 30, "Baseline training complete", {"time_s": round(time.perf_counter() - t0, 1)})

# Stage 2: Pretrain Curve Enhancer
update_status("Pretrain", 35, "Pretraining Zero-DCE++ curve enhancer with non-reference loss (3 epochs)...")
t0 = time.perf_counter()
common = dict(
    data=data_dir,
    detector="runs_live/baseline/weights/best.pt",
    epochs=3,
    batch=2,
    size=320,
    device="cpu",
    seed=42,
    lr=1e-4,
    det_weight=0.1,
    val_every=1,
    resume=None,
    width=32,
    iterations=8,
    scale_factor=1,
    grad_accum=1,
    scheduler="cosine",
    warmup_epochs=0,
    augment=True,
    workers=0,
    amp=False
)
train_enhancer(SimpleNamespace(**common, command="pretrain", init=None, out="runs_live/enhancer"))
update_status("Pretrain", 50, "Enhancer pretraining complete", {"time_s": round(time.perf_counter() - t0, 1)})

# Stage 3: Detection-Guided Enhancement
update_status("Guided", 55, "Guided training: updating enhancer through frozen detector (3 epochs)...")
t0 = time.perf_counter()
train_enhancer(SimpleNamespace(**common, command="guided", init="runs_live/enhancer/best.pt", out="runs_live/guided"))
update_status("Guided", 70, "Guided enhancement complete", {"time_s": round(time.perf_counter() - t0, 1)})

# Stage 4: Joint JE-YOLO Fine-Tuning
update_status("Joint", 75, "Joint training: simultaneously updating enhancer and detector (3 epochs)...")
t0 = time.perf_counter()
train_enhancer(SimpleNamespace(
    **common,
    command="joint",
    init="runs_live/enhancer/best.pt",
    out="runs_live/joint",
    det_lr=1e-5,
    freeze_bn=True
))
update_status("Joint", 85, "Joint fine-tuning complete", {"time_s": round(time.perf_counter() - t0, 1)})

# Stage 5: Export Test Sets
update_status("Export", 88, "Exporting enhanced test splits for guided and joint checkpoints...")
export_images(data_dir, "runs_live/enhanced_guided", "learned", "runs_live/guided/best.pt", splits=("test",), scale_factor=1)
export_images(data_dir, "runs_live/enhanced_joint", "learned", "runs_live/joint/best.pt", splits=("test",), scale_factor=1)

# Stage 6: Evaluate Test Splits
update_status("Evaluation", 92, "Evaluating baseline detector on raw vs guided vs joint...")
evaluate("runs_live/baseline/weights/best.pt", data_dir, "runs_live/results_raw", split="test", size=320)
evaluate("runs_live/baseline/weights/best.pt", "runs_live/enhanced_guided", "runs_live/results_guided", split="test", size=320)
evaluate("runs_live/joint/detector_best.pt", "runs_live/enhanced_joint", "runs_live/results_joint", split="test", size=320)

compare(
    ["runs_live/results_raw/metrics.json", "runs_live/results_guided/metrics.json", "runs_live/results_joint/metrics.json"],
    "runs_live/comparison.csv",
    "runs_live/per_class.csv"
)

# Stage 7: Single-image prediction & benchmark
update_status("Inference & Benchmark", 97, "Running predict demo and latency benchmark...")
predict(SimpleNamespace(
    image="data/exdark/images/test/sample_test_0.png",
    enhancer="runs_live/guided/best.pt",
    detector="runs_live/baseline/weights/best.pt",
    out="runs_live/predict_demo",
    device="cpu",
    size=320,
    hq=True,
    scale_factor=1,
    conf=0.25
))

benchmark(SimpleNamespace(
    enhancer="runs_live/joint/best.pt",
    detector="runs_live/joint/detector_best.pt",
    out="runs_live/benchmark.json",
    device="cpu",
    size=320,
    runs=3,
    warmup=1,
    scale_factor=1
))

total_time = round(time.perf_counter() - start_time, 1)
update_status("Complete", 100, f"Full pipeline finished successfully in {total_time}s", {"total_time_s": total_time})
