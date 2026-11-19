# Detection-guided low-light object detection

Research training pipeline for ExDark, a Zero-DCE++-inspired curve enhancer,
and YOLOv8n. Includes original-image baseline, independent enhancement,
detection-guided enhancement with a frozen detector, JE-YOLO inspired end-to-end joint fine-tuning,
conventional baselines, validation/test evaluation, exports, and image inference.

**No dataset or trained weights are bundled. No accuracy improvement is claimed.**
The enhancement module is an independent implementation inspired by Zero-DCE++;
it is not an exact reproduction or compatible with official checkpoints.
The guided stage freezes the detector and preserves gradients to the enhancer.
The joint stage enables JE-YOLO inspired end-to-end joint fine-tuning where both enhancer and detector are updated simultaneously.

## 1. Environment

Use Python 3.10 or 3.11 (also smoke-tested on 3.12). Avoid Python 3.13/3.14
for these pinned dependency versions. From this extracted directory:

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# NVIDIA GPU; install the wheel appropriate for your driver. This tested family uses CUDA 12.1:
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pytest -q
```
CPU alternative (slow for full training):
```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```
Use `--device cpu` for CPU or `--device cuda:0` for GPU. All examples below use
CPU so they run without an NVIDIA GPU; switch for actual training. Use batch 2
or 4 initially; reduce it after a CUDA out-of-memory error. Intel integrated
graphics are not a CUDA device.

## 2. Download and prepare ExDark

Get the images, bounding-box annotations, and `imageclasslist.txt` from:
https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

Keep their extracted class folders. Download manually from the authors' links;
review their dataset terms. Dataset and third-party model licenses remain separate.
The converter searches image and annotation directories recursively.

```bash
python -m lowlight prepare --images raw/ExDark --annotations raw/Groundtruth --metadata raw/imageclasslist.txt --origin 1 --out data/exdark
```
`--origin 1` assumes the MATLAB-style boxes use one-based left/top coordinates.
Verify this against your copy; use `--origin 0` for already converted zero-based
annotations. Invalid boxes fail loudly instead of being silently clipped.
The native annotation header beginning `%` is skipped; rows are `Class l t w h ...`.
The metadata's fifth field determines train=1, val=2, test=3. Official counts:
3000 / 1800 / 2563. Compare printed counts to detect incomplete downloads.

Output: RGB PNG images, normalized YOLO labels, `data.yaml`, and a manifest with
lighting metadata and pixel hashes. Identical pixels across splits are rejected;
near duplicates still need human review. Dataset paths in YAML are absolute;
if you move the dataset, update `path` in `data.yaml`.

Inspect converted boxes BEFORE training:
```bash
python inspect_labels.py --data data/exdark --out inspections --count 24
```
The converter creates a fresh output directory; do not reuse a partial failed
conversion. Remove only that partial output manually, or choose another folder.

## 3. Train the raw-image baseline

```bash
python -m lowlight baseline --data data/exdark --out runs/baseline --epochs 80 --batch 4 --size 640 --device cpu
```
The first run downloads `yolov8n.pt`; for offline training supply a local trusted
checkpoint with `--weights path/to/yolov8n.pt`. The detector is adapted to 12 classes.
Output: `runs/baseline/weights/best.pt`, plots and training metrics.

## 4. Independently train enhancement

```bash
python -m lowlight pretrain --data data/exdark --detector runs/baseline/weights/best.pt --out runs/enhancer --epochs 50 --batch 4 --size 640 --val-every 5 --device cpu
```
Only training images affect gradients. Loss combines spatial consistency,
exposure control, color constancy and curve smoothness. Validation enhancement
is evaluated by a fixed detector every five epochs; `best.pt` is selected by
validation mAP50-95. This is independent enhancement **training**, with
validation detection performance used for checkpoint selection in both stages.
Every validation pass writes native-resolution enhanced PNGs and detector plots;
expect time/disk overhead. The validation cache is reused within a run.

## 5. Detection-guided enhancement

```bash
python -m lowlight guided --data data/exdark --detector runs/baseline/weights/best.pt --init runs/enhancer/best.pt --out runs/guided --epochs 30 --batch 4 --size 640 --lr 0.00005 --det-weight 0.1 --val-every 5 --device cpu
```
Total loss = non-reference enhancement loss + det_weight * detection loss.
The YOLO training loss receives raw predictions; NMS, JPEG conversion and
inference-mode wrappers never sit on the gradient path. Detector parameters and
BatchNorm running statistics are frozen. No bright reference targets are used.

Suggested validation-only sweep: det_weight = 0, 0.01, 0.1, 1.0. Keep the best
validation setting; do not tune against test images. The zero-weight continuation
is a useful control for the extra training budget.

## 5.1 End-to-End Joint Fine-tuning (JE-YOLO)

```bash
python -m lowlight joint --data data/exdark --detector runs/baseline/weights/best.pt --init runs/enhancer/best.pt --out runs/joint --epochs 30 --batch 4 --size 640 --lr 0.00005 --det-lr 0.00001 --det-weight 0.1 --val-every 5 --augment --device cpu
```
In joint mode, both the curve enhancer and the YOLO object detector are updated simultaneously (end-to-end joint fine-tuning).
By default, detector BatchNorm running statistics stay frozen (`--no-freeze-bn` to unlock) to prevent small-batch stat drift, while all convolutional and task-head weights adapt alongside the enhancer using AdamW + cosine LR decay.
Output: `runs/joint/best.pt` (enhancer checkpoint) and `runs/joint/detector_best.pt` (eval-mode adapted detector checkpoint).

All enhancer stages (`pretrain`, `guided`, `joint`) support:
- `--scale-factor N`: Zero-DCE++ multi-scale curve estimation (downsamples input by `N` to estimate the curve map cheaply and upsamples the curve map back to native resolution).
- `--augment`: label-safe random horizontal flip augmentation.
- `--amp`: CUDA automatic mixed precision (`GradScaler`).
- `--grad-accum K`: gradient accumulation for larger effective batch sizes on low-VRAM GPUs.
- `--scheduler cosine|none` and `--warmup-epochs N`: learning rate scheduling.
- `--exposure`, `--w-spatial`, `--w-exp`, `--w-color`, `--w-tv`: configurable non-reference loss weights.

All enhancer stages produce `config.json`, `history.csv`, `training_curves.png`, `last.pt` and `best.pt`.
Resume interrupted enhancer training into a new output folder using the same
arguments plus `--resume old_run/last.pt`. Previous `history.csv` rows and scheduler/optimizer/RNG states are preserved. Keep `old_run/best.pt` beside it to preserve the previous best checkpoint; if absent, best-checkpoint selection restarts.
`--epochs` is the desired TOTAL epoch count. Use the same data, seed, detector,
batch size and loss settings when resuming. Baseline
resume uses the native Ultralytics interface (see their documentation).
Load only trusted PyTorch checkpoints: this pinned stack uses pickle serialization.

## 6. Export, evaluate, and benchmark

First compare on validation. For final evaluation, explicitly include test (export each stage with its matched enhancer checkpoint):
```bash
python -m lowlight export --data data/exdark --out data/guided --mode learned --enhancer runs/guided/best.pt --splits train val test --device cpu
python -m lowlight export --data data/exdark --out data/joint --mode learned --enhancer runs/joint/best.pt --splits val test --device cpu
python -m lowlight evaluate --data data/exdark --detector runs/baseline/weights/best.pt --out results/raw_test --split test --device cpu
python -m lowlight evaluate --data data/guided --detector runs/baseline/weights/best.pt --out results/guided_test --split test --device cpu
python -m lowlight evaluate --data data/joint --detector runs/joint/detector_best.pt --out results/joint_test --split test --device cpu
```
Repeat export/evaluate with the independent checkpoint. Use `--mode hq` for the high-quality post-processed pipeline, or `--mode gamma` / `--mode clahe` without `--enhancer` for conventional preprocessing. To measure
detector adaptation, train another baseline on the independently enhanced dataset
using the SAME original pretrained initialization and training budget.

Outputs include mAP50, mAP50-95, precision, recall, per-class mAP and plots.
`detector_speed_ms` excludes offline enhancement and MUST NOT be presented as
end-to-end latency. Use `benchmark` to measure combined parameters and true warmed end-to-end latency:
```bash
python -m lowlight benchmark --enhancer runs/joint/best.pt --detector runs/joint/detector_best.pt --size 640 --runs 50 --warmup 10 --device cpu --out results/benchmark.json
```
Export uses lossless PNG so compression changes do not confound
comparisons. Test export performs inference only, never training or selection. Test-only export also writes validation images because Ultralytics requires a validation directory even when evaluating the test split.

```bash
python compare_results.py --results results/raw_test/metrics.json results/guided_test/metrics.json results/joint_test/metrics.json --out results/comparison.csv --per-class-out results/per_class.csv
```
For a report, run seeds 42/43/44 and show mean/std. Include false positives and
missed objects, darkest scenes, and small objects. Do not claim PSNR/SSIM on ExDark;
there are no paired bright targets. Synthetic smoke data are for code tests only.

## 7. Single-image demonstration & interactive web demo

```bash
python -m lowlight predict --image example.jpg --enhancer runs/guided/best.pt --detector runs/baseline/weights/best.pt --out demo --hq --device cpu
```
Writes `enhanced.png`, `original_detection.jpg`, `enhanced_detection.jpg`, a 2×2 visual grid `comparison.jpg`, and `timing.json`.
Use `--scale-factor 4` for fast low-memory inference on very high-resolution images.

For an interactive browser UI (requires `gradio`):
```bash
python demo_app.py --port 7860
```

## Implementation and scientific limitations

- Enhancer training uses square resizing; normalized boxes remain geometrically
  correct. YOLO baseline/evaluation use native letterboxing. Enhancement export is
  native resolution. This is a documented train/inference scale difference; study
  it before deployment and keep it fixed across enhancement ablations.
- Exposure target is 0.6. Loss weights are starting values, not optimized results.
- Enhancer alone is small; report combined model parameters and measured latency.
- No guarantee enhancement helps; strong raw-image training can outperform it.
- No denoising or temporal stabilization; video flicker has not been evaluated.
- Frozen-detector guidance can overfit that detector; a second detector is useful
  for testing transfer after the primary project works.
- This is a research pipeline, not a validated production model.

## Sources

- ExDark native format and splits: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Groundtruth/README.md
- Zero-DCE: https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html
- Zero-DCE++: https://li-chongyi.github.io/Proj_Zero-DCE++.html
- YOLOv8: https://docs.ultralytics.com/models/yolov8/
- Ultralytics licensing: https://github.com/ultralytics/ultralytics/blob/main/LICENSE

See `VALIDATION.md` for the actual checks performed on this package.
