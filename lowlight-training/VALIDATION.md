# Validation performed — 2026-10-07

Environment: Linux, Python 3.12.14, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu,
Ultralytics 8.3.40, NumPy 1.26.4. GPU execution and Windows execution were not run.

Passed:
- Three automated tests: coordinate conversion/bounds, annotation preparation
  and identical-image split leakage rejection, actual YOLO detection-loss gradient
  propagation into the enhancer while detector parameters remain frozen.
- YOLOv8n 12-class baseline: one epoch from random initialization on synthetic
  rectangles, four images per split, 64x64 input, batch two.
- Independent enhancer training: one epoch, validation mAP selection, checkpoint save.
- Detection-guided training: one epoch initialized from enhancer checkpoint,
  frozen baseline detector, checkpoint save.
- Learned export, held-out synthetic test evaluation, metrics JSON and plots.
- Original/enhanced image prediction and image outputs.
- Python compilation and command-line help.

A test-only export initially failed because Ultralytics requires a validation
folder even for test evaluation. Export now includes validation images when test
is requested alone; test evaluation passed after this fix.

Synthetic accuracy is not meaningful and is deliberately not reported as model
quality. No ExDark data, trained real-world weights, production validation,
GPU latency measurements or improvement claims are included.

The CPU integration run needed a local font because automatic font downloading
was slow in this environment. In an offline environment, place a suitable TTF in
Ultralytics' user configuration directory as Arial.ttf before running plotting.

Reproduce tests: `python -m pytest -q`
Reproduce integration smoke: `python smoke_pipeline.py`
Use a fresh directory for smoke_artifacts; the script protects existing run output.
Resume behavior and full-size ExDark training have not been integration-tested.
