# Low-Light Detection & Enhancement Training Pipeline

A deep learning pipeline and interactive dashboard for object detection in extreme low-light environments, combining a Zero-DCE++ inspired curve enhancer with YOLOv8.

![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-v8n-00FFFF)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🌟 Key Highlights

- **Zero-DCE++ Curve Enhancement**: Non-reference, lightweight deep curve estimation network trained without paired bright ground truth images.
- **HQ Adaptive Post-Processing**: LAB-space CLAHE with bilateral filtering for detail preservation and contrast restoration.
- **Multi-Stage Training Workflows**:
  1. *Baseline*: Raw-image YOLOv8 detector training.
  2. *Independent Enhancer Pretraining*: Unsupervised non-reference enhancement loss (spatial consistency, exposure control, color constancy, TV smoothness).
  3. *Detection-Guided Enhancement*: Freezes detector and guides enhancer weights via detection loss gradients.
  4. *End-to-End Joint Fine-Tuning (JE-YOLO)*: Simultaneous co-adaptation of enhancer and detector with AdamW and cosine learning rate schedule.
- **Interactive Cyberpunk Dashboard**: Standalone FastAPI + HTML5 live demo interface supporting drag-and-drop image inference, side-by-side comparison, bounding-box detection viewer, and dynamic enhancement mode selection.

---

## 📁 Repository Structure

```
LowLight-Detection-Training-Pipeline/
├── README.md                          # Project documentation
├── .gitignore                         # Git ignore patterns
└── lowlight-training/
    ├── dashboard.html                 # Cyberpunk interactive live dashboard
    ├── dashboard_server.py            # FastAPI backend server for dashboard
    ├── demo_app.py                    # Standalone inference demo
    ├── smoke_pipeline.py              # End-to-end smoke test script
    ├── run_live_pipeline.py           # Full execution pipeline runner
    ├── compare_results.py             # Metrics comparator and table generator
    ├── SHOWCASE.bat                   # 1-click batch launcher for demo dashboard
    ├── requirements.txt               # Pinned Python package dependencies
    ├── lowlight/                      # Core deep learning library
    │   ├── enhancer.py                # Zero-DCE++ lightweight curve network
    │   ├── engine.py                  # Training loops, losses, joint fine-tuning
    │   ├── hq_enhance.py              # LAB CLAHE & bilateral post-processing
    │   ├── data.py                    # ExDark parser, YOLO label conversion
    │   └── __main__.py                # CLI commands (baseline, pretrain, guided, joint, benchmark)
    ├── tests/
    │   └── test_pipeline.py           # Unit and regression test suite (10/10 passing)
    └── notebooks/
        └── Colab_Training.ipynb       # Google Colab cloud training notebook
```

---

## 🚀 Quick Start

### 1. Installation

```bash
cd lowlight-training
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard

Launch the live studio with one command:

```bash
python dashboard_server.py --port 7860
```
Open **`http://localhost:7860`** in any web browser to upload images, test real-time curve enhancement, and view YOLOv8 detections.

---

## 🧪 Training Stages

### Raw Baseline Detector
```bash
python -m lowlight baseline --data data/exdark --out runs/baseline --epochs 80 --batch 4 --size 640
```

### Independent Curve Enhancer Pretraining
```bash
python -m lowlight pretrain --data data/exdark --detector runs/baseline/weights/best.pt --out runs/enhancer --epochs 50 --batch 4 --size 640
```

### Detection-Guided Training
```bash
python -m lowlight guided --data data/exdark --detector runs/baseline/weights/best.pt --init runs/enhancer/best.pt --out runs/guided --epochs 30 --batch 4 --lr 0.00005 --det-weight 0.1
```

### End-to-End Joint Fine-Tuning (JE-YOLO)
```bash
python -m lowlight joint --data data/exdark --detector runs/baseline/weights/best.pt --init runs/enhancer/best.pt --out runs/joint --epochs 30 --batch 4 --lr 0.00005 --det-lr 0.00001 --det-weight 0.1
```

---

## 📊 Evaluation & Benchmarking

Evaluate mAP50, mAP50-95, precision, and recall on the test split:
```bash
python -m lowlight evaluate --data data/joint --detector runs/joint/detector_best.pt --out results/joint_test --split test
```

Measure warmed end-to-end latency:
```bash
python -m lowlight benchmark --enhancer runs/joint/best.pt --detector runs/joint/detector_best.pt --size 640 --runs 50 --warmup 10 --out results/benchmark.json
```

---

## 🔬 Testing

Run the automated test suite:
```bash
pytest -q tests/test_pipeline.py
```
*(All 10 tests passing)*

---

## 📜 References
- **Zero-DCE / Zero-DCE++**: Chongyi Li et al., CVPR 2020 / TPAMI 2021.
- **ExDark Dataset**: Y. P. Loh and C. S. Chan, "Getting to know low-light images with the Exclusively Dark dataset", CVIU 2019.
- **Ultralytics YOLOv8**: Glenn Jocher et al.
