"""Low-Light Enhancement & Detection — Interactive Live Demo.

Gradio web app for real-time low-light photo enhancement with
structure-preserving quality controls, illumination curve inspection,
and YOLO object detection comparison.

Run:
    python demo_app.py              # local demo at http://localhost:7860
    python demo_app.py --share      # public shareable link
"""

import argparse
import io
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from lowlight.engine import load_enhancer
from lowlight.hq_enhance import enhance_high_quality, render_curve_map


# ─── Model Caches (avoid reloading checkpoints on every click) ──────────────

_ENHANCER_CACHE: Dict[Tuple[str, str, int], torch.nn.Module] = {}
_DETECTOR_CACHE: Dict[str, YOLO] = {}


def get_enhancer(ckpt_path: str, device: torch.device, scale_factor: int = 1) -> torch.nn.Module:
    key = (str(Path(ckpt_path).resolve()), str(device), int(scale_factor))
    if key not in _ENHANCER_CACHE:
        model = load_enhancer(ckpt_path, device, scale_factor=scale_factor).eval()
        _ENHANCER_CACHE[key] = model
    return _ENHANCER_CACHE[key]


def get_detector(ckpt_path: str) -> YOLO:
    key = str(Path(ckpt_path).resolve()) if Path(ckpt_path).exists() else str(ckpt_path)
    if key not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[key] = YOLO(ckpt_path)
    return _DETECTOR_CACHE[key]


def scan_checkpoints() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Scan the project for enhancer and detector .pt files and classify them."""
    enhancers: Dict[str, str] = {}
    detectors: Dict[str, str] = {}

    for pt in sorted(Path(".").rglob("*.pt")):
        if ".venv" in pt.parts or "pytest" in str(pt):
            continue
        path_str = str(pt)
        try:
            ck = torch.load(pt, map_location="cpu", weights_only=False)
            if isinstance(ck, dict) and "enhancer" in ck:
                if "guided" in path_str:
                    label = f"✨ Guided Enhancer [{path_str}]"
                elif "joint" in path_str and "detector" not in path_str:
                    label = f"⚡ Joint JE-YOLO Enhancer [{path_str}]"
                elif "enhancer" in path_str:
                    label = f"🎨 Pretrained Enhancer [{path_str}]"
                else:
                    label = f"Enhancer [{path_str}]"
                enhancers[label] = path_str
            elif isinstance(ck, dict) and ("model" in ck or "train_args" in ck):
                if "joint" in path_str:
                    label = f"🎯 Joint Adapted Detector [{path_str}]"
                elif "baseline" in path_str:
                    label = f"🎯 Baseline Detector [{path_str}]"
                elif "yolo" in path_str.lower():
                    label = f"🎯 Pretrained YOLOv8n [{path_str}]"
                else:
                    label = f"Detector [{path_str}]"
                detectors[label] = path_str
        except Exception:
            continue

    if not enhancers:
        enhancers["Default (smoke_artifacts/guided/best.pt)"] = "smoke_artifacts/guided/best.pt"
    if not detectors:
        detectors["Pretrained YOLOv8n (yolov8n.pt)"] = "yolov8n.pt"

    return enhancers, detectors


def _render_histogram(orig_pil: Image.Image, enh_pil: Image.Image) -> Optional[Image.Image]:
    """Plot luminance histograms before vs after enhancement as a PIL Image."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        orig_l = np.array(orig_pil.convert("L")).ravel()
        enh_l = np.array(enh_pil.convert("L")).ravel()

        fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=120)
        ax.hist(orig_l, bins=64, range=(0, 255), alpha=0.55, color="#64748b", label=f"Original (mean {orig_l.mean():.1f})")
        ax.hist(enh_l, bins=64, range=(0, 255), alpha=0.60, color="#f59e0b", label=f"Enhanced (mean {enh_l.mean():.1f})")
        ax.set_title("Luminance Distribution (Before vs After)")
        ax.set_xlabel("Pixel Intensity (0–255)")
        ax.set_ylabel("Pixel Count")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    except Exception:
        return None


def _summarize_detections(res) -> Tuple[Counter, Dict[str, float]]:
    counts: Counter = Counter()
    confs: Dict[str, list] = defaultdict(list)
    names = res.names
    if res.boxes is not None and len(res.boxes) > 0:
        for cls_id, conf in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            name = str(names.get(int(cls_id), int(cls_id)))
            counts[name] += 1
            confs[name].append(float(conf))
    mean_confs = {k: float(np.mean(v)) for k, v in confs.items()}
    return counts, mean_confs


def process_image(
    input_img: Optional[Image.Image],
    enhancer_ckpt: str,
    detector_ckpt: str,
    mode: str,
    scale_factor: int,
    conf_thresh: float,
    device_str: str,
    color_correction: bool,
    sharpness: float,
    denoise: float,
):
    """Run the full enhance → detect pipeline on an uploaded image."""
    if input_img is None:
        return None, None, None, None, None, None, "⚠️ Please upload an image first."

    input_rgb = input_img.convert("RGB")
    device = torch.device(
        device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu"
    )
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    # --- 1. Enhancement ---
    curve_vis = None
    method_note = mode
    try:
        model = None
        if mode in ("HQ Adaptive (Curve + Post-Process)", "Raw Learned Curve (Zero-DCE++)"):
            model = get_enhancer(enhancer_ckpt, device, scale_factor=int(scale_factor))

        sync()
        start_enh = time.perf_counter()

        if mode == "HQ Adaptive (Curve + Post-Process)":
            enhanced_pil, meta = enhance_high_quality(
                model,
                input_rgb,
                device=str(device),
                color_correction=color_correction,
                sharpness=sharpness,
                denoise_strength=denoise,
                return_meta=True,
            )
            method_note = meta.get("method", mode)
            curve_vis = render_curve_map(meta.get("curve_map"))
        elif mode == "Raw Learned Curve (Zero-DCE++)":
            arr = np.array(input_rgb, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                out_t, curve_t = model(x)
            out_np = out_t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            curve_np = curve_t[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
            enhanced_pil = Image.fromarray(np.round(out_np * 255.0).astype(np.uint8))
            curve_vis = render_curve_map(curve_np)
            method_note = "raw_learned_curve"
        elif mode == "Adaptive Gamma + CLAHE":
            enhanced_pil, meta = enhance_high_quality(
                None,
                input_rgb,
                device=str(device),
                color_correction=color_correction,
                sharpness=sharpness,
                denoise_strength=denoise,
                return_meta=True,
            )
            method_note = f"adaptive_gamma (γ={meta['gamma_used']:.2f})"
        else:  # CLAHE Baseline
            arr = np.array(input_rgb)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            enhanced_pil = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
            method_note = "clahe_baseline"

        sync()
        enh_time = (time.perf_counter() - start_enh) * 1000.0
    except Exception as e:
        msg = (
            f"❌ **Enhancement Error:** `{enhancer_ckpt}`\n\n"
            f"*Details:* `{e}`\n\n"
            "> **Tip:** Select an **Enhancer** checkpoint (e.g. `runs/guided_v1/best.pt`)."
        )
        return input_rgb, input_rgb, input_rgb, input_rgb, None, None, msg

    # --- 2. Detection (model cached; measure inference only) ---
    det_error = None
    try:
        detector = get_detector(detector_ckpt)
        det_device = str(device)

        sync()
        t_det0 = time.perf_counter()
        res_enh = detector.predict(enhanced_pil, device=det_device, conf=conf_thresh, verbose=False)[0]
        sync()
        det_time = (time.perf_counter() - t_det0) * 1000.0

        res_orig = detector.predict(input_rgb, device=det_device, conf=conf_thresh, verbose=False)[0]

        orig_det_pil = Image.fromarray(res_orig.plot()[..., ::-1])
        enh_det_pil = Image.fromarray(res_enh.plot()[..., ::-1])

        orig_counts, orig_confs = _summarize_detections(res_orig)
        enh_counts, enh_confs = _summarize_detections(res_enh)
        orig_count = sum(orig_counts.values())
        enh_count = sum(enh_counts.values())
    except Exception as e:
        orig_det_pil = input_rgb
        enh_det_pil = enhanced_pil
        det_time = 0.0
        orig_counts, orig_confs = Counter(), {}
        enh_counts, enh_confs = Counter(), {}
        orig_count = enh_count = 0
        det_error = str(e)

    total_time = enh_time + det_time
    hist_pil = _render_histogram(input_rgb, enhanced_pil)

    # --- 3. Brightness & Detection Breakdown ---
    orig_lum = float(np.array(input_rgb.convert("L")).mean())
    enh_lum = float(np.array(enhanced_pil.convert("L")).mean())
    gain = ((enh_lum - orig_lum) / max(orig_lum, 1e-5)) * 100.0
    gain_sign = "+" if gain >= 0 else ""

    lines = [
        "### ⚡ Performance & Image Metrics",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Active Method | `{method_note}` |",
        f"| Device | `{device}` |",
        f"| Enhancement Latency (warm) | `{enh_time:.1f} ms` |",
        f"| Detection Latency (enhanced) | `{det_time:.1f} ms` |",
        f"| End-to-End Pipeline | `{total_time:.1f} ms` |",
        f"| Original Brightness | `{orig_lum:.1f} / 255` |",
        f"| Enhanced Brightness | `{enh_lum:.1f} / 255` |",
        f"| Brightness Gain | `{gain_sign}{gain:.1f}%` |",
        f"| Detections (Original → Enhanced) | `{orig_count}` → `{enh_count}` |",
        f"| Output Resolution | `{enhanced_pil.size[0]}×{enhanced_pil.size[1]}` |",
    ]

    all_classes = sorted(set(orig_counts.keys()) | set(enh_counts.keys()))
    if all_classes:
        lines.append("\n### 🎯 Per-Class Detection Breakdown")
        lines.append("| Class | Original (avg conf) | Enhanced (avg conf) |")
        lines.append("|-------|---------------------|---------------------|")
        for cls_name in all_classes:
            oc = orig_counts.get(cls_name, 0)
            ec = enh_counts.get(cls_name, 0)
            o_str = f"{oc} ({orig_confs[cls_name]:.2f})" if oc else "0"
            e_str = f"{ec} ({enh_confs[cls_name]:.2f})" if ec else "0"
            lines.append(f"| {cls_name} | {o_str} | {e_str} |")

    if det_error:
        lines.append(f"\n> ⚠️ **Detector Warning:** `{det_error}`")

    metrics_md = "\n".join(lines)
    return enhanced_pil, orig_det_pil, enh_det_pil, input_rgb, curve_vis, hist_pil, metrics_md


# ─── Gradio UI ──────────────────────────────────────────────────────────────


def build_demo():
    enhancers, detectors = scan_checkpoints()
    state_maps = {"enhancers": enhancers, "detectors": detectors}

    sample_images = []
    for d in [Path("data/exdark/images/val"), Path("data/exdark/images/test")]:
        if d.exists():
            sample_images += [str(p) for p in sorted(d.glob("*.png"))[:3]]

    available_devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with gr.Blocks(
        title="Low-Light Image Enhancement & Detection Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
# 🌙 Low-Light Image Enhancement & Detection — Live Demo
Upload any low-light or dark photo to compare Zero-DCE++ learned curve enhancement,
adaptive HQ post-processing, and conventional baselines alongside YOLOv8 object detection.
            """
        )

        with gr.Row():
            # ── Left sidebar: controls ──
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="📷 Upload Low-Light Photo")

                mode_dd = gr.Dropdown(
                    choices=[
                        "HQ Adaptive (Curve + Post-Process)",
                        "Raw Learned Curve (Zero-DCE++)",
                        "Adaptive Gamma + CLAHE",
                        "CLAHE Baseline",
                    ],
                    value="HQ Adaptive (Curve + Post-Process)",
                    label="Enhancement Mode",
                )

                enhancer_dd = gr.Dropdown(
                    choices=list(enhancers.keys()),
                    value=list(enhancers.keys())[0],
                    label="Enhancer Checkpoint",
                )
                detector_dd = gr.Dropdown(
                    choices=list(detectors.keys()),
                    value=list(detectors.keys())[0],
                    label="Detector Checkpoint",
                )
                refresh_btn = gr.Button("🔄 Refresh Checkpoints", size="sm")

                with gr.Row():
                    scale_dd = gr.Dropdown(
                        choices=[1, 2, 4, 8],
                        value=1,
                        label="Curve Scale Factor (Zero-DCE++)",
                    )
                    device_radio = gr.Radio(
                        available_devices, value=default_device, label="Device"
                    )

                conf_slider = gr.Slider(
                    0.05, 0.95, 0.25, step=0.05, label="Detection Confidence"
                )

                gr.Markdown("### 🎛️ HQ Post-Processing Controls")
                color_fix = gr.Checkbox(
                    value=True, label="Color Correction (fix color cast)"
                )
                sharp_slider = gr.Slider(
                    0.0, 1.0, 0.3, step=0.05, label="Sharpness (0 = off, 1 = max)"
                )
                denoise_slider = gr.Slider(
                    0.0, 1.0, 0.5, step=0.05, label="Bilateral Denoise (0 = off, 1 = max)"
                )

                run_btn = gr.Button("⚡ Enhance & Detect", variant="primary", size="lg")

                if sample_images:
                    gr.Examples(
                        examples=sample_images,
                        inputs=input_img,
                        label="Sample Low-Light Images",
                    )

            # ── Right: outputs ──
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("🖼️ Before vs After"):
                        with gr.Row():
                            orig_out = gr.Image(type="pil", label="Original", interactive=False)
                            enh_out = gr.Image(type="pil", label="Enhanced", interactive=False)

                    with gr.TabItem("🔍 Object Detection"):
                        with gr.Row():
                            orig_det = gr.Image(
                                type="pil", label="Detections on Original", interactive=False
                            )
                            enh_det = gr.Image(
                                type="pil", label="Detections on Enhanced", interactive=False
                            )

                    with gr.TabItem("📈 Curve Map & Histogram"):
                        with gr.Row():
                            curve_out = gr.Image(
                                type="pil",
                                label="Predicted Illumination Curve Heatmap",
                                interactive=False,
                            )
                            hist_out = gr.Image(
                                type="pil",
                                label="Luminance Histogram",
                                interactive=False,
                            )

                metrics_md = gr.Markdown(
                    "### 📊 Upload a photo and click **Enhance & Detect**"
                )

        def _refresh():
            new_enh, new_det = scan_checkpoints()
            state_maps["enhancers"] = new_enh
            state_maps["detectors"] = new_det
            return (
                gr.Dropdown(choices=list(new_enh.keys()), value=list(new_enh.keys())[0]),
                gr.Dropdown(choices=list(new_det.keys()), value=list(new_det.keys())[0]),
            )

        def _run(img, enh_label, det_label, mode, sf, conf, dev, cc, sharp, dn):
            enh_ckpt = state_maps["enhancers"].get(enh_label, enh_label)
            det_ckpt = state_maps["detectors"].get(det_label, det_label)
            enh_img, o_det, e_det, orig, curve_vis, hist_vis, md = process_image(
                img, enh_ckpt, det_ckpt, mode, sf, conf, dev, cc, sharp, dn
            )
            return orig, enh_img, o_det, e_det, curve_vis, hist_vis, md

        refresh_btn.click(fn=_refresh, inputs=[], outputs=[enhancer_dd, detector_dd])
        run_btn.click(
            fn=_run,
            inputs=[
                input_img,
                enhancer_dd,
                detector_dd,
                mode_dd,
                scale_dd,
                conf_slider,
                device_radio,
                color_fix,
                sharp_slider,
                denoise_slider,
            ],
            outputs=[orig_out, enh_out, orig_det, enh_det, curve_out, hist_out, metrics_md],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LowLight Gradio Live Demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo()
    print(f"\n🌙 Launching demo → http://localhost:{args.port}\n")
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
