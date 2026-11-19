"""
FastAPI Server serving the Full Interactive Low-Light Detection Dashboard
at http://localhost:7860 with live backend inference endpoint (/api/predict).
"""
import io, time, base64, argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO

from lowlight.engine import load_enhancer
from lowlight.hq_enhance import enhance_high_quality, render_curve_map

app = FastAPI(title="Low-Light Detection Dashboard Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DASHBOARD_FILE = Path(__file__).parent / "dashboard.html"

# Cache models
ENHANCER_MODEL = None
DETECTOR_MODEL = None

def get_models():
    global ENHANCER_MODEL, DETECTOR_MODEL
    if ENHANCER_MODEL is None:
        ckpt = "runs_live/joint/best.pt"
        if not Path(ckpt).exists():
            ckpt = "runs/joint_v1/best.pt" if Path("runs/joint_v1/best.pt").exists() else None
        try:
            ENHANCER_MODEL = load_enhancer(ckpt, DEVICE, scale_factor=4).eval()
        except Exception:
            ENHANCER_MODEL = load_enhancer(None, DEVICE, scale_factor=4).eval()

    if DETECTOR_MODEL is None:
        det_ckpt = "runs_live/joint/detector_best.pt"
        if not Path(det_ckpt).exists():
            det_ckpt = "runs/baseline_v1/weights/best.pt" if Path("runs/baseline_v1/weights/best.pt").exists() else "yolov8n.pt"
        try:
            DETECTOR_MODEL = YOLO(det_ckpt)
        except Exception:
            DETECTOR_MODEL = YOLO("yolov8n.pt")

    return ENHANCER_MODEL, DETECTOR_MODEL

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the complete motion-rich research dashboard directly at root /."""
    if not DASHBOARD_FILE.exists():
        return HTMLResponse("<h1>dashboard.html not found!</h1>", status_code=404)
    return HTMLResponse(DASHBOARD_FILE.read_text(encoding="utf-8"))

@app.post("/api/predict")
async def predict_image(
    file: UploadFile = File(...),
    scale_factor: int = Form(4),
    conf: float = Form(0.25),
    mode: str = Form("hq")
):
    """Run real PyTorch Zero-DCE++ and YOLOv8n inference on user-uploaded photo."""
    try:
        enhancer, detector = get_models()
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = pil_img.size

        # 1. Enhancement
        t0 = time.perf_counter()
        if mode == "hq":
            enh_pil, meta = enhance_high_quality(
                enhancer, pil_img, device=DEVICE,
                color_correction=True, sharpness=0.3, denoise_strength=0.5,
                return_meta=True
            )
            curve_pil = render_curve_map(meta.get("curve_map"))
        else:
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out_t, curve_t = enhancer(x)
            out_np = out_t[0].clamp(0,1).permute(1,2,0).cpu().numpy()
            enh_pil = Image.fromarray(np.round(out_np * 255.0).astype(np.uint8))
            curve_pil = render_curve_map(curve_t[0].clamp(-1,1).permute(1,2,0).cpu().numpy())

        enh_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Detection on Original vs Enhanced
        t1 = time.perf_counter()
        res_orig = detector.predict(pil_img, device=DEVICE, conf=conf, verbose=False)[0]
        res_enh = detector.predict(enh_pil, device=DEVICE, conf=conf, verbose=False)[0]
        det_ms = (time.perf_counter() - t1) * 1000.0

        orig_det_pil = Image.fromarray(res_orig.plot()[..., ::-1])
        enh_det_pil = Image.fromarray(res_enh.plot()[..., ::-1])

        def to_b64(im, fmt="JPEG"):
            buf = io.BytesIO()
            im.save(buf, format=fmt, quality=90)
            return "data:image/" + fmt.lower() + ";base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        boxes = []
        if res_enh.boxes is not None and len(res_enh.boxes) > 0:
            for i, (b, c, cf) in enumerate(zip(res_enh.boxes.xyxy.tolist(), res_enh.boxes.cls.tolist(), res_enh.boxes.conf.tolist())):
                cls_name = str(detector.names.get(int(c), int(c)))
                boxes.append({
                    "id": f"#{i+1}",
                    "class": cls_name,
                    "conf": round(float(cf), 3),
                    "bbox": [round(v, 1) for v in b]
                })

        orig_lum = float(np.array(pil_img.convert("L")).mean())
        enh_lum = float(np.array(enh_pil.convert("L")).mean())
        gain = ((enh_lum - orig_lum) / max(orig_lum, 1e-4)) * 100.0

        return JSONResponse({
            "success": True,
            "orig_img": to_b64(pil_img, "JPEG"),
            "enh_img": to_b64(enh_pil, "JPEG"),
            "orig_det": to_b64(orig_det_pil, "JPEG"),
            "enh_det": to_b64(enh_det_pil, "JPEG"),
            "curve_map": to_b64(curve_pil, "PNG") if curve_pil else None,
            "orig_detections": len(res_orig.boxes),
            "enh_detections": len(res_enh.boxes),
            "boxes": boxes,
            "orig_lum": round(orig_lum, 1),
            "enh_lum": round(enh_lum, 1),
            "gain_pct": round(gain, 1),
            "enh_ms": round(enh_ms, 1),
            "det_ms": round(det_ms, 1),
            "total_ms": round(enh_ms + det_ms, 1),
            "res": f"{w} × {h}"
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    print(f"\n🚀 Serving Low-Light Research Dashboard directly at: http://localhost:{args.port}\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
