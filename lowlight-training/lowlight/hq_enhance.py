"""High-Quality Low-Light Image Enhancer.

A structure-preserving, native-resolution enhancement pipeline that:
  1. Applies the Zero-DCE++ CurveEnhancer when it produces valid brightening
  2. Safely guards against degenerate/untrained checkpoints darkening already-dark scenes
  3. Falls back to continuous analytical adaptive gamma + LAB-space CLAHE local contrast
  4. Applies adaptive color correction to prevent purple/magenta color casts
  5. Applies edge-preserving bilateral denoising to reduce dark-region sensor noise
  6. Applies subtle unsharp-mask sharpening without distorting facial/object geometry

Works at NATIVE resolution — no downscaling of the output image.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _perceptual_luminance(img_np: np.ndarray) -> np.ndarray:
    """Rec.709 perceptual luminance in [0, 1], shape [H, W]."""
    return (
        0.2126 * img_np[:, :, 0]
        + 0.7152 * img_np[:, :, 1]
        + 0.0722 * img_np[:, :, 2]
    )


def enhance_high_quality(
    model: Optional[torch.nn.Module],
    input_image: Image.Image,
    device: str = "cpu",
    color_correction: bool = True,
    sharpness: float = 0.3,
    denoise_strength: float = 0.5,
    target_brightness: float = 0.45,
    local_contrast: float = 0.35,
    return_meta: bool = False,
) -> Union[Image.Image, Tuple[Image.Image, Dict[str, Any]]]:
    """Enhance a low-light image at native resolution with color & structure preservation.

    Args:
        model: Loaded CurveEnhancer model (already in eval mode), or None to use adaptive fallback.
        input_image: PIL.Image (converted to RGB internally if needed).
        device: 'cpu' or 'cuda:0'.
        color_correction: Apply adaptive white-balance correction to prevent color casts.
        sharpness: Sharpening blend factor (0=none, 1=max). 0.3 is subtle.
        denoise_strength: Noise reduction blend (0=none, 1=max). 0.5 is moderate.
        target_brightness: Target mean luminance for adaptive gamma fallback.
        local_contrast: Subtle local contrast boost strength in [0, 1].
        return_meta: If True, return ``(enhanced_pil, meta_dict)``.

    Returns:
        PIL.Image (or tuple of ``(PIL.Image, meta_dict)`` when ``return_meta=True``).
    """
    rgb_img = input_image.convert("RGB")
    dev = torch.device(device)
    orig_size = rgb_img.size  # (W, H)
    img_np = np.array(rgb_img, dtype=np.float32) / 255.0  # [H, W, 3]

    orig_lum = float(_perceptual_luminance(img_np).mean())
    curve_result, curve_map, inverted = (
        _run_curve_enhancer(model, img_np, dev, orig_lum)
        if model is not None
        else (None, None, False)
    )

    curve_lum = (
        float(_perceptual_luminance(curve_result).mean())
        if curve_result is not None
        else 0.0
    )

    # Use learned curve model when:
    # - Image is dark (orig_lum < 0.40) and model meaningfully brightens it (>10% gain), OR
    # - Image is already moderately lit (orig_lum >= 0.40) and model doesn't collapse it
    if curve_result is not None and (
        (orig_lum < 0.40 and curve_lum > orig_lum * 1.10)
        or (orig_lum >= 0.40 and curve_lum >= orig_lum * 0.75)
    ):
        enhanced = curve_result
        method_used = "learned_curve_inverted" if inverted else "learned_curve"
        gamma_used = None
    else:
        enhanced, gamma_used = _adaptive_gamma(
            img_np,
            target_brightness=target_brightness,
            local_contrast=local_contrast,
        )
        method_used = "adaptive_gamma_clahe"

    # ── Post-processing ────────────────────────────────────────
    if color_correction:
        enhanced = _adaptive_color_correction(img_np, enhanced)

    if denoise_strength > 0:
        enhanced = _mild_denoise(enhanced, denoise_strength)

    if sharpness > 0:
        enhanced = _structure_preserving_sharpen(enhanced, sharpness)

    enhanced = np.clip(enhanced, 0.0, 1.0)
    enh_lum = float(_perceptual_luminance(enhanced).mean())

    enhanced_u8 = np.round(enhanced * 255.0).astype(np.uint8)
    result = Image.fromarray(enhanced_u8)

    if result.size != orig_size:
        result = result.resize(orig_size, Image.Resampling.LANCZOS)

    if return_meta:
        meta: Dict[str, Any] = {
            "method": method_used,
            "orig_luminance": orig_lum,
            "enhanced_luminance": enh_lum,
            "gamma_used": gamma_used,
            "curve_inverted": inverted,
            "curve_map": curve_map,
        }
        return result, meta
    return result


# ─── Core enhancement methods ───────────────────────────────────────────────


def _run_curve_enhancer(
    model: torch.nn.Module,
    img_np: np.ndarray,
    dev: torch.device,
    orig_lum: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    """Run the Zero-DCE++ curve enhancer. Returns (enhanced_np, curve_np, inverted)."""
    try:
        x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(dev)

        # Pad to multiple of 4 for clean conv/downsampling boundaries
        _, _, h, w = x.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            enhanced, curve = model(x)

        inverted = False
        # Only invert curves if an ALREADY DARK image is being darkened by a
        # degenerate/under-trained checkpoint (do not blow out bright images).
        if orig_lum < 0.35 and curve.mean().item() > 0.05:
            out = x.clone()
            neg_curve = -curve
            iters = getattr(model, "iterations", 8)
            for _ in range(iters):
                out = out + neg_curve * (out.square() - out)
            enhanced = out
            curve = neg_curve
            inverted = True

        if pad_h > 0 or pad_w > 0:
            enhanced = enhanced[:, :, :h, :w]
            curve = curve[:, :, :h, :w]

        enh_np = enhanced[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        curve_np = curve[0].clamp(-1.0, 1.0).permute(1, 2, 0).cpu().numpy()
        return enh_np, curve_np, inverted
    except Exception:
        return None, None, False


def _adaptive_gamma(
    img_np: np.ndarray,
    target_brightness: float = 0.45,
    local_contrast: float = 0.35,
) -> Tuple[np.ndarray, float]:
    """Continuous analytical adaptive gamma correction + subtle local contrast boost.

    Computes gamma analytically so the image's mean luminance moves smoothly
    toward ``target_brightness``: ``lum^gamma ≈ target_brightness``.
    """
    lum = float(_perceptual_luminance(img_np).mean())
    target = float(np.clip(target_brightness, 0.15, 0.75))

    if lum >= target:
        gamma = 1.0
    else:
        safe_lum = float(np.clip(lum, 0.015, 0.95))
        gamma = float(np.log(target) / np.log(safe_lum))
        gamma = float(np.clip(gamma, 0.25, 1.0))

    corrected = np.power(np.clip(img_np, 0.0, 1.0), gamma)

    if local_contrast > 0:
        corrected = _local_contrast_boost(corrected, strength=local_contrast)

    return np.clip(corrected, 0.0, 1.0), gamma


def _local_contrast_boost(
    img_np: np.ndarray,
    strength: float = 0.35,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Structure-preserving local contrast enhancement on luminance only.

    Uses LAB-space CLAHE via OpenCV when available, blended smoothly with the
    input to avoid over-amplifying dark-region noise. Falls back to a
    multi-scale unsharp luminance contrast boost in pure NumPy/PIL.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return np.clip(img_np, 0.0, 1.0)

    if cv2 is not None:
        u8 = np.round(np.clip(img_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(int(tile_size), int(tile_size)),
        )
        l_eq = clahe.apply(lab[:, :, 0])
        lab[:, :, 0] = np.round(
            lab[:, :, 0].astype(np.float32) * (1.0 - strength)
            + l_eq.astype(np.float32) * strength
        ).astype(np.uint8)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return np.clip(rgb, 0.0, 1.0)

    # Pure NumPy/PIL fallback: preserve & gently boost deviations from local mean
    lum = _perceptual_luminance(img_np)[:, :, np.newaxis]  # [H, W, 1]
    h, w = lum.shape[:2]
    small_h = max(h // max(tile_size * 2, 1), 1)
    small_w = max(w // max(tile_size * 2, 1), 1)

    lum_pil = Image.fromarray(np.round(np.clip(lum[:, :, 0], 0.0, 1.0) * 255.0).astype(np.uint8))
    local_mean = (
        np.array(
            lum_pil.resize((small_w, small_h), Image.Resampling.BILINEAR).resize(
                (w, h), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )[:, :, np.newaxis]

    gain = 1.0 + 0.5 * strength
    new_lum = np.clip(local_mean + gain * (lum - local_mean), 0.0, 1.0)
    scale = new_lum / (lum + 1e-6)
    scale = np.clip(scale, 0.5, 2.0)
    return np.clip(img_np * scale, 0.0, 1.0)


# ─── Color correction ───────────────────────────────────────────────────────


def _adaptive_color_correction(
    original: np.ndarray,
    enhanced: np.ndarray,
    blend: float = 0.85,
) -> np.ndarray:
    """Fix color cast by preserving the original image's RGB channel ratios.

    Prevents purple/magenta color shifts that can arise from extreme curves.
    """
    orig_means = original.mean(axis=(0, 1)) + 1e-6
    enh_means = enhanced.mean(axis=(0, 1)) + 1e-6

    orig_ratios = orig_means / orig_means.mean()
    enh_ratios = enh_means / enh_means.mean()

    correction = np.clip(orig_ratios / enh_ratios, 0.6, 1.6)
    correction = 1.0 + blend * (correction - 1.0)

    corrected = enhanced * correction[np.newaxis, np.newaxis, :]
    return np.clip(corrected, 0.0, 1.0)


# ─── Denoising ──────────────────────────────────────────────────────────────


def _mild_denoise(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Edge-preserving denoising. Suppresses flat-region noise while keeping crisp edges."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return np.clip(img, 0.0, 1.0)

    u8 = np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if cv2 is not None:
        sigma_color = 20.0 + 45.0 * strength
        sigma_space = 5.0 + 10.0 * strength
        filtered = cv2.bilateralFilter(u8, d=5, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        filtered_np = filtered.astype(np.float32) / 255.0
        blended = img * (1.0 - 0.85 * strength) + filtered_np * (0.85 * strength)
        return np.clip(blended, 0.0, 1.0)

    pil_img = Image.fromarray(u8)
    smoothed = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
    smoothed_np = np.array(smoothed, dtype=np.float32) / 255.0

    gray = _perceptual_luminance(img)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edge_map = np.sqrt(gx**2 + gy**2)

    edge_max = float(np.percentile(edge_map, 95)) + 1e-8
    edge_mask = np.clip(edge_map / edge_max, 0.0, 1.0)[:, :, np.newaxis]

    blended = img * edge_mask + smoothed_np * (1.0 - edge_mask)
    result = img * (1.0 - strength) + blended * strength
    return np.clip(result, 0.0, 1.0)


# ─── Sharpening ─────────────────────────────────────────────────────────────


def _structure_preserving_sharpen(img: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Subtle unsharp-mask sharpening without creating halos or distorting geometry."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return np.clip(img, 0.0, 1.0)

    pil_img = Image.fromarray(np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    blurred_np = np.array(blurred, dtype=np.float32) / 255.0

    detail = img - blurred_np
    sharpened = img + strength * detail
    return np.clip(sharpened, 0.0, 1.0)


# ─── Visualization Helpers ──────────────────────────────────────────────────


def render_curve_map(curve_np: Optional[np.ndarray]) -> Optional[Image.Image]:
    """Render a 3-channel curve parameter map in [-1, 1] as a visual heatmap PIL Image.

    Negative values (brightening strength) map to warmer/brighter intensities.
    """
    if curve_np is None:
        return None
    # Mean brightening strength across RGB channels: -curve in [-1, 1] -> [0, 1]
    strength = np.clip((-curve_np.mean(axis=2) + 1.0) * 0.5, 0.0, 1.0)
    u8 = np.round(strength * 255.0).astype(np.uint8)
    if cv2 is not None:
        colored_bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        return Image.fromarray(cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB))
    return Image.fromarray(np.stack([u8, u8, u8], axis=-1))
