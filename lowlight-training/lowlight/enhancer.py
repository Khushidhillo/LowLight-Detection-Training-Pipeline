"""Independent Zero-DCE++-inspired curve enhancer and non-reference losses.

Not an exact reproduction of official Zero-DCE++ checkpoints. Supports optional
multi-scale curve estimation via ``scale_factor`` (estimating curves on a
downsampled image and bilinearly upsampling the curve map to native resolution)
for fast, low-memory high-resolution enhancement while remaining 100%
state-dict compatible with checkpoints trained at ``scale_factor=1``.
"""

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class DSConv(nn.Module):
    """Depthwise-separable 3x3 convolution block."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, padding=1, groups=cin),
            nn.Conv2d(cin, cout, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CurveEnhancer(nn.Module):
    """Zero-DCE++-inspired deep curve estimation network.

    Predicts a 3-channel pixel-wise curve parameter map ``curve`` in ``[-1, 1]``
    and iteratively applies ``out = out + curve * (out^2 - out)``.
    When ``scale_factor > 1``, the curve map is estimated on a downsampled
    copy of ``x`` and bilinearly upsampled back to ``x``'s spatial resolution,
    dramatically reducing compute and VRAM on high-resolution inputs.
    """

    def __init__(self, width: int = 32, iterations: int = 8, scale_factor: int = 1):
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")

        self.width = int(width)
        self.iterations = int(iterations)
        self.scale_factor = int(scale_factor)

        self.layers = nn.ModuleList(
            [
                DSConv(3, width),
                DSConv(width, width),
                DSConv(width, width),
                DSConv(width, width),
                DSConv(width * 2, width),
                DSConv(width * 2, width),
                DSConv(width * 2, 3),
            ]
        )

    def get_config(self) -> Dict[str, int]:
        """Return serializable architecture hyperparameters."""
        return {
            "width": self.width,
            "iterations": self.iterations,
            "scale_factor": self.scale_factor,
        }

    def param_count(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def _estimate_curve(self, x: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.layers[0](x))
        b = F.relu(self.layers[1](a))
        c = F.relu(self.layers[2](b))
        d = F.relu(self.layers[3](c))
        e = F.relu(self.layers[4](torch.cat([c, d], dim=1)))
        f = F.relu(self.layers[5](torch.cat([b, e], dim=1)))
        return torch.tanh(self.layers[6](torch.cat([a, f], dim=1)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = x.shape
        if self.scale_factor > 1 and min(h, w) >= self.scale_factor * 4:
            x_down = F.interpolate(
                x,
                scale_factor=1.0 / self.scale_factor,
                mode="bilinear",
                align_corners=False,
            )
            curve_down = self._estimate_curve(x_down)
            curve = F.interpolate(
                curve_down,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            curve = self._estimate_curve(x)

        out = x
        for _ in range(self.iterations):
            out = out + curve * (out.square() - out)
        return out, curve


def enhancement_loss(
    x: torch.Tensor,
    out: torch.Tensor,
    curve: torch.Tensor,
    exposure: float = 0.6,
    w_spatial: float = 1.0,
    w_exp: float = 10.0,
    w_color: float = 5.0,
    w_tv: float = 200.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Zero-reference low-light enhancement loss.

    Combines spatial structure consistency, local patch exposure control,
    gray-world color constancy, and illumination curve total-variation smoothness.
    """
    # Color constancy (gray-world prior across RGB channel means)
    means = out.mean(dim=(2, 3))
    color = (
        (means[:, 0] - means[:, 1]).square()
        + (means[:, 0] - means[:, 2]).square()
        + (means[:, 1] - means[:, 2]).square()
    ).mean()

    # Exposure control on local patches (gracefully shrink patch size on tiny images)
    _, _, h, w = out.shape
    exp_k = min(16, h, w)
    exp = (F.avg_pool2d(out.mean(dim=1, keepdim=True), exp_k) - exposure).square().mean()

    # Spatial consistency of local luminance differences
    sp_k = min(4, max(1, h // 2), max(1, w // 2))
    src = F.avg_pool2d(x.mean(dim=1, keepdim=True), sp_k)
    dst = F.avg_pool2d(out.mean(dim=1, keepdim=True), sp_k)
    spatial = torch.zeros((), device=x.device, dtype=x.dtype)
    if src.shape[2] > 1:
        spatial = spatial + (
            (src[:, :, 1:, :] - src[:, :, :-1, :])
            - (dst[:, :, 1:, :] - dst[:, :, :-1, :])
        ).square().mean()
    if src.shape[3] > 1:
        spatial = spatial + (
            (src[:, :, :, 1:] - src[:, :, :, :-1])
            - (dst[:, :, :, 1:] - dst[:, :, :, :-1])
        ).square().mean()

    # Curve illumination smoothness (TV loss)
    tv = torch.zeros((), device=x.device, dtype=x.dtype)
    if curve.shape[2] > 1:
        tv = tv + (curve[:, :, 1:, :] - curve[:, :, :-1, :]).square().mean()
    if curve.shape[3] > 1:
        tv = tv + (curve[:, :, :, 1:] - curve[:, :, :, :-1]).square().mean()

    total = w_spatial * spatial + w_exp * exp + w_color * color + w_tv * tv
    return total, {
        "spatial": spatial,
        "exposure": exp,
        "color": color,
        "tv": tv,
    }
