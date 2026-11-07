"""Shared helpers for the benchmark algorithm modules.

This file starts with an underscore so benchmark.py will not discover it as
an algorithm module.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageOps


def validate_amount(compression_amount: float) -> float:
    """Validate a compression strength in the inclusive range (0, 1]."""
    if isinstance(compression_amount, bool):
        raise TypeError("compression_amount must be a float in the range (0, 1]")

    amount = float(compression_amount)
    if not math.isfinite(amount) or not 0.0 < amount <= 1.0:
        raise ValueError(
            f"compression_amount must be in the range (0, 1], got {compression_amount!r}"
        )
    return amount


def load_rgb_or_rgba(image_path: str | Path, *, preserve_alpha: bool = True) -> Image.Image:
    """Load an image, apply EXIF orientation, and normalize to 8-bit RGB(A)."""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        mode = "RGBA" if preserve_alpha and has_alpha else "RGB"
        return image.convert(mode)


def decode_to_png(compressed_path: str | Path, output_path: str | Path) -> None:
    """Decode a Pillow-supported compressed image and save it as PNG."""
    with Image.open(compressed_path) as source:
        image = ImageOps.exif_transpose(source)
        image.save(output_path, format="PNG")


def require_pillow_feature(feature_name: str, format_name: str) -> None:
    """Raise an actionable error if this Pillow build lacks a codec."""
    from PIL import features

    try:
        available = features.check(feature_name)
    except Exception as exc:  # pragma: no cover - defensive for Pillow variants
        raise RuntimeError(
            f"Could not check Pillow support for {format_name} ({feature_name})."
        ) from exc

    if not available:
        raise RuntimeError(
            f"This Pillow installation does not include {format_name} support. "
            "Install a Pillow wheel/build with the required codec enabled."
        )
