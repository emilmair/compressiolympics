"""Lossy WebP encoder using Pillow/libwebp."""

from __future__ import annotations

from pathlib import Path

from ._utils import (
    decode_to_png,
    load_rgb_or_rgba,
    require_pillow_feature,
    validate_amount,
)


def compress(
    image_path: str | Path,
    output_path: str | Path,
    compression_amount: float,
) -> None:
    """Encode as lossy WebP; preserve transparency when present."""
    require_pillow_feature("webp", "WebP")
    amount = validate_amount(compression_amount)
    image = load_rgb_or_rgba(image_path, preserve_alpha=True)
    image.save(
        output_path,
        format="WEBP",
        lossless=False,
        quality=round(100.0 - 100.0 * amount),
        method=6,
        exact=True,
    )


def decompress(image_path: str | Path, output_path: str | Path) -> None:
    """Decode the WebP bitstream and write a PNG reconstruction."""
    decode_to_png(image_path, output_path)
