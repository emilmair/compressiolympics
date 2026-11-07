"""Lossy AVIF encoder using Pillow/libavif."""

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
    """Encode as AVIF; preserve transparency when present."""
    require_pillow_feature("avif", "AVIF")
    amount = validate_amount(compression_amount)
    image = load_rgb_or_rgba(image_path, preserve_alpha=True)
    image.save(
        output_path,
        format="AVIF",
        quality=round(100.0 - 100.0 * amount),
        speed=6,
    )


def decompress(image_path: str | Path, output_path: str | Path) -> None:
    """Decode the AVIF bitstream and write a PNG reconstruction."""
    decode_to_png(image_path, output_path)
