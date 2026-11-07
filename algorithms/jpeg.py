"""Baseline lossy JPEG encoder using Pillow/libjpeg."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ._utils import decode_to_png, load_rgb_or_rgba, validate_amount


def compress(
    image_path: str | Path,
    output_path: str | Path,
    compression_amount: float,
) -> None:
    """Encode an image as JPEG.

    JPEG does not support transparency, so inputs are converted to RGB.
    Higher compression_amount values map to lower JPEG quality.
    """
    amount = validate_amount(compression_amount)
    quality = round(100.0 - 100.0 * amount)
    image = load_rgb_or_rgba(image_path, preserve_alpha=False)
    image.save(
        output_path,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling=2,
    )


def decompress(image_path: str | Path, output_path: str | Path) -> None:
    """Decode the JPEG bitstream and write a PNG reconstruction."""
    decode_to_png(image_path, output_path)
