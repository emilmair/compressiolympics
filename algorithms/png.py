"""Lossless PNG baseline using Pillow/zlib."""

from __future__ import annotations

from pathlib import Path

from ._utils import load_rgb_or_rgba, validate_amount
from PIL import Image, ImageOps


def compress(
    image_path: str | Path,
    output_path: str | Path,
    compression_amount: float,
) -> None:
    """Re-encode as PNG without changing pixels.

    For PNG, compression_amount controls encoder effort (zlib level 1-9),
    not a lossy quality setting. PSNR should therefore be infinite for the
    8-bit RGB(A) content that the benchmark compares.
    """
    validate_amount(compression_amount)
    image = load_rgb_or_rgba(image_path, preserve_alpha=True)
    image.save(
        output_path,
        format="PNG",
        compress_level=0,
        optimize=False,
    )


def decompress(image_path: str | Path, output_path: str | Path) -> None:
    """Decode the PNG and write a normalized PNG reconstruction."""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        image.save(output_path, format="PNG")
