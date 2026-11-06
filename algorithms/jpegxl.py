"""JPEG XL encoder/decoder using the Pillow JPEG XL plugin.

Install the codec with:
    python -m pip install pillow-jxl-plugin
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

try:
    # Importing the plugin registers the JXL format with Pillow.
    import pillow_jxl  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on user's environment
    raise ImportError(
        "jpegxl.py requires the Pillow JPEG XL plugin. Install it with:\n"
        "    python -m pip install --upgrade pillow-jxl-plugin"
    ) from exc

from ._utils import load_rgb_or_rgba, quality_from_amount, validate_amount


def compress(
    image_path: str | Path,
    output_path: str | Path,
    compression_amount: float,
) -> None:
    """Encode an image as JPEG XL (JXL).

    The benchmark compression amount is mapped to Pillow-plugin's quality
    scale: lower amounts use higher quality, while higher amounts use lower
    quality and generally produce smaller files.
    """
    # validate_amount(compression_amount)
    image = load_rgb_or_rgba(image_path, preserve_alpha=True)

    image.save(
        output_path,
        format="JXL",
        quality=quality_from_amount(compression_amount, "jpegxl"),
    )


def decompress(image_path: str | Path, output_path: str | Path) -> None:
    """Decode a JPEG XL image and write its reconstructed pixels as PNG."""
    with Image.open(image_path) as compressed:
        reconstructed = ImageOps.exif_transpose(compressed)
        reconstructed.save(output_path, format="PNG")
