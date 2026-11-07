"""Lossy JPEG 2000 encoder using Pillow/OpenJPEG."""

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
    """Encode as JPEG 2000 (JP2) using irreversible 9/7 wavelets.

    Pillow's ``quality_layers`` in ``rates`` mode represents approximate
    size-reduction factors. The target factor grows with compression_amount.
    """
    require_pillow_feature("jpg_2000", "JPEG 2000")

    amount = validate_amount(compression_amount)
    image = load_rgb_or_rgba(image_path, preserve_alpha=True)
    target_rate = round(200.0 - 200.0 * amount)

    image.save(
        output_path,
        format="JPEG2000",
        quality_mode="rates",
        quality_layers=[target_rate],
        irreversible=True,  # lossy
        # The multi-component transform is useful for three-channel RGB but
        # is not applicable to RGBA, so enable it only for RGB images.
        mct=(image.mode == "RGB"),
    )


def decompress(image_path: str | Path, output_path: str | Path) -> None:
    """Decode the JPEG 2000 bitstream and write a PNG reconstruction."""
    decode_to_png(image_path, output_path)
