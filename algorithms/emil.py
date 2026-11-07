#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ._utils import validate_amount


MAGIC = b"NNCOMP32"
VERSION = 1
HEADER = struct.Struct("<IIQQ")
JXL_EFFORT = 9
RESIDUAL_OFFSET = 32768


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )


def _encode_jxl(source: Path, destination: Path) -> None:
    _run([
        "cjxl",
        "--distance", "0",
        "--effort", str(JXL_EFFORT),
        str(source),
        str(destination),
    ])


def _save_residual(path: Path, residual: np.ndarray) -> None:
    packed = (residual.astype(np.int32) + RESIDUAL_OFFSET).astype(np.uint16)
    if not cv2.imwrite(
        str(path),
        cv2.cvtColor(packed, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    ):
        raise RuntimeError(f"Could not write residual image: {path}")


def compress(image_path: str | Path, output_path: str | Path, amount: float) -> None:
    amount = validate_amount(amount)
    base = 128
    q = amount * 128

    """Write a self-contained base/residual codec file."""
    if base < 1:
        raise ValueError("base must be at least 1")
    if q <= 0:
        raise ValueError("q must be greater than zero")

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width, channels = image.shape
    scale = min(base / max(width, height), 1.0)
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))

    with tempfile.TemporaryDirectory(prefix="thirtytwo-") as directory:
        work = Path(directory)
        small_path = work / "base.png"
        residual_path = work / "residual.png"
        base_jxl = work / "base.jxl"
        residual_jxl = work / "residual.jxl"

        small = Image.fromarray(image).resize(small_size, Image.Resampling.LANCZOS)
        decoded_base = np.asarray(
            small.resize((width, height), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
        residual = image.astype(np.int32) - decoded_base.astype(np.int32)
        quantized = np.rint(residual / q)
        if np.any(quantized < -32768) or np.any(quantized > 32767):
            raise ValueError("q is too small for the int16 residual format")
        quantized = quantized.astype(np.int16)

        small.save(small_path)
        _save_residual(residual_path, quantized)
        _encode_jxl(small_path, base_jxl)
        _encode_jxl(residual_path, residual_jxl)

        base_data = base_jxl.read_bytes()
        residual_data = residual_jxl.read_bytes()
        metadata = {
            "width": width,
            "height": height,
            "channels": channels,
            "base_width": small_size[0],
            "base_height": small_size[1],
            "base": base,
            "q": q,
            "residual_offset": RESIDUAL_OFFSET,
        }
        header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as stream:
            stream.write(MAGIC)
            stream.write(HEADER.pack(VERSION, len(header), len(base_data), len(residual_data)))
            stream.write(header)
            stream.write(base_data)
            stream.write(residual_data)


def _read_container(path: Path) -> tuple[dict, bytes, bytes]:
    with path.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError("not a thirtytwo codec file")
        header_data = stream.read(HEADER.size)
        if len(header_data) != HEADER.size:
            raise ValueError("truncated codec header")
        version, metadata_size, base_size, residual_size = HEADER.unpack(header_data)
        if version != VERSION:
            raise ValueError(f"unsupported codec version: {version}")
        try:
            metadata = json.loads(stream.read(metadata_size))
        except json.JSONDecodeError as error:
            raise ValueError("invalid codec metadata") from error
        base_data = stream.read(base_size)
        residual_data = stream.read(residual_size)
        if len(base_data) != base_size or len(residual_data) != residual_size:
            raise ValueError("truncated codec payload")
        if stream.read(1):
            raise ValueError("unexpected data after codec payload")
    return metadata, base_data, residual_data


def decompress(input_path: str | Path, output_path: str | Path) -> None:
    """Decode a thirtytwo file and write the reconstructed image."""
    metadata, base_data, residual_data = _read_container(Path(input_path))
    width = int(metadata["width"])
    height = int(metadata["height"])
    q = float(metadata["q"])
    offset = int(metadata["residual_offset"])
    if width < 1 or height < 1 or q <= 0 or offset != 32768:
        raise ValueError("invalid codec metadata")

    with tempfile.TemporaryDirectory(prefix="thirtytwo-") as directory:
        work = Path(directory)
        base_jxl = work / "base.jxl"
        residual_jxl = work / "residual.jxl"
        base_png = work / "base.png"
        residual_png = work / "residual.png"
        base_jxl.write_bytes(base_data)
        residual_jxl.write_bytes(residual_data)
        _run(["djxl", str(base_jxl), str(base_png)])
        _run(["djxl", str(residual_jxl), str(residual_png)])

        small = Image.open(base_png).convert("RGB")
        base = np.asarray(
            small.resize((width, height), Image.Resampling.BICUBIC),
            dtype=np.int32,
        )
        packed_bgr = cv2.imread(str(residual_png), cv2.IMREAD_UNCHANGED)
        if packed_bgr is None or packed_bgr.dtype != np.uint16:
            raise ValueError("decoded residual is not a 16-bit PNG")
        if packed_bgr.ndim != 3 or packed_bgr.shape[2] != 3:
            raise ValueError("decoded residual does not have three channels")
        packed = cv2.cvtColor(packed_bgr, cv2.COLOR_BGR2RGB)
        if packed.shape[:2] != (height, width):
            raise ValueError("decoded residual dimensions do not match metadata")
        quantized = packed.astype(np.int32) - offset
        reconstructed = np.clip(
            base + quantized * q,
            0,
            255,
        ).astype(np.uint8)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(reconstructed, mode="RGB").save(output, format="PNG")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="input image path")
    parser.add_argument("output", help="output codec file")
    parser.add_argument("base", type=int, help="maximum base-image edge length")
    parser.add_argument("q", type=float, help="residual quantization step")
    args = parser.parse_args()
    compress(args.image, args.output, args.base, args.q)


if __name__ == "__main__":
    main()
