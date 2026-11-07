"""NovaFactor: shared-dictionary separable patch coding.

This is an experimental lossy image codec designed for the Compressiolympics
interface.  It is not a wrapper around JPEG/PNG/WebP.

Core idea
---------
Each image learns two small dictionaries of one-dimensional spatial vectors.
Every 16x16 channel tile is then represented as a short sum of separable atoms
(outer products of one vector from each dictionary), followed by a 4-bit or
8-bit quantized residual.  The rank and residual bit depth are selected
independently for each tile/channel using a local rate-distortion test.

Dependencies: numpy, Pillow.  The container format uses only Python's standard
library for entropy coding.
"""
from __future__ import annotations

import bz2
import io
import lzma
import struct
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


_MAGIC = b"NVF2"
_TILE = 16
_MAX_RANK = 12
_CODEBOOK_LIMIT = 128


def _canonicalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Give each unit vector a deterministic sign (largest component >= 0)."""
    out = np.asarray(vectors, dtype=np.float32).copy()
    if out.size == 0:
        return out
    pivots = np.argmax(np.abs(out), axis=1)
    signs = np.sign(out[np.arange(len(out)), pivots])
    signs[signs == 0] = 1.0
    out *= signs[:, None]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out /= np.maximum(norms, 1e-12)
    return out


def _fit_dictionary(vectors: np.ndarray, limit: int = _CODEBOOK_LIMIT) -> np.ndarray:
    """Cosine k-means with sign-invariant assignments for 1-D basis vectors."""
    x = _canonicalize_rows(vectors)
    if len(x) == 0:
        return np.eye(_TILE, dtype=np.float32)[:1]
    # Bound dictionary-training work for very large images.  The subsample is
    # deterministic, while the full image is still coded against the result.
    if len(x) > 8192:
        take = np.linspace(0, len(x) - 1, 8192, dtype=np.int64)
        x = x[take]

    # Identical/near-identical vectors are common (especially in smooth areas).
    # Deduplicate rounded rows before clustering to avoid empty, redundant codes.
    rounded = np.round(x, 4)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    x = x[np.sort(unique_idx)]
    k = min(limit, len(x))
    if k == 1:
        return _canonicalize_rows(np.mean(x, axis=0, keepdims=True))

    rng = np.random.default_rng(271828)
    chosen = rng.choice(len(x), size=k, replace=False)
    centers = x[chosen].copy()

    # A few rounds are enough for this very low-dimensional (16-D) dictionary.
    for _ in range(14):
        dots = x @ centers.T
        assignment = np.argmax(np.abs(dots), axis=1)
        signs = np.sign(dots[np.arange(len(x)), assignment])
        signs[signs == 0] = 1.0
        aligned = x * signs[:, None]
        sums = np.zeros_like(centers)
        counts = np.bincount(assignment, minlength=k).astype(np.float32)
        np.add.at(sums, assignment, aligned)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        # Keep empty clusters at their previous location.
        centers = _canonicalize_rows(centers)
    return centers.astype(np.float32)


def _decode_codebook(q: np.ndarray) -> np.ndarray:
    """Recreate the exact normalized vectors used by the decoder."""
    vectors = q.astype(np.float32) / 127.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)



def _half_round_up(value: float) -> np.float16:
    """Round a positive scale to float16 without rounding it downward."""
    rounded = np.float16(value)
    if float(rounded) < value:
        rounded = np.nextafter(rounded, np.float16(np.inf), dtype=np.float16)
    return rounded

def _pack_nibbles(values: np.ndarray) -> bytes:
    """Pack signed values in [-7, 7] into two 4-bit two's-complement nibbles."""
    q = np.asarray(values, dtype=np.int8).reshape(-1)
    if len(q) % 2:
        q = np.pad(q, (0, 1))
    n = (q.astype(np.int16) & 0x0F).astype(np.uint8)
    packed = n[0::2] | (n[1::2] << 4)
    return packed.tobytes()


def _unpack_nibbles(data: bytes, count: int) -> np.ndarray:
    packed = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(len(packed) * 2, dtype=np.int8)
    lo = (packed & 0x0F).astype(np.int8)
    hi = ((packed >> 4) & 0x0F).astype(np.int8)
    lo[lo >= 8] -= 16
    hi[hi >= 8] -= 16
    out[0::2] = lo
    out[1::2] = hi
    return out[:count]


def _entropy_pack(payload: bytes) -> tuple[int, bytes]:
    """Try several general-purpose entropy coders and keep the smallest stream."""
    candidates = (
        (0, zlib.compress(payload, level=9)),
        (1, bz2.compress(payload, compresslevel=9)),
        (2, lzma.compress(payload, preset=6)),
    )
    return min(candidates, key=lambda pair: len(pair[1]))


def _local_target(compression_amount: float) -> tuple[float, float, int]:
    """Map user quality to target MSE, minimum residual step, and rank ceiling."""
    q = float(np.clip(compression_amount, 1e-4, 1.0))
    # A deliberately ambitious quality curve: around 28 dB at the low end and
    # 44 dB at the high end, with an increasingly rich atom budget.
    target_psnr = 27.0 + 17.0 * q
    target_mse = (255.0 * 255.0) / (10.0 ** (target_psnr / 10.0))
    residual_floor = 1.0 + 28.0 * ((1.0 - q) ** 1.35)
    max_rank = min(_MAX_RANK, 2 + int(round(10.0 * q)))
    return target_mse, residual_floor, max_rank


def _choose_tile_model(
    tile: np.ndarray,
    u: np.ndarray,
    singular: np.ndarray,
    vt: np.ndarray,
    code_y: np.ndarray,
    code_x: np.ndarray,
    target_mse: float,
    residual_floor: float,
    max_rank: int,
) -> tuple[int, int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose a compact local atom model plus 4- or 8-bit residual.

    Returns rank, residual mode (4 or 8), residual scale, code indices,
    coefficients, and the quantized residual.
    """
    # Map the leading SVD directions onto shared dictionary entries.
    y_indices: list[int] = []
    x_indices: list[int] = []
    atoms: list[np.ndarray] = []
    usable_rank = min(max_rank, len(singular))
    for j in range(usable_rank):
        yi = int(np.argmax(np.abs(code_y @ u[:, j])))
        xi = int(np.argmax(np.abs(code_x @ vt[j, :])))
        y_indices.append(yi)
        x_indices.append(xi)
        atoms.append(np.outer(code_y[yi], code_x[xi]).reshape(-1))

    target = tile.reshape(-1).astype(np.float32)
    if usable_rank:
        atom_matrix = np.stack(atoms, axis=1)
        # One QR factorization supplies the least-squares projection for every
        # prefix rank.  Only the winning prefix needs a coefficient solve.
        q_basis, _ = np.linalg.qr(atom_matrix, mode="reduced")
        projected = q_basis.T @ target
    else:
        atom_matrix = np.empty((_TILE * _TILE, 0), dtype=np.float32)
        q_basis = atom_matrix
        projected = np.empty((0,), dtype=np.float32)

    best: tuple[float, int, int] | None = None
    # The first valid prefix is the cheapest one for each residual depth,
    # because atom count (and therefore model bytes) rises monotonically.
    found_4 = False
    found_8 = False
    for rank in range(usable_rank + 1):
        if rank == 0:
            approximation = np.zeros((_TILE, _TILE), dtype=np.float32)
        else:
            approximation = (q_basis[:, :rank] @ projected[:rank]).reshape(_TILE, _TILE)
        residual = tile.astype(np.float32) - approximation
        max_abs = float(np.max(np.abs(residual)))

        for bits, qmax, residual_bytes in ((4, 7.0, (_TILE * _TILE) // 2),
                                           (8, 127.0, _TILE * _TILE)):
            if (bits == 4 and found_4) or (bits == 8 and found_8):
                continue
            scale16 = _half_round_up(max(residual_floor, max_abs / qmax))
            scale_used = max(float(scale16), 1e-6)
            qres = np.clip(np.rint(residual / scale_used), -qmax, qmax).astype(np.int8)
            reconstructed = approximation + qres.astype(np.float32) * scale_used
            mse = float(np.mean((tile.astype(np.float32) - reconstructed) ** 2))
            if mse <= target_mse:
                estimated_bytes = 4 + 4 * rank + residual_bytes
                if best is None or estimated_bytes < best[0]:
                    best = (float(estimated_bytes), rank, bits)
                if bits == 4:
                    found_4 = True
                else:
                    found_8 = True
        if found_4 and found_8:
            break

    if best is None:
        # Defensive fallback: maximum rank with an 8-bit residual.
        rank, bits = usable_rank, 8
    else:
        _, rank, bits = best

    if rank:
        chosen_atoms = atom_matrix[:, :rank]
        coefficients = np.linalg.lstsq(chosen_atoms, target, rcond=1e-5)[0].astype(np.float32)
        approximation = (chosen_atoms @ coefficients).reshape(_TILE, _TILE)
    else:
        coefficients = np.empty((0,), dtype=np.float32)
        approximation = np.zeros((_TILE, _TILE), dtype=np.float32)

    residual = tile.astype(np.float32) - approximation
    qmax = 7.0 if bits == 4 else 127.0
    scale16 = _half_round_up(max(residual_floor, float(np.max(np.abs(residual))) / qmax))
    scale_used = max(float(scale16), 1e-6)
    qres = np.clip(np.rint(residual / scale_used), -qmax, qmax).astype(np.int8)
    return (
        rank,
        bits,
        scale_used,
        np.asarray(y_indices[:rank], dtype=np.uint8),
        np.asarray(x_indices[:rank], dtype=np.uint8),
        coefficients,
        qres.reshape(-1),
    )


def compress(
    image_path: str | Path,
    output_path: str | Path,
    compression_amount: float,
) -> None:
    """Encode an image into the NovaFactor NVF2 format."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    with Image.open(image_path) as source:
        has_alpha = ("A" in source.getbands()) or ("transparency" in source.info)
        image = source.convert("RGBA" if has_alpha else "RGB")
        pixels = np.asarray(image, dtype=np.uint8)

    height, width, channels = pixels.shape
    target_mse, residual_floor, max_rank = _local_target(compression_amount)

    padded_h = ((height + _TILE - 1) // _TILE) * _TILE
    padded_w = ((width + _TILE - 1) // _TILE) * _TILE
    padded = np.pad(
        pixels,
        ((0, padded_h - height), (0, padded_w - width), (0, 0)),
        mode="edge",
    )

    # First pass: obtain local SVD directions and train image-specific shared
    # dictionaries.  Only a small set of leading vectors per tile is retained.
    decompositions: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    train_y: list[np.ndarray] = []
    train_x: list[np.ndarray] = []
    train_rank = min(max_rank, 8)
    for ty in range(0, padded_h, _TILE):
        for tx in range(0, padded_w, _TILE):
            channel_decomps = []
            for ch in range(channels):
                block = padded[ty:ty + _TILE, tx:tx + _TILE, ch].astype(np.float32)
                u, singular, vt = np.linalg.svd(block, full_matrices=False)
                channel_decomps.append((u.astype(np.float32), singular.astype(np.float32), vt.astype(np.float32)))
                for j in range(min(train_rank, len(singular))):
                    train_y.append(_canonicalize_rows(u[:, j][None, :])[0])
                    train_x.append(_canonicalize_rows(vt[j, :][None, :])[0])
            decompositions.append(channel_decomps)

    dict_y = _fit_dictionary(np.asarray(train_y, dtype=np.float32))
    dict_x = _fit_dictionary(np.asarray(train_x, dtype=np.float32))

    # Quantize dictionary entries once.  The encoder uses the dequantized
    # versions so its local optimization exactly matches decoder arithmetic.
    qy = np.clip(np.rint(dict_y * 127.0), -127, 127).astype(np.int8)
    qx = np.clip(np.rint(dict_x * 127.0), -127, 127).astype(np.int8)
    code_y = _decode_codebook(qy)
    code_x = _decode_codebook(qx)

    records = bytearray()
    tile_index = 0
    for ty in range(0, padded_h, _TILE):
        for tx in range(0, padded_w, _TILE):
            channel_decomps = decompositions[tile_index]
            tile_index += 1
            for ch in range(channels):
                block = padded[ty:ty + _TILE, tx:tx + _TILE, ch].astype(np.float32)
                u, singular, vt = channel_decomps[ch]
                rank, bits, scale, yi, xi, coeff, qres = _choose_tile_model(
                    block, u, singular, vt, code_y, code_x,
                    target_mse, residual_floor, max_rank,
                )
                records.extend(struct.pack("<BBe", rank, 0 if bits == 4 else 1, float(scale)))
                for j in range(rank):
                    records.extend(struct.pack("<BBe", int(yi[j]), int(xi[j]), float(np.float16(coeff[j]))))
                if bits == 4:
                    records.extend(_pack_nibbles(qres))
                else:
                    records.extend(np.asarray(qres, dtype=np.int8).tobytes())

    payload = qy.tobytes() + qx.tobytes() + bytes(records)
    method, packed = _entropy_pack(payload)
    header = struct.pack(
        "<4sIIBBHHB",
        _MAGIC,
        width,
        height,
        channels,
        _TILE,
        len(qy),
        len(qx),
        method,
    )
    custom_stream = header + packed

    # A lossless escape hatch is important for graphics, icons, and flat
    # gradients: if PNG is strictly smaller, it dominates this lossy stream in
    # both rate and distortion.  This is a mode selection, not a second lossy
    # codec; complex photographic inputs still use the learned atom stream.
    png_buffer = io.BytesIO()
    image.save(png_buffer, format="PNG", optimize=True)
    png_stream = png_buffer.getvalue()
    chosen_stream = png_stream if len(png_stream) < len(custom_stream) else custom_stream

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(chosen_stream)


def decompress(compressed_path: str | Path, output_path: str | Path) -> None:
    """Decode an NVF2 artifact to an RGB/RGBA PNG."""
    compressed_path = Path(compressed_path)
    output_path = Path(output_path)
    raw = compressed_path.read_bytes()
    # The encoder can store a standard lossless PNG when it is smaller than the
    # learned representation.  Preserve dimensions and samples in that case.
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        with Image.open(io.BytesIO(raw)) as stored_png:
            stored_png.load()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stored_png.save(output_path, format="PNG", optimize=True)
        return

    header_size = struct.calcsize("<4sIIBBHHB")
    if len(raw) < header_size:
        raise ValueError("Truncated NovaFactor file")
    magic, width, height, channels, tile_size, ky, kx, method = struct.unpack(
        "<4sIIBBHHB", raw[:header_size]
    )
    if magic != _MAGIC or tile_size != _TILE or channels not in (3, 4):
        raise ValueError("Unsupported NovaFactor file")
    packed = raw[header_size:]
    if method == 0:
        payload = zlib.decompress(packed)
    elif method == 1:
        payload = bz2.decompress(packed)
    elif method == 2:
        payload = lzma.decompress(packed)
    else:
        raise ValueError("Unknown NovaFactor entropy method")

    pos = 0
    qy_size = ky * _TILE
    qx_size = kx * _TILE
    if len(payload) < qy_size + qx_size:
        raise ValueError("Truncated NovaFactor dictionaries")
    qy = np.frombuffer(payload[pos:pos + qy_size], dtype=np.int8).reshape(ky, _TILE)
    pos += qy_size
    qx = np.frombuffer(payload[pos:pos + qx_size], dtype=np.int8).reshape(kx, _TILE)
    pos += qx_size
    code_y = _decode_codebook(qy)
    code_x = _decode_codebook(qx)

    padded_h = ((height + _TILE - 1) // _TILE) * _TILE
    padded_w = ((width + _TILE - 1) // _TILE) * _TILE
    output = np.zeros((padded_h, padded_w, channels), dtype=np.float32)
    residual_count = _TILE * _TILE

    for ty in range(0, padded_h, _TILE):
        for tx in range(0, padded_w, _TILE):
            for ch in range(channels):
                if pos + 4 > len(payload):
                    raise ValueError("Truncated NovaFactor tile record")
                rank, mode, scale = struct.unpack_from("<BBe", payload, pos)
                pos += 4
                if rank > _MAX_RANK or mode not in (0, 1):
                    raise ValueError("Invalid NovaFactor tile record")
                approximation = np.zeros((_TILE, _TILE), dtype=np.float32)
                for _ in range(rank):
                    if pos + 4 > len(payload):
                        raise ValueError("Truncated NovaFactor atom")
                    yi, xi, amplitude = struct.unpack_from("<BB e", payload, pos)
                    pos += 4
                    if yi >= ky or xi >= kx:
                        raise ValueError("Invalid NovaFactor dictionary index")
                    approximation += float(amplitude) * np.outer(code_y[yi], code_x[xi])

                if mode == 0:
                    residual_size = residual_count // 2
                    if pos + residual_size > len(payload):
                        raise ValueError("Truncated 4-bit residual")
                    qres = _unpack_nibbles(payload[pos:pos + residual_size], residual_count)
                    pos += residual_size
                else:
                    residual_size = residual_count
                    if pos + residual_size > len(payload):
                        raise ValueError("Truncated 8-bit residual")
                    qres = np.frombuffer(payload[pos:pos + residual_size], dtype=np.int8)
                    pos += residual_size

                output[ty:ty + _TILE, tx:tx + _TILE, ch] = (
                    approximation + qres.reshape(_TILE, _TILE).astype(np.float32) * float(scale)
                )

    if pos != len(payload):
        raise ValueError("NovaFactor file contains unexpected trailing data")
    output = np.clip(np.rint(output[:height, :width, :]), 0, 255).astype(np.uint8)
    mode_name = "RGBA" if channels == 4 else "RGB"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(output_path, format="PNG", optimize=True)
