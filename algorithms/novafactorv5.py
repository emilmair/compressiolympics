"""NovaFactor v5: searched chromatic lattices and adaptive spatial meshes.

The encoder searches two unrelated image-specific representations: NFG1, a
downsampled palette-indexed color lattice, and NVF4, an adaptive mesh of
quadrilateral and triangular patches. It measures every candidate and selects
the smallest stream meeting its target PSNR; PNG is the exact escape mode.
Increasing compression_amount means more compression and lower target PSNR.

Only numpy and Pillow are required.  Lossy coding is implemented here rather
than delegated to JPEG, WebP, AVIF, or another image codec.
"""
from __future__ import annotations

import bz2
import io
import lzma
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

_MAGIC = b"NVF4"
_HEADER = struct.Struct("<4sIIBHBB")  # mesh: magic, width, height, channels, palette, min leaf, entropy
_GRID_MAGIC = b"NFG1"
_GRID_HEADER = struct.Struct("<4sIIHHBHBB")  # grid: magic, full W/H, low W/H, channels, palette, index bits, entropy


def _index_bits(palette_count: int) -> int:
    return max(1, int(np.ceil(np.log2(max(2, palette_count)))))


def _pack_palette_indices(indices: np.ndarray, bits: int) -> bytes:
    values = np.asarray(indices, dtype=np.uint16).reshape(-1)
    if bits == 4 and len(values) == 4:
        return bytes((int(values[0] | (values[1] << 4)), int(values[2] | (values[3] << 4))))
    out = bytearray((len(values) * bits + 7) // 8)
    bitpos = 0
    mask = (1 << bits) - 1
    for value in values:
        v = int(value) & mask
        j, shift = bitpos >> 3, bitpos & 7
        out[j] |= (v << shift) & 255
        if shift + bits > 8:
            out[j + 1] |= v >> (8 - shift)
        bitpos += bits
    return bytes(out)


def _unpack_palette_indices(data: bytes, bits: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    if bits == 4 and len(raw) == 2:
        return np.array([raw[0] & 15, raw[0] >> 4, raw[1] & 15, raw[1] >> 4], dtype=np.uint16)
    out = np.empty(4, dtype=np.uint16)
    bitpos = 0
    mask = (1 << bits) - 1
    for i in range(4):
        j, shift = bitpos >> 3, bitpos & 7
        value = int(raw[j]) >> shift
        if shift + bits > 8:
            value |= int(raw[j + 1]) << (8 - shift)
        out[i] = value & mask
        bitpos += bits
    return out


def _entropy_pack(data: bytes) -> tuple[int, bytes]:
    variants = (
        (0, zlib.compress(data, 9)),
        (1, bz2.compress(data, 9)),
        (2, lzma.compress(data, preset=6)),
    )
    return min(variants, key=lambda x: len(x[1]))


def _palette_size(compression_amount: float) -> int:
    a = float(np.clip(compression_amount, 1e-4, 1.0))
    # Reserve more chromatic codewords for gentle compression.  The spatial
    # tree, not palette size alone, carries most of the rate-distortion change.
    return int(np.clip(round(32 + 992 * ((1.0 - a) ** 1.35)), 16, 1024))


_AMOUNT_KNOTS = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
# Empirical Kodak PSNR curve from the strongest conventional baseline in the
# supplied CSV (AVIF), shifted down by 2 dB to explore a more rate-aggressive
# part of the Pareto frontier. The codec still measures the resulting PSNR.
_TARGET_PSNR_KNOTS = np.asarray(
    [39.66, 37.38, 35.60, 33.52, 31.45, 29.30, 27.61, 26.12, 24.79, 23.61],
    dtype=np.float32,
)


def _target_psnr(compression_amount: float) -> float:
    a = float(np.clip(compression_amount, 0.1, 1.0))
    return float(np.interp(a, _AMOUNT_KNOTS, _TARGET_PSNR_KNOTS))


def _target_mse(compression_amount: float) -> float:
    target = _target_psnr(compression_amount)
    return (255.0 * 255.0) / (10.0 ** (target / 10.0))


def _minimum_leaf(compression_amount: float) -> int:
    a = float(np.clip(compression_amount, 1e-4, 1.0))
    return 2 + int(round(4.0 * (a ** 1.5)))


def _fit_palette(pixels: np.ndarray, count: int) -> np.ndarray:
    """Image-conditioned weighted k-means palette, trained deterministically."""
    h, w, channels = pixels.shape
    flat = pixels.reshape(-1, channels)
    # A deterministic spatially distributed sample bounds training memory.
    stride = max(1, len(flat) // 12000)
    sample = flat[::stride].astype(np.float32)
    colors, counts = np.unique(sample.astype(np.uint8), axis=0, return_counts=True)
    x = colors.astype(np.float32)
    weights = counts.astype(np.float32)
    k = min(count, len(x))
    if k <= 1:
        return np.rint(np.mean(x, axis=0, keepdims=True)).astype(np.uint8)

    # Farthest-first seeds span the image's chromatic volume better than
    # frequency-only palette selection on natural photographs.
    mean = np.average(x, axis=0, weights=weights)
    centers = np.empty((k, channels), dtype=np.float32)
    centers[0] = mean
    min_dist = np.sum((x - mean) ** 2, axis=1)
    for j in range(1, k):
        idx = int(np.argmax(min_dist * np.sqrt(weights)))
        centers[j] = x[idx]
        dist = np.sum((x - centers[j]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)

    for _ in range(10):
        x2 = np.sum(x * x, axis=1, keepdims=True)
        c2 = np.sum(centers * centers, axis=1)[None, :]
        distances = np.maximum(x2 + c2 - 2.0 * (x @ centers.T), 0.0)
        labels = np.argmin(distances, axis=1)
        sums = np.zeros_like(centers)
        mass = np.bincount(labels, weights=weights, minlength=k).astype(np.float32)
        for ch in range(channels):
            sums[:, ch] = np.bincount(labels, weights=weights * x[:, ch], minlength=k)
        nonempty = mass > 0
        centers[nonempty] = sums[nonempty] / mass[nonempty, None]
    palette = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
    palette = np.unique(palette, axis=0)
    return palette


def _nearest_palette(colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Assign colors to the nearest codeword in bounded-size chunks."""
    c = np.asarray(colors, dtype=np.float32).reshape(-1, palette.shape[1])
    p = palette.astype(np.float32)
    p2 = np.sum(p * p, axis=1)[None, :]
    out = np.empty(len(c), dtype=np.uint16)
    for start in range(0, len(c), 8192):
        chunk = c[start:start + 8192]
        c2 = np.sum(chunk * chunk, axis=1, keepdims=True)
        d = np.maximum(c2 + p2 - 2.0 * (chunk @ p.T), 0.0)
        out[start:start + len(chunk)] = np.argmin(d, axis=1).astype(np.uint16)
    return out


def _pack_index_array(indices: np.ndarray, bits: int) -> bytes:
    values = np.asarray(indices, dtype=np.uint16).reshape(-1)
    shifts = np.arange(bits, dtype=np.uint16)
    bit_matrix = ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1), bitorder="little").tobytes()


def _unpack_index_array(data: bytes, count: int, bits: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    unpacked = np.unpackbits(raw, bitorder="little")[:count * bits]
    matrix = unpacked.reshape(count, bits).astype(np.uint16)
    weights = (1 << np.arange(bits, dtype=np.uint16))[None, :]
    return np.sum(matrix * weights, axis=1, dtype=np.uint16)


def _corners_from_palette(indices: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[np.asarray(indices, dtype=np.uint8)].astype(np.float32)


def _predictive_palette_stream(indices: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, bytes]:
    """Sort palette colors along their dominant axis, then code row deltas."""
    centered = palette.astype(np.float32) - np.mean(palette, axis=0, keepdims=True)
    covariance = centered.T @ centered
    _, vectors = np.linalg.eigh(covariance)
    order = np.argsort(centered @ vectors[:, -1], kind="stable")
    inverse = np.empty(len(order), dtype=np.uint16)
    inverse[order] = np.arange(len(order), dtype=np.uint16)
    ordered_palette = palette[order]
    ordered_indices = inverse[np.asarray(indices, dtype=np.uint16)]
    grid = ordered_indices.reshape(indices.shape[0], indices.shape[1]).astype(np.int16)
    deltas = np.empty_like(grid, dtype="<i2")
    deltas[:, 0] = grid[:, 0]
    deltas[:, 1:] = grid[:, 1:] - grid[:, :-1]
    return ordered_palette, deltas.tobytes()


def _render_patch(width: int, height: int, corners: np.ndarray, mode: int) -> np.ndarray:
    """Render a palette-coded quadrilateral using one of three surfaces."""
    tl, tr, bl, br = corners.astype(np.float32)
    if width > 1:
        u = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    else:
        u = np.zeros((1, 1, 1), dtype=np.float32)
    if height > 1:
        v = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    else:
        v = np.zeros((1, 1, 1), dtype=np.float32)

    if mode == 0 or width == 1 or height == 1:
        top = (1.0 - u) * tl + u * tr
        bottom = (1.0 - u) * bl + u * br
        return (1.0 - v) * top + v * bottom

    uu = np.broadcast_to(u, (height, width, 1))
    vv = np.broadcast_to(v, (height, width, 1))
    if mode == 1:  # diagonal TL -> BR
        first = uu >= vv
        a = (1.0 - uu) * tl + (uu - vv) * tr + vv * br
        b = (1.0 - vv) * tl + (vv - uu) * bl + uu * br
        return np.where(first, a, b)
    # mode 2: diagonal TR -> BL
    first = (uu + vv) <= 1.0
    a = (1.0 - uu - vv) * tl + uu * tr + vv * bl
    b = (1.0 - vv) * tr + (1.0 - uu) * bl + (uu + vv - 1.0) * br
    return np.where(first, a, b)


def _children(x0: int, y0: int, x1: int, y1: int, minimum_leaf: int) -> list[tuple[int, int, int, int]]:
    """Split only axes larger than the quality-dependent terminal cell size."""
    w, h = x1 - x0, y1 - y0
    if w <= minimum_leaf and h <= minimum_leaf:
        return []
    if w > minimum_leaf:
        xm = x0 + w // 2
        xranges = [(x0, xm), (xm, x1)]
    else:
        xranges = [(x0, x1)]
    if h > minimum_leaf:
        ym = y0 + h // 2
        yranges = [(y0, ym), (ym, y1)]
    else:
        yranges = [(y0, y1)]
    return [(xa, ya, xb, yb) for ya, yb in yranges for xa, xb in xranges]


def _leaf_candidate(
    image: np.ndarray,
    palette: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Return best palette-surface MSE, mode, indices, and palette corners."""
    src = image[y0:y1, x0:x1].astype(np.float32)
    corners_src = np.stack((image[y0, x0], image[y0, x1 - 1],
                            image[y1 - 1, x0], image[y1 - 1, x1 - 1]))
    indices = _nearest_palette(corners_src, palette)
    corners = _corners_from_palette(indices, palette)
    width, height = x1 - x0, y1 - y0
    modes = (0,) if width <= 1 or height <= 1 else (0, 1, 2)
    best_mse, best_mode = float("inf"), 0
    for mode in modes:
        recon = _render_patch(width, height, corners, mode)
        mse = float(np.mean((src - recon) ** 2))
        if mse < best_mse:
            best_mse, best_mode = mse, mode
    return best_mse, best_mode, indices, corners


def _decompress_entropy(method: int, packed: bytes) -> bytes:
    if method == 0:
        return zlib.decompress(packed)
    if method == 1:
        return bz2.decompress(packed)
    if method == 2:
        return lzma.decompress(packed)
    raise ValueError("Unknown NovaFactor entropy method")


def _encode_grid_candidate(image: np.ndarray, amount: float, target_psnr: float) -> tuple[bytes, float]:
    """Search a small scale-space and palette family for the best rate point."""
    height, width, channels = image.shape
    source_image = Image.fromarray(image, mode="RGBA" if channels == 4 else "RGB")
    a = float(np.clip(amount, 1e-4, 1.0))
    max_factor = 1 + int(round(15.0 * a))
    factors = [f for f in (1, 2, 3, 4, 6, 8, 12, 16) if f <= max_factor]
    if a <= 0.2:
        palette_sizes = (768, 512, 256, 128, 64)
    elif a <= 0.7:
        palette_sizes = (512, 256, 128, 64, 32, 16)
    else:
        palette_sizes = (128, 64, 32, 16, 8)

    best_feasible: tuple[int, float, bytes] | None = None
    best_quality: tuple[float, int, bytes] | None = None
    sample_source = image.astype(np.float32)
    target_mse = (255.0 * 255.0) / (10.0 ** (target_psnr / 10.0))

    for factor in factors:
        low_w = max(1, int(round(width / factor)))
        low_h = max(1, int(round(height / factor)))
        low_image = source_image.resize((low_w, low_h), Image.Resampling.LANCZOS)
        low = np.asarray(low_image, dtype=np.uint8)
        # The highest palette is tried first; if it cannot meet the distortion
        # target at this scale, no smaller palette at this scale can do better.
        factor_max_psnr = -1.0
        for requested_k in palette_sizes:
            palette = _fit_palette(low, requested_k)
            indices = _nearest_palette(low.reshape(-1, channels), palette)
            bits = max(1, int(np.ceil(np.log2(max(2, len(palette))))))
            index_stream = _pack_index_array(indices, bits)
            payload = palette.tobytes() + index_stream
            method, packed = _entropy_pack(payload)
            candidate = _GRID_HEADER.pack(
                _GRID_MAGIC, width, height, low_w, low_h, channels, len(palette), bits, method
            ) + packed

            # Try a second coding mode: sort palette colors along their
            # dominant axis and encode horizontal index differences. The
            # compressed-size comparison decides whether it is worthwhile.
            ordered_palette, delta_stream = _predictive_palette_stream(
                indices.reshape(low_h, low_w), palette
            )
            predicted_payload = ordered_palette.tobytes() + delta_stream
            predicted_method, predicted_packed = _entropy_pack(predicted_payload)
            predicted_candidate = _GRID_HEADER.pack(
                _GRID_MAGIC, width, height, low_w, low_h, channels,
                len(ordered_palette), bits, predicted_method + 3
            ) + predicted_packed
            if len(predicted_candidate) < len(candidate):
                candidate = predicted_candidate

            low_reconstruction = palette[indices.astype(np.int64)].reshape(low_h, low_w, channels)
            reconstructed = np.asarray(
                Image.fromarray(low_reconstruction.astype(np.uint8),
                                mode="RGBA" if channels == 4 else "RGB").resize(
                                    (width, height), Image.Resampling.LANCZOS
                                ),
                dtype=np.float32,
            )
            mse = float(np.mean((sample_source - reconstructed) ** 2))
            psnr = float("inf") if mse <= 1e-12 else 10.0 * np.log10((255.0 * 255.0) / mse)
            size = len(candidate)
            factor_max_psnr = max(factor_max_psnr, psnr)
            if best_quality is None or psnr > best_quality[0] or (psnr == best_quality[0] and size < best_quality[1]):
                best_quality = (psnr, size, candidate)
            if psnr >= target_psnr:
                if best_feasible is None or size < best_feasible[0]:
                    best_feasible = (size, psnr, candidate)
        # If even the largest palette fails at this factor, coarser scales
        # cannot meet the same target on ordinary low-pass reconstructions.
        if factor_max_psnr < target_psnr:
            break

    if best_feasible is not None:
        return best_feasible[2], best_feasible[1]
    assert best_quality is not None
    return best_quality[2], best_quality[0]


def _encode_mesh_candidate(image: np.ndarray, amount: float, target_psnr: float) -> tuple[bytes, float]:
    """Encode adaptive chromatic mesh candidate; retained for edge-rich images."""
    height, width, channels = image.shape
    palette = _fit_palette(image, _palette_size(amount))
    target_mse = (255.0 * 255.0) / (10.0 ** (target_psnr / 10.0))
    minimum_leaf = _minimum_leaf(amount)
    tree = bytearray()
    total_sse = 0.0

    def encode_node(x0: int, y0: int, x1: int, y1: int) -> None:
        nonlocal total_sse
        mse, mode, indices, _ = _leaf_candidate(image, palette, x0, y0, x1, y1)
        kids = _children(x0, y0, x1, y1, minimum_leaf)
        if mse <= target_mse or not kids:
            tree.append(0)
            tree.append(mode)
            tree.extend(_pack_palette_indices(indices, _index_bits(len(palette))))
            total_sse += mse * (x1 - x0) * (y1 - y0) * channels
            return
        tree.append(1)
        for child in kids:
            encode_node(*child)

    encode_node(0, 0, width, height)
    payload = palette.tobytes() + bytes(tree)
    method, packed = _entropy_pack(payload)
    candidate = _HEADER.pack(_MAGIC, width, height, channels, len(palette), minimum_leaf, method) + packed
    mse = total_sse / (width * height * channels)
    psnr = float("inf") if mse <= 1e-12 else 10.0 * np.log10((255.0 * 255.0) / mse)
    return candidate, psnr


def compress(image_path: str | Path, output_path: str | Path, compression_amount: float) -> None:
    """Encode the best rate-distortion candidate; larger amounts mean more compression."""
    output_path = Path(output_path)
    with Image.open(image_path) as source:
        has_alpha = ("A" in source.getbands()) or ("transparency" in source.info)
        image_obj = source.convert("RGBA" if has_alpha else "RGB")
        image = np.asarray(image_obj, dtype=np.uint8)

    amount = float(np.clip(compression_amount, 1e-4, 1.0))
    target_psnr = _target_psnr(amount)
    candidates: list[tuple[int, float, bytes, str]] = []

    grid_stream, grid_psnr = _encode_grid_candidate(image, amount, target_psnr)
    candidates.append((len(grid_stream), grid_psnr, grid_stream, "grid"))

    # The adaptive mesh is a second, spatially local representation.  It is
    # especially useful when edges and smooth regions are strongly localized.
    if amount >= 0.65:
        mesh_stream, mesh_psnr = _encode_mesh_candidate(image, amount, target_psnr)
        candidates.append((len(mesh_stream), mesh_psnr, mesh_stream, "mesh"))

    png_buffer = io.BytesIO()
    image_obj.save(png_buffer, format="PNG", optimize=True)
    png_stream = png_buffer.getvalue()
    candidates.append((len(png_stream), float("inf"), png_stream, "png"))

    # Treat the target as a rate-distortion operating point, not an absolute
    # quality promise. If no lossy representation can quite reach it, permit a
    # 1.25 dB tolerance before falling back to the highest-quality candidate.
    lossy = [candidate for candidate in candidates if candidate[3] != "png"]
    feasible = [candidate for candidate in lossy if candidate[1] >= target_psnr]
    near = [candidate for candidate in lossy if candidate[1] >= target_psnr - 1.25]
    if feasible:
        chosen = min(feasible, key=lambda item: item[0])
    elif near:
        chosen = min(near, key=lambda item: item[0])
    else:
        chosen = max(lossy, key=lambda item: (item[1], -item[0]))
    # PNG is retained only when it is genuinely smaller than the chosen model.
    png_candidate = next(candidate for candidate in candidates if candidate[3] == "png")
    if png_candidate[0] < chosen[0]:
        chosen = png_candidate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(chosen[2])


def decompress(compressed_path: str | Path, output_path: str | Path) -> None:
    """Decode NFG1, NVF4, or its lossless PNG escape mode into a same-size PNG."""
    compressed_path, output_path = Path(compressed_path), Path(output_path)
    raw = compressed_path.read_bytes()
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        with Image.open(io.BytesIO(raw)) as stored:
            stored.load()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stored.save(output_path, format="PNG", optimize=True)
        return
    if raw.startswith(_GRID_MAGIC):
        if len(raw) < _GRID_HEADER.size:
            raise ValueError("Truncated NovaFactor grid header")
        magic, width, height, low_w, low_h, channels, palette_count, bits, method = _GRID_HEADER.unpack(
            raw[:_GRID_HEADER.size]
        )
        if (magic != _GRID_MAGIC or channels not in (3, 4) or low_w < 1 or low_h < 1
                or not (1 <= palette_count <= 1024) or not (1 <= bits <= 10)):
            raise ValueError("Unsupported NovaFactor grid stream")
        index_mode, entropy_method = divmod(method, 3)
        if index_mode not in (0, 1):
            raise ValueError("Unknown NovaFactor grid index mode")
        payload = _decompress_entropy(entropy_method, raw[_GRID_HEADER.size:])
        palette_bytes = palette_count * channels
        count = low_w * low_h
        palette = np.frombuffer(payload[:palette_bytes], dtype=np.uint8)
        if len(palette) != palette_bytes:
            raise ValueError("Truncated NovaFactor grid palette")
        palette = palette.reshape(palette_count, channels)
        if index_mode == 0:
            index_bytes = (count * bits + 7) // 8
            if len(payload) != palette_bytes + index_bytes:
                raise ValueError("Invalid NovaFactor grid payload size")
            indices = _unpack_index_array(payload[palette_bytes:], count, bits)
        else:
            delta_bytes = count * 2
            if len(payload) != palette_bytes + delta_bytes:
                raise ValueError("Invalid predictive NovaFactor grid payload size")
            deltas = np.frombuffer(payload[palette_bytes:], dtype="<i2").reshape(low_h, low_w)
            indices = np.cumsum(deltas.astype(np.int32), axis=1).reshape(-1).astype(np.uint16)
        if np.any(indices >= palette_count):
            raise ValueError("NovaFactor grid palette index out of range")
        low = palette[indices.astype(np.int64)].reshape(low_h, low_w, channels)
        mode_name = "RGBA" if channels == 4 else "RGB"
        restored = Image.fromarray(low.astype(np.uint8), mode=mode_name).resize(
            (width, height), Image.Resampling.LANCZOS
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        restored.save(output_path, format="PNG", optimize=True)
        return
    if len(raw) < _HEADER.size:
        raise ValueError("Truncated NovaFactor file")
    magic, width, height, channels, palette_count, minimum_leaf, method = _HEADER.unpack(raw[:_HEADER.size])
    if magic != _MAGIC or channels not in (3, 4) or not (1 <= palette_count <= 1024) or minimum_leaf < 2:
        raise ValueError("Unsupported NovaFactor stream")
    packed = raw[_HEADER.size:]
    if method == 0:
        payload = zlib.decompress(packed)
    elif method == 1:
        payload = bz2.decompress(packed)
    elif method == 2:
        payload = lzma.decompress(packed)
    else:
        raise ValueError("Unknown NovaFactor entropy method")

    palette_bytes = palette_count * channels
    if len(payload) < palette_bytes:
        raise ValueError("Truncated NovaFactor palette")
    palette = np.frombuffer(payload[:palette_bytes], dtype=np.uint8).reshape(palette_count, channels)
    pos = palette_bytes
    output = np.zeros((height, width, channels), dtype=np.uint8)

    def decode_node(x0: int, y0: int, x1: int, y1: int) -> None:
        nonlocal pos
        if pos >= len(payload):
            raise ValueError("Truncated NovaFactor tree")
        flag = payload[pos]
        pos += 1
        if flag == 1:
            kids = _children(x0, y0, x1, y1, minimum_leaf)
            if not kids:
                raise ValueError("Invalid subdivision at minimum cell")
            for child in kids:
                decode_node(*child)
            return
        if flag != 0 or pos >= len(payload):
            raise ValueError("Invalid NovaFactor tree node")
        meta = payload[pos]
        pos += 1
        if meta not in (0, 1, 2):
            raise ValueError("Invalid palette leaf")
        mode = meta
        index_bytes = (4 * _index_bits(palette_count) + 7) // 8
        if pos + index_bytes > len(payload):
            raise ValueError("Truncated palette indices")
        indices = _unpack_palette_indices(payload[pos:pos + index_bytes], _index_bits(palette_count))
        pos += index_bytes
        if np.any(indices >= palette_count):
            raise ValueError("Palette index out of range")
        corners = palette[indices].astype(np.float32)
        patch = _render_patch(x1 - x0, y1 - y0, corners, mode)
        output[y0:y1, x0:x1] = np.clip(np.rint(patch), 0, 255).astype(np.uint8)

    decode_node(0, 0, width, height)
    if pos != len(payload):
        raise ValueError("NovaFactor stream has trailing data")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGBA" if channels == 4 else "RGB").save(output_path, format="PNG", optimize=True)
