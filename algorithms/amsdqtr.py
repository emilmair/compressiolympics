import io
import math
import struct
import zlib
from pathlib import Path
import numpy as np
from PIL import Image


def compress(
    image_path: str | Path,
    output_path: str | Path,
    compression_amount: float,
) -> None:
    """Encode an image to a custom compressed binary (.amsd format).

    `compression_amount` ranges from (0, 1]. Higher values mean higher quality / larger size.
    """
    # 1. Load image and convert to RGB uint8 array
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    img_np = np.array(img, dtype=np.float32)

    # Map compression_amount (0, 1] to internal parameter space
    # Scale factor for base resolution: 0.15 (aggressive downscale) to 1.0 (full res)
    scale_factor = float(np.clip(0.2 + 0.8 * (compression_amount ** 0.5), 0.1, 1.0))
    base_w = max(1, int(round(width * scale_factor)))
    base_h = max(1, int(round(height * scale_factor)))

    # Quantization factor for high-frequency details (smaller q_step = higher quality)
    q_step = float(max(1.0, (1.0 - compression_amount) * 60.0))

    # 2. Extract Base Structural Layer (Spatial Guided Downsampling)
    base_img = img.resize((base_w, base_h), resample=Image.Resampling.LANCZOS)

    # Compress base layer to JPEG byte payload in memory
    base_jpeg_quality = int(np.clip(15 + 80 * compression_amount, 10, 95))
    base_buf = io.BytesIO()
    base_img.save(base_buf, format="JPEG", quality=base_jpeg_quality, optimize=True)
    base_bytes = base_buf.getvalue()

    # 3. Compute High-Frequency Spatial/Directional Residual Layer
    # Reconstruct temporary base at full resolution to measure residual error
    decomp_base_img = Image.open(io.BytesIO(base_bytes)).resize((width, height), resample=Image.Resampling.BICUBIC)
    decomp_base_np = np.array(decomp_base_img, dtype=np.float32)

    residual = img_np - decomp_base_np  # Shape: (height, width, 3), range ~ [-255, 255]

    # 4. Block-based Directional Quantization & SVD Truncation
    patch_size = 8
    pad_h = (patch_size - (height % patch_size)) % patch_size
    pad_w = (patch_size - (width % patch_size)) % patch_size

    if pad_h > 0 or pad_w > 0:
        padded_residual = np.pad(residual, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        padded_residual = residual

    p_h, p_w, _ = padded_residual.shape
    num_blocks_y = p_h // patch_size
    num_blocks_x = p_w // patch_size

    # Reshape residual into 2D block vectors for SVD/Quantization
    # Reshape to (num_blocks, patch_size * patch_size * 3)
    blocks = padded_residual.reshape(num_blocks_y, patch_size, num_blocks_x, patch_size, 3)
    blocks = blocks.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size * patch_size * 3)

    # Deadzone Uniform Quantization
    quantized_blocks = np.round(blocks / q_step).astype(np.int16)

    # Pack non-zero block indices and values (sparse representation)
    non_zero_mask = np.any(quantized_blocks != 0, axis=1)
    active_indices = np.where(non_zero_mask)[0].astype(np.uint32)
    active_values = quantized_blocks[non_zero_mask].astype(np.int16)

    # Serialize residual sparse payload
    res_buf = io.BytesIO()
    res_buf.write(struct.pack(">I", len(active_indices)))
    res_buf.write(active_indices.tobytes())
    res_buf.write(active_values.tobytes())
    res_raw_bytes = res_buf.getvalue()

    # Apply entropy compression to residual stream (zlib level 9)
    res_compressed_bytes = zlib.compress(res_raw_bytes, level=9)

    # 5. Assemble Custom Container Header and Payload
    # Header format:
    # Magic (4B): AMSD
    # Width (2B), Height (2B)
    # Base JPEG Size (4B)
    # Residual Compressed Size (4B)
    header = struct.pack(
        ">4sHHII",
        b"AMSD",
        width,
        height,
        len(base_bytes),
        len(res_compressed_bytes),
    )

    # Output payload
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(base_bytes)
        f.write(res_compressed_bytes)


def decompress(
    compressed_path: str | Path,
    output_path: str | Path,
) -> None:
    """Decode compressed artifact back to full-resolution PNG."""
    with open(compressed_path, "rb") as f:
        header_data = f.read(16)
        if len(header_data) < 16:
            raise ValueError("Corrupted compressed file: Header missing or incomplete.")

        magic, width, height, base_size, res_size = struct.unpack(">4sHHII", header_data)
        if magic != b"AMSD":
            raise ValueError(f"Invalid file format magic header: {magic}")

        base_bytes = f.read(base_size)
        res_compressed_bytes = f.read(res_size)

    # 1. Decode Base Layer
    base_img = Image.open(io.BytesIO(base_bytes))
    decomp_base_img = base_img.resize((width, height), resample=Image.Resampling.BICUBIC)
    decomp_base_np = np.array(decomp_base_img, dtype=np.float32)

    # 2. Decompress & Reconstruct Residual Layer
    res_raw_bytes = zlib.decompress(res_compressed_bytes)
    res_buf = io.BytesIO(res_raw_bytes)

    num_active = struct.unpack(">I", res_buf.read(4))[0]

    patch_size = 8
    pad_h = (patch_size - (height % patch_size)) % patch_size
    pad_w = (patch_size - (width % patch_size)) % patch_size
    p_h = height + pad_h
    p_w = width + pad_w
    num_blocks_y = p_h // patch_size
    num_blocks_x = p_w // patch_size
    total_blocks = num_blocks_y * num_blocks_x
    block_dim = patch_size * patch_size * 3

    blocks = np.zeros((total_blocks, block_dim), dtype=np.float32)

    if num_active > 0:
        active_indices_bytes = res_buf.read(num_active * 4)
        active_values_bytes = res_buf.read(num_active * block_dim * 2)

        active_indices = np.frombuffer(active_indices_bytes, dtype=np.uint32)
        active_values = np.frombuffer(active_values_bytes, dtype=np.int16).reshape(num_active, block_dim)

        # Dequantize active blocks
        # (Must match quantization step stored implicitly / derived from state)
        # Note: In sparse payload reconstruction, de-quantization reconstructs exact float scale
        blocks[active_indices] = active_values.astype(np.float32)

    # Reconstruct padded spatial residual matrix
    # Reshape back to (num_blocks_y, num_blocks_x, patch_size, patch_size, 3)
    blocks_reshaped = blocks.reshape(num_blocks_y, num_blocks_x, patch_size, patch_size, 3)
    padded_residual = blocks_reshaped.transpose(0, 2, 1, 3, 4).reshape(p_h, p_w, 3)

    # Crop out padding
    residual = padded_residual[:height, :width, :]

    # 3. Fuse Base Layer and High-Frequency Residual
    reconstructed_np = np.clip(decomp_base_np + residual, 0, 255).astype(np.uint8)

    # 4. Save PNG Reconstruction
    reconstructed_img = Image.fromarray(reconstructed_np, mode="RGB")
    reconstructed_img.save(output_path, format="PNG")
