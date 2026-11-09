# Compressiolympics

Benchmark image-compression algorithms across configurable quality levels,
recording compressed size, compression ratio, and reconstruction PSNR.

## Usage

```
python benchmark.py DATASET_NAME ALGORITHM QUALITY_STEPS
```

The dataset must exist at `datasets/DATASET_NAME/`. Run one algorithm against
that dataset. Results are written to
`benchmark_results/DATASET_NAME/ALGORITHM.csv`.

Generate a plot for all algorithms benchmarked on a dataset:

```
python plot.py DATASET_NAME
```

Optionally filter the plot by algorithm and/or image:

```
python plot.py DATASET_NAME --algorithm ALGORITHM --image IMAGE_NAME
```

## Installation

This is for miniconda. For other environments, figure it out.

Create and activate an environment with the Python modules used by the benchmark and algorithms:

```bash
conda create -n compressiolympics -c conda-forge \
  python=3.12 numpy pillow opencv libjxl
conda activate compressiolympics
python -m pip install pillow-jxl-plugin
```

The `libjxl` package provides the `cjxl` and `djxl` commands required by
`algorithms/emil.py`. The `pillow-jxl-plugin` package enables JPEG XL support
for `algorithms/jpegxl.py`. Pillow's AVIF, JPEG 2000, and WebP support must
also be available in the installed Pillow build; verify the available codecs
with:

```bash
python - <<'PY'
from PIL import features

for codec in ("avif", "jpg_2000", "webp"):
    print(f"{codec}: {features.check(codec)}")
PY
```

## Algorithm module specification

An algorithm must be a Python module in the `algorithms/` directory. Its
filename must be a valid Python identifier, must not start with `_`, and must
define these two callable functions:

```python
from pathlib import Path


def compress(
    image_path: str | Path,
    output_path: str | Path,
    compression_amount: float,
) -> None:
    """Encode image_path to output_path."""


def decompress(
    compressed_path: str | Path,
    output_path: str | Path,
) -> None:
    """Decode compressed_path to a PNG at output_path."""
```

### `compress`

- `image_path` is a readable source image.
- `output_path` is the path where the compressed artifact must be written.
- `compression_amount` is a float in the range `(0, 1]`. The benchmark passes
  evenly spaced values from `1 / QUALITY_STEPS` through `1.0`.
- The algorithm owns the meaning of `compression_amount` and may map it to
  whatever quality, rate, effort, quantization, or other internal setting it
  needs.
- The function must create a non-empty compressed file at `output_path`, or
  raise an exception.

### `decompress`

- `compressed_path` is the artifact produced by `compress`.
- `output_path` is the path where the reconstruction must be written.
- The function must create a non-empty PNG at `output_path`.
- The reconstruction must have the same displayed pixel dimensions as the
  source image. The benchmark compares RGB or RGBA pixels and calculates PSNR.

Modules that cannot be imported, do not define both functions, or use an
invalid filename are skipped. A failure during an individual compression or
decompression run is recorded as a benchmark failure for that algorithm and
input.
