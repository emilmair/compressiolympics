#!/usr/bin/env python3
"""Benchmark image compression algorithms.

Expected project layout:

    project/
      algorithms/
        algorithm_a.py   # defines compress(image_path, output_path, compression_amount)
                           # and decompress(compressed_path, png_output_path)
        algorithm_b.py
      benchmark.py

Run from anywhere with:

    python benchmark.py dataset 10

The first argument is the dataset directory and the second is the number of
evenly spaced compression amounts. For N=10, amounts are 0.1, 0.2, ..., 1.0.

Results are written to a numbered and timestamped CSV in benchmark_results/,
with failures written to a matching failure CSV.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
ALGORITHMS_DIR = ROOT / "algorithms"
RESULTS_DIR = ROOT / "benchmark_results"

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".jxl"
}

CompressFunction = Callable[[str | Path, str | Path, float], None]
DecompressFunction = Callable[[str | Path, str | Path], None]
Algorithm = tuple[str, CompressFunction, DecompressFunction]


def positive_int(value: str) -> int:
    """argparse type that accepts only positive integers."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def discover_algorithms() -> list[Algorithm]:
    """Import modules exposing both compress() and decompress()."""
    if not ALGORITHMS_DIR.is_dir():
        raise FileNotFoundError(f"Algorithms directory not found: {ALGORITHMS_DIR}")

    # Ensure `import algorithms.some_module` resolves to this project's folder.
    root_string = str(ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    algorithms: list[Algorithm] = []
    module_files = sorted(
        path
        for path in ALGORITHMS_DIR.glob("*.py")
        if path.name != "__init__.py" and not path.name.startswith("_")
    )

    for module_file in module_files:
        if not module_file.stem.isidentifier():
            print(
                f"Skipping {module_file.name}: filename must be a valid Python module name.",
                file=sys.stderr,
            )
            continue

        module_name = f"algorithms.{module_file.stem}"
        try:
            module = importlib.import_module(module_name)
            compress = getattr(module, "compress", None)
            decompress = getattr(module, "decompress", None)
            if not callable(compress):
                raise AttributeError("module does not define a callable compress()")
            if not callable(decompress):
                raise AttributeError("module does not define a callable decompress()")
            algorithms.append((module_file.stem, compress, decompress))
        except Exception as exc:
            print(
                f"Could not load {module_file.name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    return algorithms


def discover_images(dataset_dir: Path) -> list[Path]:
    """Return supported image files under the dataset directory."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    images: list[Path] = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            images.append(path)
        except Exception as exc:
            print(f"Skipping unreadable image {path}: {exc}", file=sys.stderr)
    return images


def image_has_alpha(path: Path) -> bool:
    """Whether an image contains an alpha channel or palette transparency."""
    with Image.open(path) as image:
        return "A" in image.getbands() or "transparency" in image.info


def load_pixels(path: Path, mode: str) -> np.ndarray:
    """Load displayed pixels, accounting for EXIF orientation."""
    with Image.open(path) as image:
        oriented = ImageOps.exif_transpose(image)
        return np.asarray(oriented.convert(mode), dtype=np.uint8)


def calculate_psnr(original_path: Path, reconstruction_path: Path) -> float:
    """Calculate PSNR for the original and decompressed PNG; require equal dimensions."""
    use_alpha = image_has_alpha(original_path) or image_has_alpha(reconstruction_path)
    mode = "RGBA" if use_alpha else "RGB"

    original = load_pixels(original_path, mode)
    reconstructed = load_pixels(reconstruction_path, mode)

    if original.shape != reconstructed.shape:
        raise ValueError(
            "image dimensions differ after decoding: "
            f"original={original.shape}, reconstruction={reconstructed.shape}"
        )

    difference = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = float(np.mean(np.square(difference)))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def run_benchmark(
    subdivisions: int,
    algorithms: list[Algorithm],
    images: list[Path],
    dataset_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run every algorithm against every image at every compression amount."""
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    amounts = [step / subdivisions for step in range(1, subdivisions + 1)]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(algorithms) * len(images) * len(amounts)
    completed = 0

    for algorithm_name, compress, decompress in algorithms:
        for image_path in images:
            input_size = image_path.stat().st_size
            relative_image = image_path.relative_to(dataset_dir)

            for amount in amounts:
                completed += 1
                print(
                    f"[{completed}/{total}] {algorithm_name} | "
                    f"{relative_image} | amount={amount:.6g}"
                )

                try:
                    # A fresh directory per run prevents stale files and name collisions.
                    with tempfile.TemporaryDirectory(
                        prefix="trial_", dir=RESULTS_DIR
                    ) as temporary_directory:
                        # This is the compressed artifact. Keep the source extension
                        # in case the compressor uses it to choose an encoding.
                        compressed_path = (
                            Path(temporary_directory)
                            / f"compressed{image_path.suffix.lower()}"
                        )
                        # The decompressor must write a normal PNG for pixel comparison.
                        decompressed_path = Path(temporary_directory) / "reconstructed.png"

                        compress(image_path, compressed_path, amount)

                        if not compressed_path.is_file():
                            raise FileNotFoundError(
                                "algorithm did not create compressed file: "
                                f"{compressed_path.name}"
                            )

                        compressed_size = compressed_path.stat().st_size
                        if compressed_size <= 0:
                            raise ValueError("algorithm produced an empty compressed file")

                        decompress(compressed_path, decompressed_path)

                        if not decompressed_path.is_file():
                            raise FileNotFoundError(
                                "algorithm did not create decompressed PNG: "
                                f"{decompressed_path.name}"
                            )

                        decompressed_size = decompressed_path.stat().st_size
                        if decompressed_size <= 0:
                            raise ValueError("algorithm produced an empty decompressed PNG")

                        # PSNR compares the original against the decoded reconstruction,
                        # never against the compressed bitstream itself.
                        psnr = calculate_psnr(image_path, decompressed_path)
                        compression_ratio = input_size / compressed_size

                        results.append(
                            {
                                "algorithm": algorithm_name,
                                "image": relative_image.as_posix(),
                                "compression_amount": amount,
                                "input_size_bytes": input_size,
                                # output_size_bytes is the size of the compressed artifact
                                # and is the denominator of compression_ratio.
                                "output_size_bytes": compressed_size,
                                "decompressed_size_bytes": decompressed_size,
                                "compression_ratio": compression_ratio,
                                "psnr_db": psnr,
                            }
                        )
                except Exception as exc:
                    failures.append(
                        {
                            "algorithm": algorithm_name,
                            "image": relative_image.as_posix(),
                            "compression_amount": amount,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(
                        f"  FAILED: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )

    return results, failures


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def next_benchmark_number() -> int:
    """Return the next number for a benchmark result file."""
    numbers = []
    for path in RESULTS_DIR.glob("benchmark_*.csv"):
        match = re.fullmatch(r"benchmark_(\d+)(?:_.*)?\.csv", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def benchmark_output_paths() -> tuple[Path, Path]:
    """Return unique result and failure paths for a benchmark run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    number = next_benchmark_number()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    stem = f"benchmark_{number:03d}_{timestamp}"
    return (
        RESULTS_DIR / f"{stem}.csv",
        RESULTS_DIR / f"{stem}_failures.csv",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark every algorithm in ./algorithms against every image in "
            "the supplied dataset directory at evenly spaced compression amounts "
            "from 1/N to 1.0."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="directory containing the images to benchmark",
    )
    parser.add_argument(
        "subdivisions",
        type=positive_int,
        help="number of quality steps (e.g. 10 runs amounts 0.1 through 1.0)",
    )
    args = parser.parse_args()

    algorithms = discover_algorithms()
    images = discover_images(args.dataset_dir)

    if not algorithms:
        print(
            f"No algorithms with callable compress() and decompress() functions found in {ALGORITHMS_DIR}",
            file=sys.stderr,
        )
        return 1
    if not images:
        print(f"No supported images found in {args.dataset_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(algorithms)} algorithm(s) and {len(images)} image(s).")
    results, failures = run_benchmark(
        args.subdivisions, algorithms, images, args.dataset_dir
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv, failures_csv = benchmark_output_paths()

    result_fields = [
        "algorithm",
        "image",
        "compression_amount",
        "input_size_bytes",
        "output_size_bytes",
        "decompressed_size_bytes",
        "compression_ratio",
        "psnr_db",
    ]
    failure_fields = ["algorithm", "image", "compression_amount", "error"]

    write_csv(results_csv, results, result_fields)
    write_csv(failures_csv, failures, failure_fields)

    if results:
        print(f"\nSaved {len(results)} successful result(s) to: {results_csv}")
    else:
        print("\nNo successful benchmark runs.", file=sys.stderr)

    if failures:
        print(f"{len(failures)} failed run(s) recorded in: {failures_csv}")
    else:
        # Keep the output directory tidy when there are no failures.
        failures_csv.unlink(missing_ok=True)

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
