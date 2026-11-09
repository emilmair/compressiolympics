#!/usr/bin/env python3
"""Generate scatter plots from benchmark results for a dataset."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "benchmark_plots"
RESULTS_DIR = ROOT / "benchmark_results"


def read_results_csv(path: Path) -> list[dict[str, object]]:
    """Read and validate benchmark results for plotting."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    required_fields = {"algorithm", "compression_ratio", "psnr_db"}
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fields = set(reader.fieldnames or [])
        missing_fields = required_fields - fields
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"CSV is missing required field(s): {missing}")
        return list(reader)


def dataset_csv_paths(
    dataset_name: str,
    algorithms: list[str] | None,
) -> list[Path]:
    """Return result CSVs for a dataset and optional algorithm filters."""
    dataset_dir = RESULTS_DIR / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"benchmark dataset results not found: {dataset_dir}")

    if algorithms:
        paths = [dataset_dir / f"{algorithm}.csv" for algorithm in algorithms]
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"algorithm result CSV not found in {dataset_dir}: "
                + ", ".join(missing)
            )
        return paths

    paths = sorted(
        path
        for path in dataset_dir.glob("*.csv")
        if not path.name.endswith("_failures.csv")
    )
    if not paths:
        raise FileNotFoundError(f"no algorithm result CSVs found in {dataset_dir}")
    return paths


def read_dataset_results(
    dataset_name: str,
    algorithms: list[str] | None,
) -> list[dict[str, object]]:
    """Read and combine algorithm results for a dataset."""
    results: list[dict[str, object]] = []
    for path in dataset_csv_paths(dataset_name, algorithms):
        results.extend(read_results_csv(path))
    return results


def filter_results(
    results: list[dict[str, object]],
    algorithms: list[str] | None,
    images: list[str] | None,
) -> list[dict[str, object]]:
    """Keep only rows matching the requested algorithm and image filters."""
    filtered = results
    if algorithms:
        filtered = [row for row in filtered if row["algorithm"] in algorithms]
    if images:
        filtered = [row for row in filtered if row["image"] in images]
    return filtered


def filename_part(value: str) -> str:
    """Make a filter value safe to include in a filename."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return cleaned or "unnamed"


def output_path(
    dataset_name: str,
    algorithms: list[str] | None,
    images: list[str] | None,
) -> Path:
    """Build a plot path that distinguishes the selected filters."""
    parts = [filename_part(dataset_name)]
    if algorithms:
        parts.append(
            "algorithms-" + "-".join(filename_part(value) for value in algorithms)
        )
    if images:
        parts.append("images-" + "-".join(filename_part(value) for value in images))
    return PLOTS_DIR / ("__".join(parts) + ".png")


def create_scatter_plot(results: list[dict[str, object]], path: Path) -> None:
    """Plot compression ratio vs PSNR, using one color per algorithm."""
    finite_psnr = [
        float(row["psnr_db"])
        for row in results
        if math.isfinite(float(row["psnr_db"]))
    ]
    if finite_psnr:
        low, high = min(finite_psnr), max(finite_psnr)
        top_for_infinity = high + max(3.0, (high - low) * 0.05)
    else:
        top_for_infinity = 100.0

    fig, ax = plt.subplots(figsize=(11, 7))
    algorithm_names = sorted({str(row["algorithm"]) for row in results})

    for algorithm_name in algorithm_names:
        algorithm_rows = [
            row for row in results if row["algorithm"] == algorithm_name
        ]
        x_values = [float(row["compression_ratio"]) for row in algorithm_rows]
        y_values = [
            (
                float(row["psnr_db"])
                if math.isfinite(float(row["psnr_db"]))
                else top_for_infinity
            )
            for row in algorithm_rows
        ]
        ax.scatter(x_values, y_values, s=30, alpha=0.7, label=algorithm_name)

    ax.set_xscale("log")
    ax.set_xlabel("Compression ratio (input size / output size)")
    ax.set_ylabel("PSNR (dB; exact matches shown at the top)")
    ax.set_title("Image compression benchmark")
    ax.grid(True, alpha=0.25)
    if algorithm_names:
        ax.legend(title="Algorithm")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a scatter plot from benchmark results for a dataset."
    )
    parser.add_argument(
        "dataset_name",
        help="dataset name used by the benchmark",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        dest="algorithms",
        metavar="NAME",
        help="plot only this algorithm (repeat for multiple algorithms)",
    )
    parser.add_argument(
        "--image",
        action="append",
        dest="images",
        metavar="PATH",
        help="plot only this image path from the CSV (repeat for multiple images)",
    )
    args = parser.parse_args()

    try:
        results = read_dataset_results(args.dataset_name, args.algorithms)
        results = filter_results(results, args.algorithms, args.images)
        if not results:
            raise ValueError("the selected algorithm/image filters matched no CSV rows")
        path = output_path(args.dataset_name, args.algorithms, args.images)
        create_scatter_plot(results, path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Saved scatter plot to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
