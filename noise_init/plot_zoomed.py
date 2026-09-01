#!/usr/bin/env python3
"""Render focused, supplementary Q--D figures for the SDXL Turbo finer run.

This script reads the already aggregated analysis table.  It neither runs
generation/metrics nor changes the standard ``run_experiment.py --stage
analyze`` figures.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PROJECT_DIR / "outputs" / "sdxl_turbo_full_finer"
PRIMARY_PAIR = ("hpsv3", "dreamsim_mean_pair_distance")
BASELINE_ALPHAS = (0.2, 0.3, 0.4, 0.5)
DISPLAY_GAMMAS = (0.0125, 0.025, 0.05)
FIXED_ALPHAS = (0.8, 0.9)


def _number(value: float) -> str:
    return f"{value:.8g}"


def _matches(value: str | float, allowed: tuple[float, ...]) -> bool:
    return any(math.isclose(float(value), candidate, abs_tol=1e-9) for candidate in allowed)


def _read_rows(table: Path) -> list[dict[str, Any]]:
    with table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No aggregate rows in {table}")
    return rows


def _pair_rows(rows: list[dict[str, Any]], pair: tuple[str, str]) -> list[dict[str, Any]]:
    quality, diversity = pair
    selected = [row for row in rows if row["quality_metric"] == quality and row["diversity_metric"] == diversity]
    if not selected:
        raise RuntimeError(f"No rows for {quality} × {diversity}")
    return selected


def _pair_target_dir(run_dir: Path, pair: tuple[str, str]) -> Path:
    if pair == PRIMARY_PAIR:
        return run_dir / "analysis" / "primary"
    return run_dir / "analysis" / "robustness" / f"{pair[0]}__{pair[1]}"


def _metric_label(metric: str) -> str:
    labels = {
        "hpsv3": "HPSv3",
        "clip_cosine": "CLIP cosine",
        "dreamsim_mean_pair_distance": "DreamSim distance",
        "lpips_alex_mean_pair_distance": "LPIPS-Alex distance",
        "vendi_clip": "Vendi-CLIP",
    }
    return labels.get(metric, metric)


def _style(axis, quality: str, diversity: str, title: str) -> None:
    axis.set(xlabel=_metric_label(diversity), ylabel=_metric_label(quality), title=title)
    axis.grid(alpha=.22)


def _limits(axis, rows: list[dict[str, Any]], *, padding: float = .16) -> None:
    """Set a roomy viewport containing every requested reference and ours point."""
    for column, setter in (("diversity", axis.set_xlim), ("quality", axis.set_ylim)):
        values = [float(row[column]) for row in rows]
        lower, upper = min(values), max(values)
        margin = max((upper - lower) * padding, 1e-5)
        setter(lower - margin, upper + margin)


def _baseline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if row["family"] == "baseline" and _matches(row["alpha"], BASELINE_ALPHAS)],
        key=lambda row: float(row["alpha"]),
    )


def _ours(rows: list[dict[str, Any]], family: str, *, fixed_alpha: float | None = None) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row["family"] == family and _matches(row["gamma"], DISPLAY_GAMMAS)
        and (fixed_alpha is None or math.isclose(float(row["alpha"]), fixed_alpha, abs_tol=1e-9))
    ]
    return sorted(selected, key=lambda row: (float(row["gamma"]), float(row["alpha"])))


def _plot_baseline_curve(axis, rows: list[dict[str, Any]]) -> None:
    axis.plot(
        [float(row["diversity"]) for row in rows],
        [float(row["quality"]) for row in rows],
        color="black", marker="o", linewidth=2.2, markersize=4, label="baseline (α=0.2–0.5)", zorder=3,
    )
    for row in rows:
        axis.annotate(f"α={_number(float(row['alpha']))}", (float(row["diversity"]), float(row["quality"])), fontsize=7, xytext=(3, 3), textcoords="offset points")


def _plot_baseline_points(axis, rows: list[dict[str, Any]]) -> None:
    """Reference points for a fixed-alpha gamma path: deliberately not a path."""
    axis.scatter(
        [float(row["diversity"]) for row in rows],
        [float(row["quality"]) for row in rows],
        color="black", s=28,
        label=f"baseline reference (α={_number(float(rows[0]['alpha']))}–{_number(float(rows[-1]['alpha']))})",
        zorder=4,
    )
    for row in rows:
        axis.annotate(f"α={_number(float(row['alpha']))}", (float(row["diversity"]), float(row["quality"])), fontsize=7, xytext=(3, 3), textcoords="offset points")


def _gamma_colour(cmap, gamma: float):
    low, high = min(DISPLAY_GAMMAS), max(DISPLAY_GAMMAS)
    return cmap(.18 + .78 * (gamma - low) / (high - low))


def _nearby_baseline(rows: list[dict[str, Any]], proposed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep baseline references local to a fixed-alpha zoom's proposed paths."""
    all_baseline = sorted(
        [row for row in rows if row["family"] == "baseline"],
        key=lambda row: float(row["alpha"]),
    )
    x_values = [float(row["diversity"]) for row in proposed]
    y_values = [float(row["quality"]) for row in proposed]
    x_margin = max((max(x_values) - min(x_values)) * .20, 1e-5)
    y_margin = max((max(y_values) - min(y_values)) * .20, 1e-5)
    local = [
        row for row in all_baseline
        if min(x_values) - x_margin <= float(row["diversity"]) <= max(x_values) + x_margin
        and min(y_values) - y_margin <= float(row["quality"]) <= max(y_values) + y_margin
    ]
    # A very unusual metric may have no local baseline point.  Retain the
    # requested central baseline segment in that case rather than omitting a
    # baseline reference entirely.
    return local or _baseline(rows)


def _draw_gamma_curves(axis, rows: list[dict[str, Any]], family: str, cmap) -> None:
    for gamma in DISPLAY_GAMMAS:
        series = [row for row in rows if math.isclose(float(row["gamma"]), gamma, abs_tol=1e-9)]
        if not series:
            raise RuntimeError(f"Missing {family} gamma={gamma} rows")
        axis.plot(
            [float(row["diversity"]) for row in series],
            [float(row["quality"]) for row in series],
            color=_gamma_colour(cmap, gamma), marker="o", markersize=4, linewidth=1.4,
            label=f"γ={_number(gamma)}", zorder=2,
        )


def _draw_fixed_alpha_curves(axis, rows: list[dict[str, Any]], family: str, cmap) -> None:
    for index, alpha in enumerate(FIXED_ALPHAS):
        series = _ours(rows, family, fixed_alpha=alpha)
        if len(series) != len(DISPLAY_GAMMAS):
            raise RuntimeError(f"Missing {family} fixed-alpha rows for alpha={alpha}")
        color = cmap(.18 + .72 * index / max(len(FIXED_ALPHAS) - 1, 1))
        axis.plot(
            [float(row["diversity"]) for row in series],
            [float(row["quality"]) for row in series],
            color=color, marker="o", markersize=5, linewidth=1.8,
            label=f"α={_number(alpha)} (γ=0.0125–0.05)", zorder=2,
        )
        for row in series:
            axis.annotate(f"γ={_number(float(row['gamma']))}", (float(row["diversity"]), float(row["quality"])), fontsize=7, xytext=(3, 3), textcoords="offset points")


def _legend(axis) -> None:
    axis.legend(ncol=2, fontsize=8, frameon=False, loc="best")


def plot_fixed_gamma_two_panel(rows: list[dict[str, Any]], target: Path, quality: str, diversity: str) -> None:
    import matplotlib.pyplot as plt

    baseline = _baseline(rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)
    for axis, family, title, cmap_name in (
        (axes[0], "same_phase", "Same-phase", "Blues"),
        (axes[1], "independent_white", "Independent-white", "Oranges"),
    ):
        ours = _ours(rows, family)
        _plot_baseline_curve(axis, baseline)
        _draw_gamma_curves(axis, ours, family, plt.colormaps[cmap_name])
        _limits(axis, baseline + ours)
        _style(axis, quality, diversity, title)
        _legend(axis)
    figure.suptitle(
        f"Zoomed Q–D comparison: {_metric_label(quality)} × {_metric_label(diversity)}\n"
        "baseline α=0.2–0.5; proposed γ=0.0125–0.05",
        fontsize=14,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_fixed_alpha_two_panel(rows: list[dict[str, Any]], target: Path, quality: str, diversity: str) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)
    for axis, family, title, cmap_name in (
        (axes[0], "same_phase", "Same-phase: fixed α", "Blues"),
        (axes[1], "independent_white", "Independent-white: fixed α", "Oranges"),
    ):
        paths = [row for alpha in FIXED_ALPHAS for row in _ours(rows, family, fixed_alpha=alpha)]
        baseline = _nearby_baseline(rows, paths)
        _plot_baseline_points(axis, baseline)
        _draw_fixed_alpha_curves(axis, rows, family, plt.colormaps[cmap_name])
        # The limits are deliberately based on both requested alpha paths and
        # nearby baseline references, so alpha=0.8/0.9 paths stay fully
        # visible without a remote baseline point flattening the zoom.
        _limits(axis, baseline + paths, padding=.20)
        _style(axis, quality, diversity, title)
        _legend(axis)
    figure.suptitle(
        f"Zoomed fixed-α γ paths: {_metric_label(quality)} × {_metric_label(diversity)}\n"
        "proposed α=0.8, 0.9; γ=0.0125–0.05; baseline shown as reference points",
        fontsize=14,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Completed run directory (default: SDXL Turbo full finer output).")
    parser.add_argument("--quality-metric", help="Optional quality metric; provide together with --diversity-metric to render one pair only.")
    parser.add_argument("--diversity-metric", help="Optional diversity metric; provide together with --quality-metric to render one pair only.")
    args = parser.parse_args()
    if bool(args.quality_metric) != bool(args.diversity_metric):
        parser.error("--quality-metric and --diversity-metric must be supplied together")

    table = args.run_dir / "analysis" / "tables" / "all_curve_points.csv"
    all_rows = _read_rows(table)
    available_pairs = sorted({(row["quality_metric"], row["diversity_metric"]) for row in all_rows})
    if args.quality_metric:
        pairs = [(args.quality_metric, args.diversity_metric)]
    else:
        # Primary first for readable console output, then all robustness pairs.
        pairs = ([PRIMARY_PAIR] if PRIMARY_PAIR in available_pairs else []) + [pair for pair in available_pairs if pair != PRIMARY_PAIR]
    for pair in pairs:
        rows = _pair_rows(all_rows, pair)
        target_dir = _pair_target_dir(args.run_dir, pair)
        fixed_gamma = target_dir / "methods_qd_two_panel_zoomed.png"
        fixed_alpha = target_dir / "fixed_alpha_gamma_qd_two_panel_zoomed.png"
        plot_fixed_gamma_two_panel(rows, fixed_gamma, *pair)
        plot_fixed_alpha_two_panel(rows, fixed_alpha, *pair)
        print(f"Wrote {fixed_gamma}")
        print(f"Wrote {fixed_alpha}")


if __name__ == "__main__":
    main()
