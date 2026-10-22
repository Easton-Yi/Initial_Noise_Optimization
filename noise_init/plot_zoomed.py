#!/usr/bin/env python3
"""Render supplementary Q--D figures from existing aggregate metrics.
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
from io_utils import read_json, write_json, file_hash
from result_integrity import validate_analysis_inputs
from run_support import resolve_run, presentation_config, add_display_arguments, input_hashes
from analysis import _two_panel_rows, validate_aggregate_rows


def _grid_values(rows, family, axis):
    return tuple(sorted({float(row[axis]) for row in rows if row["family"] == family}))


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

def _pair_target_dir(run_dir, pair, primary_pair):
    if pair == primary_pair:
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
    axis.tick_params(labelleft=True)
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
        [row for row in rows if row["family"] == "baseline"],
        key=lambda row: float(row["alpha"]),
    )

def _ours(rows: list[dict[str, Any]], family: str, *, fixed_alpha: float | None = None) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row["family"] == family
        and (fixed_alpha is None or math.isclose(float(row["alpha"]), fixed_alpha, abs_tol=1e-9))
    ]
    return sorted(selected, key=lambda row: (float(row["gamma"]), float(row["alpha"])))

def _plot_baseline_points(axis, rows: list[dict[str, Any]]) -> None:
    """Reference points for a fixed-alpha gamma path: deliberately not a path."""
    if not rows:
        return
    axis.scatter(
        [float(row["diversity"]) for row in rows],
        [float(row["quality"]) for row in rows],
        color="black", s=28,
        label=f"baseline reference (α={_number(float(rows[0]['alpha']))}–{_number(float(rows[-1]['alpha']))})",
        zorder=4,
    )
    for row in rows:
        axis.annotate(f"α={_number(float(row['alpha']))}", (float(row["diversity"]), float(row["quality"])), fontsize=7, xytext=(3, 3), textcoords="offset points")

def _gamma_colour(cmap, gamma, gammas):
    low, high = min(gammas), max(gammas)
    return cmap(.18 + .78 * ((gamma - low) / (high - low) if high != low else .5))

def _display_description(rows):
    parts = []
    baseline = _grid_values(rows, "baseline", "alpha")
    if baseline:
        parts.append(f"baseline α={_number(min(baseline))}–{_number(max(baseline))}")
    for family, name in (("same_phase", "Same-phase"), ("independent_white", "Independent-white")):
        gammas = _grid_values(rows, family, "gamma")
        if gammas:
            parts.append(f"{name} γ={_number(min(gammas))}–{_number(max(gammas))}")
    return "; ".join(parts)

def _draw_gamma_curves(axis, rows: list[dict[str, Any]], family: str, cmap) -> None:
    for gamma in _grid_values(rows, family, "gamma"):
        series = sorted([row for row in rows if math.isclose(float(row["gamma"]), gamma, abs_tol=1e-9)], key=lambda row: float(row["alpha"]))
        if not series:
            raise RuntimeError(f"Missing {family} gamma={gamma} rows")
        axis.plot(
            [float(row["diversity"]) for row in series],
            [float(row["quality"]) for row in series],
            color=_gamma_colour(cmap, gamma, _grid_values(rows, family, "gamma")), marker="o", markersize=4, linewidth=1.4,
            label=f"γ={_number(gamma)}", zorder=2,
        )

def _draw_fixed_alpha_curves(axis, rows: list[dict[str, Any]], family: str, cmap) -> None:
    alphas = _grid_values(rows, family, "alpha")
    gammas = _grid_values(rows, family, "gamma")
    for index, alpha in enumerate(alphas):
        series = _ours(rows, family, fixed_alpha=alpha)
        if len(series) != len(gammas):
            raise RuntimeError(f"Missing {family} fixed-alpha rows for alpha={alpha}")
        color = cmap(.18 + .72 * index / max(len(alphas) - 1, 1))
        axis.plot(
            [float(row["diversity"]) for row in series],
            [float(row["quality"]) for row in series],
            color=color, marker="o", markersize=5, linewidth=1.8,
            label=f"α={_number(alpha)} (γ={_number(min(gammas))}–{_number(max(gammas))})", zorder=2,
        )
        for row in series:
            axis.annotate(f"γ={_number(float(row['gamma']))}", (float(row["diversity"]), float(row["quality"])), fontsize=7, xytext=(3, 3), textcoords="offset points")

def _legend(axis) -> None:
    axis.legend(ncol=2, fontsize=8, frameon=False, loc="best")

def plot_fixed_gamma_two_panel(rows: list[dict[str, Any]], target: Path, quality: str, diversity: str) -> None:
    import matplotlib.pyplot as plt
    baseline = _baseline(rows)
    display_rows = baseline + _ours(rows, "same_phase") + _ours(rows, "independent_white")
    families = [f for f in ("same_phase", "independent_white") if _ours(rows, f)] or ["baseline"]
    figure, axes = plt.subplots(1, len(families), figsize=(14 if len(families) == 2 else 7, 6.2),
                                sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    for axis, family in zip(axes[0], families):
        title = {"same_phase": "Same-phase", "independent_white": "Independent-white", "baseline": "Baseline"}[family]
        cmap_name = "Blues" if family == "same_phase" else "Oranges"
        ours = _ours(rows, family) if family != "baseline" else []
        _plot_baseline_points(axis, baseline)
        if family != "baseline":
            _draw_gamma_curves(axis, ours, family, plt.colormaps[cmap_name])
        _limits(axis, display_rows)
        _style(axis, quality, diversity, title)
        _legend(axis)
    figure.suptitle(
        f"Zoomed Q–D comparison: {_metric_label(quality)} × {_metric_label(diversity)}\n"
        f"{_display_description(rows)}",
        fontsize=14,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=240, bbox_inches="tight")
    plt.close(figure)

def plot_fixed_alpha_two_panel(rows: list[dict[str, Any]], target: Path, quality: str, diversity: str) -> None:
    import matplotlib.pyplot as plt
    baseline = _baseline(rows)
    display_rows = baseline + [row for row in rows if row["family"] != "baseline"]
    alphas = sorted({float(row["alpha"]) for row in display_rows if row["family"] != "baseline"})
    families = [f for f in ("same_phase", "independent_white") if _ours(rows, f)] or ["baseline"]
    figure, axes = plt.subplots(1, len(families), figsize=(14 if len(families) == 2 else 7, 6.2),
                                sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    for axis, family in zip(axes[0], families):
        title = {"same_phase": "Same-phase", "independent_white": "Independent-white", "baseline": "Baseline"}[family] + ": fixed α"
        cmap_name = "Blues" if family == "same_phase" else "Oranges"
        _plot_baseline_points(axis, baseline)
        if family != "baseline":
            _draw_fixed_alpha_curves(axis, rows, family, plt.colormaps[cmap_name])
        _limits(axis, display_rows, padding=.20)
        _style(axis, quality, diversity, title)
        _legend(axis)
    figure.suptitle(
        f"Zoomed fixed-α γ paths: {_metric_label(quality)} × {_metric_label(diversity)}\n"
        f"proposed α={', '.join(_number(a) for a in alphas)}; {_display_description(rows)}",
        fontsize=14,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=240, bbox_inches="tight")
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-id")
    add_display_arguments(parser)
    args = parser.parse_args()
    try:
        run = resolve_run(args.run_id, args.run_dir)
        contract = validate_analysis_inputs(run)
        options = vars(args).copy()
        config = presentation_config(contract.config, options)
        table = run / "analysis" / "tables" / "all_curve_points.csv"
        if not table.is_file():
            raise RuntimeError(f"Missing {table}; run run_experiment.py --stage analyze first")
        metadata_path = run / "analysis" / "analysis_manifest.json"
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        expected_settings = {key: contract.config["analysis"][key] for key in ("bootstrap_seed", "bootstrap_replicates", "confidence_level")}
        expected_settings["bootstrap_unit"] = "prompt"
        if (metadata.get("aggregation_unit") != "prompt_equal_after_seed_group_mean"
                or metadata.get("bootstrap_settings") != expected_settings
                or metadata.get("input_hashes") != input_hashes(run)
                or metadata.get("aggregate_table_hash") != file_hash(table)):
            raise RuntimeError("Aggregate table lacks current prompt-level analysis metadata or has changed inputs; run run_experiment.py --stage analyze first")
        all_rows = _read_rows(table)
        validate_aggregate_rows(all_rows, contract, table)
        selected = _two_panel_rows(all_rows, config["analysis"]["two_panel_display"])
        primary_config = contract.config["analysis"]["primary_metric_pair"]
        primary = (primary_config["quality"], primary_config["diversity"])
        requested = config["analysis"]["primary_metric_pair"]
        pair = (requested["quality"], requested["diversity"])
        available = sorted({(r["quality_metric"], r["diversity_metric"]) for r in all_rows})
        pairs = [pair] if args.quality_metric or args.diversity_metric else [primary] + [p for p in available if p != primary]
        for pair in pairs:
            rows = _pair_rows(selected, pair)
            target = _pair_target_dir(run, pair, primary)
            fixed_gamma = target / "methods_qd_two_panel_zoomed.png"
            fixed_alpha = target / "fixed_alpha_gamma_qd_two_panel_zoomed.png"
            plot_fixed_gamma_two_panel(rows, fixed_gamma, *pair)
            plot_fixed_alpha_two_panel(rows, fixed_alpha, *pair)
            print(f"Wrote {fixed_gamma}")
            print(f"Wrote {fixed_alpha}")
        write_json(run / "analysis" / "zoom_manifest.json", {
            "source_table": str(table), "source_table_hash": file_hash(table),
            "run_manifest_hash": file_hash(run / "run_manifest.json"),
            "display_selection": config["analysis"]["two_panel_display"], "metric_pairs": pairs,
            "plot_style": "original user plot_zoomed.py", "statistics": "read existing aggregate table; unchanged"})
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
if __name__ == "__main__":
    main()
