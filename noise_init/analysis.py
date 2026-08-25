"""Block-level aggregation, paired bootstrap, Pareto and no-extrapolation Q--D analysis."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep non-dominated points when both diversity and quality are maximized."""
    result = []
    for point in points:
        dominated = any((other["diversity"] >= point["diversity"] and other["quality"] >= point["quality"] and
                         (other["diversity"] > point["diversity"] or other["quality"] > point["quality"])) for other in points)
        if not dominated:
            result.append(point)
    return sorted(result, key=lambda point: point["diversity"])


def interpolate_within(points: list[dict[str, Any]], diversity: float) -> float | None:
    points = sorted(points, key=lambda value: value["diversity"])
    if len(points) < 2 or diversity < points[0]["diversity"] or diversity > points[-1]["diversity"]:
        return None
    for left, right in zip(points, points[1:]):
        if left["diversity"] <= diversity <= right["diversity"]:
            if right["diversity"] == left["diversity"]:
                return max(left["quality"], right["quality"])
            weight = (diversity - left["diversity"]) / (right["diversity"] - left["diversity"])
            return left["quality"] + weight * (right["quality"] - left["quality"])
    return None


def analyze_run(run_dir: str | Path, config: dict[str, Any]) -> None:
    run_dir, analysis_dir = Path(run_dir), Path(run_dir) / "analysis"
    group_rows = _read_csv(run_dir / "metrics" / "per_group.csv")
    if not group_rows:
        raise RuntimeError("Metrics have not been computed")
    quality_metrics = ["clip_cosine", "hpsv3"]
    diversity_metrics = ["dreamsim_mean_pair_distance", "lpips_alex_mean_pair_distance", "vendi_clip"]
    by_metric: dict[str, dict[tuple[str, str], dict[str, dict[str, Any]]]] = {}
    for row in group_rows:
        key = (row["block_id"], row["condition_id"])
        by_metric.setdefault(row["metric"], {}).setdefault(key, {})[row["metric"]] = row
    # Make a conventional block/condition/metric map.
    values: dict[tuple[str, str], dict[str, float]] = {}
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for row in group_rows:
        key = (row["block_id"], row["condition_id"])
        values.setdefault(key, {})[row["metric"]] = float(row["score"])
        metadata.setdefault(key, row)
    aggregates, paired, frontier_rows, improvement_rows, interval_rows = [], [], [], [], []
    conditions = sorted({condition for _, condition in values})
    for quality in quality_metrics:
        for diversity in diversity_metrics:
            complete = {condition: {block for block, candidate in values if candidate == condition and quality in values[(block, candidate)] and diversity in values[(block, candidate)]} for condition in conditions}
            for condition, blocks in complete.items():
                if not blocks: continue
                q = np.array([values[(block, condition)][quality] for block in sorted(blocks)])
                d = np.array([values[(block, condition)][diversity] for block in sorted(blocks)])
                q_interval, d_interval = _bootstrap(q, config["analysis"]), _bootstrap(d, config["analysis"])
                aggregates.append({"condition_id": condition, "quality_metric": quality, "diversity_metric": diversity, "quality": q.mean(), "diversity": d.mean(), "block_count": len(blocks), "quality_se": q_interval["standard_error"], "diversity_se": d_interval["standard_error"], "quality_ci_low": q_interval["ci_low"], "quality_ci_high": q_interval["ci_high"], "diversity_ci_low": d_interval["ci_low"], "diversity_ci_high": d_interval["ci_high"], **_condition_metadata(metadata[(next(iter(blocks)), condition)])})
            curve_rows = [row for row in aggregates if row["quality_metric"] == quality and row["diversity_metric"] == diversity]
            for family in ("baseline", "same_phase", "independent_white"):
                front = pareto_frontier([row for row in curve_rows if row["family"] == family])
                frontier_rows.extend([{**row, "family": family} for row in front])
            baseline = pareto_frontier([row for row in curve_rows if row["family"] == "baseline"])
            for family in ("same_phase", "independent_white"):
                ours = pareto_frontier([row for row in curve_rows if row["family"] == family])
                if len(baseline) < 2 or len(ours) < 2: continue
                lo, hi = max(baseline[0]["diversity"], ours[0]["diversity"]), min(baseline[-1]["diversity"], ours[-1]["diversity"])
                if lo > hi:
                    improvement_rows.append({"quality_metric": quality, "diversity_metric": diversity, "family": family, "available": False, "reason": "no_shared_diversity_range"})
                    continue
                for target in np.linspace(lo, hi, 25):
                    q_base, q_ours = interpolate_within(baseline, float(target)), interpolate_within(ours, float(target))
                    if q_base is not None and q_ours is not None:
                        improvement_rows.append({"quality_metric": quality, "diversity_metric": diversity, "family": family, "available": True, "target_diversity": target, "quality_improvement": q_ours - q_base})
    # Paired effects use only block intersections, never individual images/pairs.
    for condition in conditions:
        family = _condition_metadata(metadata[(next(block for block, candidate in values if candidate == condition), condition)])["family"]
        reference = "baseline/alpha_0p0" if family == "baseline" else _anchor_condition(condition)
        for quality in quality_metrics:
            for diversity in diversity_metrics:
                common = sorted({block for block, candidate in values if candidate == condition and (block, reference) in values and quality in values[(block, candidate)] and diversity in values[(block, candidate)] and quality in values[(block, reference)] and diversity in values[(block, reference)]})
                for block in common:
                    paired.append({"block_id": block, "condition_id": condition, "reference_condition_id": reference, "quality_metric": quality, "diversity_metric": diversity,
                                   "delta_quality": values[(block, condition)][quality] - values[(block, reference)][quality], "delta_diversity": values[(block, condition)][diversity] - values[(block, reference)][diversity]})
                if common:
                    diffs = np.array([values[(b, condition)][quality] - values[(b, reference)][quality] for b in common])
                    interval_rows.append({"condition_id": condition, "reference_condition_id": reference, "quality_metric": quality, "diversity_metric": diversity, **_bootstrap(diffs, config["analysis"])})
    _write_csv(analysis_dir / "aggregate_conditions.csv", aggregates)
    _write_csv(analysis_dir / "paired_effects.csv", paired)
    _write_csv(analysis_dir / "qd_frontiers.csv", frontier_rows)
    _write_csv(analysis_dir / "qd_improvement_at_matched_diversity.csv", improvement_rows)
    _write_csv(analysis_dir / "bootstrap_intervals.csv", interval_rows)
    _plots(run_dir / "plots", aggregates, config)


def _condition_metadata(row: dict[str, Any]) -> dict[str, Any]:
    condition = row["condition_id"]
    family = condition.split("/", 1)[0]
    return {"family": family, "method": row["method"], "alpha": row["alpha"], "gamma": row.get("gamma", "")}


def _anchor_condition(condition: str) -> str:
    # same_phase/alpha_0p5_gamma_0p3 -> baseline/alpha_0p5
    alpha = condition.split("_gamma_", 1)[0].split("/", 1)[1]
    return f"baseline/{alpha}"


def _bootstrap(values: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(int(config["bootstrap_replicates"]))])
    alpha = (1 - float(config["confidence_level"])) / 2
    return {"point_estimate": float(values.mean()), "standard_error": float(means.std(ddof=1)), "ci_low": float(np.quantile(means, alpha)), "ci_high": float(np.quantile(means, 1 - alpha)), "block_count": len(values)}


def _se(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def _plots(plot_dir: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    # Quality/alpha and diversity/alpha (or gamma) views.  ``rows`` contains a
    # quality value once per diversity pairing, so deduplicate by condition.
    for metric, value_key, error_key, directory in [
        ("clip_cosine", "quality", "quality_se", "baseline"), ("hpsv3", "quality", "quality_se", "baseline"),
        ("dreamsim_mean_pair_distance", "diversity", "diversity_se", "baseline"),
        ("lpips_alex_mean_pair_distance", "diversity", "diversity_se", "baseline"), ("vendi_clip", "diversity", "diversity_se", "baseline"),
    ]:
        relevant = [row for row in rows if (row["quality_metric"] == metric if value_key == "quality" else row["diversity_metric"] == metric)]
        unique = {row["condition_id"]: row for row in relevant}.values()
        for family in ("baseline", "same_phase", "independent_white"):
            series = [row for row in unique if row["family"] == family]
            if not series: continue
            parameter = "alpha" if family == "baseline" else "gamma"
            series = sorted(series, key=lambda row: float(row[parameter]))
            fig, axis = plt.subplots(figsize=(6, 4))
            axis.errorbar([float(row[parameter]) for row in series], [float(row[value_key]) for row in series],
                          yerr=[[float(row[f"{value_key}_ci_low"]) - float(row[value_key]) for row in series],
                                [float(row[f"{value_key}_ci_high"]) - float(row[value_key]) for row in series]], marker="o", capsize=3)
            axis.set(xlabel=parameter, ylabel=metric, title=f"{metric} vs {parameter} ({family})")
            axis.grid(alpha=.2)
            target = plot_dir / ("baseline" if family == "baseline" else "methods") / f"{family}__{metric}__{parameter}"
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight"); fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)
    for quality in ("clip_cosine", "hpsv3"):
        for diversity in ("dreamsim_mean_pair_distance", "lpips_alex_mean_pair_distance", "vendi_clip"):
            current = [row for row in rows if row["quality_metric"] == quality and row["diversity_metric"] == diversity]
            if not current: continue
            fig, axis = plt.subplots(figsize=(6, 4))
            for family in sorted({row["family"] for row in current}):
                series = sorted([row for row in current if row["family"] == family], key=lambda row: row["diversity"])
                axis.plot([row["diversity"] for row in series], [row["quality"] for row in series], marker="o", label=family)
                for row in series: axis.annotate(str(row["alpha"] if family == "baseline" else row["gamma"]), (row["diversity"], row["quality"]), fontsize=7)
            axis.set(xlabel=diversity, ylabel=quality, title=f"Q-D ({quality} / {diversity})")
            axis.legend(); axis.grid(alpha=.2)
            target = plot_dir / "qd" / f"{quality}__{diversity}"; target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight"); fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
