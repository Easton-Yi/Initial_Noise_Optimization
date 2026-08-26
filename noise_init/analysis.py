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
    aggregates, paired, frontier_rows, improvement_rows, improvement_summary_rows, interval_rows = [], [], [], [], [], []
    conditions = sorted({condition for _, condition in values})
    for quality in quality_metrics:
        for diversity in diversity_metrics:
            complete = {condition: {block for block, candidate in values if candidate == condition and quality in values[(block, candidate)] and diversity in values[(block, candidate)]} for condition in conditions}
            for condition, blocks in complete.items():
                if not blocks: continue
                q = np.array([values[(block, condition)][quality] for block in sorted(blocks)])
                d = np.array([values[(block, condition)][diversity] for block in sorted(blocks)])
                q_interval, d_interval = _bootstrap(q, config["analysis"]), _bootstrap(d, config["analysis"])
                condition_meta = _condition_metadata(metadata[(next(iter(blocks)), condition)])
                aggregates.append({"condition_id": condition, "quality_metric": quality, "diversity_metric": diversity, "quality": q.mean(), "diversity": d.mean(), "block_count": len(blocks), "quality_se": q_interval["standard_error"], "diversity_se": d_interval["standard_error"], "quality_ci_low": q_interval["ci_low"], "quality_ci_high": q_interval["ci_high"], "diversity_ci_low": d_interval["ci_low"], "diversity_ci_high": d_interval["ci_high"], **condition_meta, "curve_id": _curve_id(condition_meta)})
            curve_rows = [row for row in aggregates if row["quality_metric"] == quality and row["diversity_metric"] == diversity]
            for curve_id in sorted({row["curve_id"] for row in curve_rows}):
                frontier_rows.extend(pareto_frontier([row for row in curve_rows if row["curve_id"] == curve_id]))
            baseline = pareto_frontier([row for row in curve_rows if row["curve_id"] == "baseline"])
            for curve_id in sorted({row["curve_id"] for row in curve_rows if row["curve_id"] != "baseline"}):
                ours = pareto_frontier([row for row in curve_rows if row["curve_id"] == curve_id])
                example = ours[0] if ours else {"family": "unknown", "gamma": ""}
                if len(baseline) < 2 or len(ours) < 2: continue
                lo, hi = max(baseline[0]["diversity"], ours[0]["diversity"]), min(baseline[-1]["diversity"], ours[-1]["diversity"])
                if lo > hi:
                    unavailable = {"quality_metric": quality, "diversity_metric": diversity, "curve_id": curve_id, "family": example["family"], "gamma": example["gamma"], "available": False, "reason": "no_shared_diversity_range"}
                    improvement_rows.append(unavailable)
                    improvement_summary_rows.append(unavailable)
                    continue
                targets = [float(value) for value in np.linspace(lo, hi, 25)]
                uncertainties, summary_uncertainty = _bootstrap_matched_improvements(
                    values,
                    [row["condition_id"] for row in curve_rows if row["curve_id"] == "baseline"],
                    [row["condition_id"] for row in curve_rows if row["curve_id"] == curve_id],
                    quality, diversity, targets, config["analysis"],
                )
                point_improvements = []
                for target, uncertainty in zip(targets, uncertainties):
                    q_base, q_ours = interpolate_within(baseline, float(target)), interpolate_within(ours, float(target))
                    if q_base is not None and q_ours is not None:
                        delta = q_ours - q_base
                        point_improvements.append(delta)
                        improvement_rows.append({"quality_metric": quality, "diversity_metric": diversity, "curve_id": curve_id, "family": example["family"], "gamma": example["gamma"], "available": True, "target_diversity": target, "quality_improvement": delta, **uncertainty})
                improvement_summary_rows.append({"quality_metric": quality, "diversity_metric": diversity, "curve_id": curve_id, "family": example["family"], "gamma": example["gamma"], "available": bool(point_improvements), "matched_diversity_low": lo, "matched_diversity_high": hi, "mean_quality_improvement": float(np.mean(point_improvements)) if point_improvements else "", **summary_uncertainty})
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
    tables_dir = analysis_dir / "tables"
    _write_csv(tables_dir / "all_curve_points.csv", aggregates)
    _write_csv(tables_dir / "paired_effects.csv", paired)
    _write_csv(tables_dir / "qd_frontiers.csv", frontier_rows)
    _write_csv(tables_dir / "matched_diversity.csv", improvement_rows)
    _write_csv(tables_dir / "matched_diversity_summary.csv", improvement_summary_rows)
    _write_csv(tables_dir / "bootstrap_results.csv", interval_rows)
    _plots(analysis_dir, aggregates, improvement_summary_rows, config)


def _condition_metadata(row: dict[str, Any]) -> dict[str, Any]:
    condition = row["condition_id"]
    family = condition.split("/", 1)[0]
    return {"family": family, "method": row["method"], "alpha": row["alpha"], "gamma": row.get("gamma", "")}


def _curve_id(metadata: dict[str, Any]) -> str:
    """Baseline is one alpha sweep; every proposed fixed gamma is another sweep."""
    if metadata["family"] == "baseline":
        return "baseline"
    return f"{metadata['family']}_gamma_{float(metadata['gamma']):.1f}"


def _anchor_condition(condition: str) -> str:
    # same_phase/alpha_0p5_gamma_0p3 -> baseline/alpha_0p5
    alpha = condition.split("_gamma_", 1)[0].split("/", 1)[1]
    return f"baseline/{alpha}"


def _bootstrap(values: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(int(config["bootstrap_replicates"]))])
    alpha = (1 - float(config["confidence_level"])) / 2
    return {"point_estimate": float(values.mean()), "standard_error": float(means.std(ddof=1)), "ci_low": float(np.quantile(means, alpha)), "ci_high": float(np.quantile(means, 1 - alpha)), "block_count": len(values)}


def _bootstrap_matched_improvements(values: dict[tuple[str, str], dict[str, float]], baseline_conditions: list[str], ours_conditions: list[str], quality: str, diversity: str, targets: list[float], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Paired cluster bootstrap that recomputes curves/frontiers inside every draw."""
    baseline_conditions, ours_conditions = sorted(set(baseline_conditions)), sorted(set(ours_conditions))
    required = baseline_conditions + ours_conditions
    blocks = sorted({block for block, _ in values if all((block, condition) in values and quality in values[(block, condition)] and diversity in values[(block, condition)] for condition in required)})
    if not blocks:
        unavailable = {"bootstrap_replicates": 0, "bootstrap_standard_error": "", "bootstrap_ci_low": "", "bootstrap_ci_high": ""}
        return [unavailable.copy() for _ in targets], unavailable
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    differences = [[] for _ in targets]
    mean_differences = []
    for _ in range(int(config["bootstrap_replicates"])):
        sampled = [blocks[index] for index in rng.integers(0, len(blocks), len(blocks))]
        baseline = _bootstrap_curve(values, baseline_conditions, sampled, quality, diversity)
        ours = _bootstrap_curve(values, ours_conditions, sampled, quality, diversity)
        replicate_differences = []
        for index, target in enumerate(targets):
            base_value, ours_value = interpolate_within(baseline, target), interpolate_within(ours, target)
            if base_value is not None and ours_value is not None:
                difference = ours_value - base_value
                differences[index].append(difference)
                replicate_differences.append(difference)
        if replicate_differences:
            mean_differences.append(float(np.mean(replicate_differences)))
    alpha = (1 - float(config["confidence_level"])) / 2
    results = []
    for difference in differences:
        if not difference:
            results.append({"bootstrap_replicates": 0, "bootstrap_standard_error": "", "bootstrap_ci_low": "", "bootstrap_ci_high": ""})
            continue
        samples = np.asarray(difference)
        results.append({"bootstrap_replicates": len(samples), "bootstrap_standard_error": float(samples.std(ddof=1)) if len(samples) > 1 else 0.0,
                        "bootstrap_ci_low": float(np.quantile(samples, alpha)), "bootstrap_ci_high": float(np.quantile(samples, 1 - alpha))})
    if not mean_differences:
        summary = {"bootstrap_replicates": 0, "bootstrap_standard_error": "", "bootstrap_ci_low": "", "bootstrap_ci_high": ""}
    else:
        samples = np.asarray(mean_differences)
        summary = {"bootstrap_replicates": len(samples), "bootstrap_standard_error": float(samples.std(ddof=1)) if len(samples) > 1 else 0.0,
                   "bootstrap_ci_low": float(np.quantile(samples, alpha)), "bootstrap_ci_high": float(np.quantile(samples, 1 - alpha))}
    return results, summary


def _bootstrap_curve(values: dict[tuple[str, str], dict[str, float]], conditions: list[str], sampled_blocks: list[str], quality: str, diversity: str) -> list[dict[str, float]]:
    return pareto_frontier([{"condition_id": condition, "quality": float(np.mean([values[(block, condition)][quality] for block in sampled_blocks])),
                             "diversity": float(np.mean([values[(block, condition)][diversity] for block in sampled_blocks]))}
                            for condition in conditions])


def _se(values: np.ndarray) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def _plots(analysis_dir: Path, rows: list[dict[str, Any]], summaries: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Render the compact, pre-registered figure set; tables retain every curve."""
    pairs = sorted({(row["quality_metric"], row["diversity_metric"]) for row in rows})
    if not pairs:
        return
    requested = config["analysis"].get("primary_metric_pair", {})
    primary = (requested.get("quality", "hpsv3"), requested.get("diversity", "dreamsim_mean_pair_distance"))
    if primary not in pairs:
        raise RuntimeError(f"Configured primary metric pair has no complete results: {primary}")
    primary_rows = _pair_rows(rows, primary)
    _plot_baseline_qd(analysis_dir / "primary" / "baseline_qd.png", primary_rows, primary)
    has_proposed = any(row["curve_id"] != "baseline" for row in rows)
    if not has_proposed:
        return
    _plot_methods_two_panel(analysis_dir / "primary" / "methods_qd_two_panel.png", primary_rows, primary)
    _plot_matched_diversity_gain(analysis_dir / "primary" / "matched_diversity_gain.png", _pair_rows(summaries, primary), primary)
    for pair in pairs:
        if pair == primary:
            continue
        pair_dir = analysis_dir / "robustness" / _pair_token(pair)
        pair_rows = _pair_rows(rows, pair)
        _plot_methods_two_panel(pair_dir / "methods_qd_two_panel.png", pair_rows, pair)
        _plot_matched_diversity_gain(pair_dir / "matched_diversity_gain.png", _pair_rows(summaries, pair), pair)
    for detail in config["analysis"].get("optional_detail_curves", []):
        if not isinstance(detail, dict) or "curve_id" not in detail:
            raise ValueError("analysis.optional_detail_curves entries must be mappings with curve_id")
        pair = (detail.get("quality", primary[0]), detail.get("diversity", primary[1]))
        if pair not in pairs:
            raise ValueError(f"Optional detail refers to unavailable metric pair: {pair}")
        _plot_single_curve_comparison(analysis_dir / "optional_details" / _pair_token(pair) / f"{detail['curve_id']}.png", _pair_rows(rows, pair), pair, detail["curve_id"])


def _pair_rows(rows: list[dict[str, Any]], pair: tuple[str, str]) -> list[dict[str, Any]]:
    return [row for row in rows if row["quality_metric"] == pair[0] and row["diversity_metric"] == pair[1]]


def _pair_token(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__{pair[1]}"


def _metric_label(metric: str) -> str:
    return {"hpsv3": "HPSv3", "clip_cosine": "CLIP cosine", "dreamsim_mean_pair_distance": "DreamSim distance", "lpips_alex_mean_pair_distance": "LPIPS-Alex distance", "vendi_clip": "Vendi-CLIP"}.get(metric, metric)


def _save_figure(figure, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=220, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(figure)


def _series(rows: list[dict[str, Any]], curve_id: str) -> list[dict[str, Any]]:
    return sorted([row for row in rows if row["curve_id"] == curve_id], key=lambda row: float(row["alpha"]))


def _plot_baseline(axis, rows: list[dict[str, Any]]) -> None:
    series = _series(rows, "baseline")
    if not series:
        raise RuntimeError("A Q-D comparison requires the baseline curve")
    axis.plot([row["diversity"] for row in series], [row["quality"] for row in series], color="black", marker="o", linewidth=2.2, label="baseline")


def _style_qd_axis(axis, pair: tuple[str, str], title: str) -> None:
    axis.set(xlabel=_metric_label(pair[1]), ylabel=_metric_label(pair[0]), title=title)
    axis.grid(alpha=.2)


def _plot_baseline_qd(target: Path, rows: list[dict[str, Any]], pair: tuple[str, str]) -> None:
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6, 4.5))
    _plot_baseline(axis, rows)
    for row in _series(rows, "baseline"):
        axis.annotate(f"α={float(row['alpha']):.1f}", (row["diversity"], row["quality"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    _style_qd_axis(axis, pair, "Baseline Q–D curve")
    axis.legend(frameon=False)
    _save_figure(figure, target)


def _plot_family_panel(axis, rows: list[dict[str, Any]], family: str, pair: tuple[str, str]) -> None:
    import matplotlib.pyplot as plt
    _plot_baseline(axis, rows)
    cmap = plt.colormaps["Blues" if family == "same_phase" else "Oranges"]
    curve_ids = sorted({row["curve_id"] for row in rows if row["family"] == family}, key=lambda value: float(value.rsplit("_", 1)[1]))
    for curve_id in curve_ids:
        series = _series(rows, curve_id)
        gamma = float(series[0]["gamma"])
        axis.plot([row["diversity"] for row in series], [row["quality"] for row in series], color=cmap(.18 + .78 * gamma), marker="o", markersize=3, linewidth=1.25, label=f"γ={gamma:.1f}")
    _style_qd_axis(axis, pair, "Same-phase" if family == "same_phase" else "Independent-white")
    axis.legend(ncol=2, fontsize=8, frameon=False)


def _plot_methods_two_panel(target: Path, rows: list[dict[str, Any]], pair: tuple[str, str]) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=False, sharey=False)
    _plot_family_panel(axes[0], rows, "same_phase", pair)
    _plot_family_panel(axes[1], rows, "independent_white", pair)
    figure.suptitle(f"Q–D comparison: {_metric_label(pair[0])} × {_metric_label(pair[1])}")
    _save_figure(figure, target)


def _plot_matched_diversity_gain(target: Path, rows: list[dict[str, Any]], pair: tuple[str, str]) -> None:
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6, 4.5))
    styles = (("same_phase", "Same-phase", "#2166ac"), ("independent_white", "Independent-white", "#b35806"))
    for family, label, color in styles:
        series = sorted([row for row in rows if row.get("family") == family and row.get("available")], key=lambda row: float(row["gamma"]))
        if not series:
            continue
        gamma = [float(row["gamma"]) for row in series]
        gain = [float(row["mean_quality_improvement"]) for row in series]
        lower = [value - float(row["bootstrap_ci_low"]) for value, row in zip(gain, series)]
        upper = [float(row["bootstrap_ci_high"]) - value for value, row in zip(gain, series)]
        axis.errorbar(gamma, gain, yerr=[lower, upper], color=color, marker="o", capsize=3, label=label)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set(xlabel="γ", ylabel="Mean Δ quality at matched diversity", title="Matched-diversity quality gain")
    axis.set_xticks([round(index / 10, 1) for index in range(1, 10)])
    axis.grid(alpha=.2)
    axis.legend(frameon=False)
    _save_figure(figure, target)


def _plot_single_curve_comparison(target: Path, rows: list[dict[str, Any]], pair: tuple[str, str], curve_id: str) -> None:
    import matplotlib.pyplot as plt
    if not _series(rows, curve_id):
        raise ValueError(f"Optional detail curve is unavailable: {curve_id}")
    figure, axis = plt.subplots(figsize=(6, 4.5))
    _plot_baseline(axis, rows)
    series = _series(rows, curve_id)
    axis.plot([row["diversity"] for row in series], [row["quality"] for row in series], marker="o", label=curve_id)
    _style_qd_axis(axis, pair, f"Baseline vs {curve_id}")
    axis.legend(frameon=False)
    _save_figure(figure, target)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
