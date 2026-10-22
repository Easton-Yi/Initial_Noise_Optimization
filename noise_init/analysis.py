"""Prompt-equal Q-D statistics with the original plot styles and frozen run grids."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from io_utils import read_json, write_json, condition_id, file_hash
from run_support import new_output, input_hashes, presentation_config, display_selection, Condition, selected_condition
from result_integrity import enabled_metric_names, validate_analysis_inputs


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


def analyze_run(run_dir: str | Path, config=None, *, block_ids=None, output_dir=None, kind="overall", contract=None, make_plots=True):
    run_dir = Path(run_dir)
    contract = contract or validate_analysis_inputs(run_dir)
    # Only presentation is accepted from callers; all statistics use frozen settings.
    requested = config
    config = presentation_config(contract.config)
    if requested is not None:
        config["analysis"]["two_panel_display"] = requested.get("analysis", {}).get("two_panel_display") or display_selection(contract.config)
        config["analysis"]["primary_metric_pair"] = requested.get("analysis", {}).get("primary_metric_pair", config["analysis"]["primary_metric_pair"])
    selected_blocks = set(contract.blocks) if block_ids is None else set(block_ids)
    if not selected_blocks or not selected_blocks <= set(contract.blocks):
        raise ValueError("Analysis block selection must be a nonempty subset of frozen blocks")
    analysis_dir = Path(output_dir) if output_dir is not None else new_output(run_dir, kind)
    group_rows = _read_csv(run_dir / "metrics" / "per_group.csv")
    group_rows = [row for row in group_rows if row["block_id"] in selected_blocks]
    if not group_rows:
        raise RuntimeError("Metrics have not been computed")
    quality_metrics, pairwise_metrics, group_only_metrics = enabled_metric_names(contract.config)
    diversity_metrics = pairwise_metrics + group_only_metrics
    if not quality_metrics or not diversity_metrics:
        raise RuntimeError("Analysis requires at least one enabled quality and diversity metric")
    frozen_analysis = dict(contract.config["analysis"])
    frozen_analysis["bootstrap_unit"] = "prompt"
    values, metadata = prompt_values(group_rows, contract.blocks)
    prompt_count = len({str(contract.blocks[b]["prompt_id"]) for b in selected_blocks})
    aggregates, paired, frontier_rows, improvement_rows, improvement_summary_rows, interval_rows = [], [], [], [], [], []
    conditions = sorted({condition for _, condition in values})
    for quality in quality_metrics:
        for diversity in diversity_metrics:
            complete = {condition: {block for block, candidate in values if candidate == condition and quality in values[(block, candidate)] and diversity in values[(block, candidate)]} for condition in conditions}
            for condition, blocks in complete.items():
                if not blocks: continue
                q = np.array([values[(block, condition)][quality] for block in sorted(blocks)])
                d = np.array([values[(block, condition)][diversity] for block in sorted(blocks)])
                q_interval, d_interval = _bootstrap(q, frozen_analysis), _bootstrap(d, frozen_analysis)
                condition_meta = _condition_metadata(metadata[(next(iter(blocks)), condition)])
                aggregates.append({"condition_id": condition, "quality_metric": quality, "diversity_metric": diversity, "quality": q.mean(), "diversity": d.mean(), "prompt_count": prompt_count, "block_count": len(selected_blocks), "quality_se": q_interval["standard_error"], "diversity_se": d_interval["standard_error"], "quality_ci_low": q_interval["ci_low"], "quality_ci_high": q_interval["ci_high"], "diversity_ci_low": d_interval["ci_low"], "diversity_ci_high": d_interval["ci_high"], **condition_meta, "curve_id": _curve_id(condition_meta)})
            curve_rows = [row for row in aggregates if row["quality_metric"] == quality and row["diversity_metric"] == diversity]
            for curve_id in sorted({row["curve_id"] for row in curve_rows}):
                frontier_rows.extend(pareto_frontier([row for row in curve_rows if row["curve_id"] == curve_id]))
            baseline = pareto_frontier([row for row in curve_rows if row["curve_id"] == "baseline"])
            for curve_id in sorted({row["curve_id"] for row in curve_rows if row["curve_id"] != "baseline"}):
                ours = pareto_frontier([row for row in curve_rows if row["curve_id"] == curve_id])
                example = ours[0] if ours else {"family": "unknown", "gamma": ""}
                if len(baseline) < 2 or len(ours) < 2:
                    unavailable = {"quality_metric": quality, "diversity_metric": diversity, "curve_id": curve_id,
                                   "family": example["family"], "gamma": example["gamma"], "available": False,
                                   "status": "unavailable", "reason": "baseline_disabled" if not baseline else "insufficient_frontier_points"}
                    improvement_rows.append(unavailable); improvement_summary_rows.append(unavailable)
                    continue
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
                    quality, diversity, targets, frozen_analysis,
                )
                point_improvements = []
                for target, uncertainty in zip(targets, uncertainties):
                    q_base, q_ours = interpolate_within(baseline, float(target)), interpolate_within(ours, float(target))
                    if q_base is not None and q_ours is not None:
                        delta = q_ours - q_base
                        point_improvements.append(delta)
                        improvement_rows.append({"quality_metric": quality, "diversity_metric": diversity, "curve_id": curve_id, "family": example["family"], "gamma": example["gamma"], "available": True, "target_diversity": target, "quality_improvement": delta, **uncertainty})
                improvement_summary_rows.append({"quality_metric": quality, "diversity_metric": diversity, "curve_id": curve_id, "family": example["family"], "gamma": example["gamma"], "available": bool(point_improvements), "matched_diversity_low": lo, "matched_diversity_high": hi, "mean_quality_improvement": float(np.mean(point_improvements)) if point_improvements else "", **summary_uncertainty})
    # Pair condition effects within prompts after averaging their seed groups.
    for condition in conditions:
        family = _condition_metadata(metadata[(next(block for block, candidate in values if candidate == condition), condition)])["family"]
        reference = condition_id("baseline", 0.) if family == "baseline" else _anchor_condition(condition)
        for quality in quality_metrics:
            for diversity in diversity_metrics:
                common = sorted({block for block, candidate in values if candidate == condition and (block, reference) in values and quality in values[(block, candidate)] and diversity in values[(block, candidate)] and quality in values[(block, reference)] and diversity in values[(block, reference)]})
                for block in common:
                    paired.append({"prompt_id": block, "condition_id": condition, "reference_condition_id": reference, "quality_metric": quality, "diversity_metric": diversity,
                                   "delta_quality": values[(block, condition)][quality] - values[(block, reference)][quality], "delta_diversity": values[(block, condition)][diversity] - values[(block, reference)][diversity]})
                if not common:
                    interval_rows.append({"condition_id": condition, "reference_condition_id": reference,
                        "quality_metric": quality, "diversity_metric": diversity, "status": "unavailable",
                        "reason": "reference_condition_not_enabled"})
                if common:
                    diffs = np.array([values[(b, condition)][quality] - values[(b, reference)][quality] for b in common])
                    interval_rows.append({"condition_id": condition, "reference_condition_id": reference, "quality_metric": quality, "diversity_metric": diversity, "status": "passed", "block_count": len(selected_blocks), **_bootstrap(diffs, frozen_analysis)})
    tables_dir = analysis_dir / "tables"
    _write_csv(tables_dir / "all_curve_points.csv", aggregates)
    _write_csv(tables_dir / "paired_effects.csv", paired)
    _write_csv(tables_dir / "qd_frontiers.csv", frontier_rows)
    _write_csv(tables_dir / "matched_diversity.csv", improvement_rows)
    _write_csv(tables_dir / "matched_diversity_summary.csv", improvement_summary_rows)
    _write_csv(tables_dir / "bootstrap_results.csv", interval_rows)
    if make_plots:
        _plots(analysis_dir, aggregates, improvement_summary_rows, config)
    run_manifest = read_json(run_dir / "run_manifest.json")
    write_json(analysis_dir / "analysis_manifest.json", {
        "code_hashes": {name: file_hash(Path(__file__).parent / name) for name in
                        ("analysis.py", "run_support.py", "result_integrity.py")},
        "aggregation_unit": "prompt_equal_after_seed_group_mean", "bootstrap_unit": "prompt",
        "paired_draws": True, "block_count": len(selected_blocks),
        "prompt_count": len({str(contract.blocks[b]["prompt_id"]) for b in selected_blocks}),
        "block_ids": sorted(selected_blocks), "input_hashes": input_hashes(run_dir),
        "aggregate_table_hash": file_hash(tables_dir / "all_curve_points.csv"),
        "source_run_dir": str(run_dir.resolve()),
        "image_path_resolution": "prefer current run for exact recorded model/block/source-condition/sample suffix; retain source records; verify metric image hashes",
        "baseline_comparisons": "available" if any(c.startswith("baseline/") for c in conditions) else "unavailable: baseline disabled",
        "source_run_config_hash": run_manifest.get("config_hash"),
        "frozen_statistical_analysis": contract.config["analysis"],
        "bootstrap_settings": {key: frozen_analysis[key] for key in ("bootstrap_seed", "bootstrap_replicates", "confidence_level", "bootstrap_unit")},
        "presentation_analysis": {
            "primary_metric_pair": config["analysis"]["primary_metric_pair"],
            "optional_detail_curves": config["analysis"].get("optional_detail_curves", []),
            "two_panel_display": config["analysis"].get("two_panel_display"),
        },
    })

    return analysis_dir


def prompt_values(rows, blocks):
    """Equal prompt means, after averaging every seed group of that prompt."""
    samples, metadata = {}, {}
    for row in rows:
        key = (str(blocks[row["block_id"]]["prompt_id"]), row["condition_id"])
        samples.setdefault(key, {}).setdefault(row["metric"], []).append(float(row["score"]))
        metadata.setdefault(key, row)
    values = {key: {metric: float(np.mean(scores)) for metric, scores in metrics.items()}
              for key, metrics in samples.items()}
    return values, metadata


def _condition_metadata(row: dict[str, Any]) -> dict[str, Any]:
    condition = row["condition_id"]
    family = condition.split("/", 1)[0]
    return {"family": family, "method": row["method"], "alpha": row["alpha"], "gamma": row.get("gamma", "")}


def _curve_id(metadata: dict[str, Any]) -> str:
    """Baseline is one alpha sweep; every proposed fixed gamma is another sweep."""
    if metadata["family"] == "baseline":
        return "baseline"
    # ``.1f`` aliases fine-grid values such as 0.0125 and 0.025 to 0.0,
    # merging distinct experimental curves before they reach the plotter.
    return f"{metadata['family']}_gamma_{_format_grid_value(float(metadata['gamma']))}"


def _format_grid_value(value: float) -> str:
    """Stable, concise display/token form without collapsing fine-grid values."""
    return f"{value:.8g}"


def _anchor_condition(condition: str) -> str:
    # same_phase/alpha_0p5_gamma_0p3 -> baseline/alpha_0p5
    alpha = condition.split("_gamma_", 1)[0].split("/", 1)[1]
    return f"baseline/{alpha}"


def _bootstrap(values: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(int(config["bootstrap_replicates"]))])
    alpha = (1 - float(config["confidence_level"])) / 2
    return {"point_estimate": float(values.mean()), "standard_error": float(means.std(ddof=1)) if len(means) > 1 else 0.0, "ci_low": float(np.quantile(means, alpha)), "ci_high": float(np.quantile(means, 1 - alpha)), "prompt_count": len(values)}


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
    """Render the compact registered figures; all gamma-level results remain in tables."""
    pairs = sorted({(row["quality_metric"], row["diversity_metric"]) for row in rows})
    if not pairs:
        return
    requested = config["analysis"].get("primary_metric_pair", {})
    primary = (requested["quality"], requested["diversity"])
    if primary not in pairs:
        raise RuntimeError(f"Configured primary metric pair has no complete results: {primary}")
    display = _two_panel_display(config)
    primary_rows = _pair_rows(rows, primary)
    if any(row["family"] == "baseline" for row in primary_rows):
        _plot_baseline_qd(analysis_dir / "primary" / "baseline_qd.png", _two_panel_rows(primary_rows, display), primary)
    if not any(row["curve_id"] != "baseline" for row in rows):
        return
    _plot_methods_two_panel(analysis_dir / "primary" / "methods_qd_two_panel.png", primary_rows, primary, display)
    _plot_fixed_alpha_gamma_two_panel(analysis_dir / "primary" / "fixed_alpha_gamma_qd_two_panel.png", primary_rows, primary, display)
    _plot_matched_diversity_gain(analysis_dir / "primary" / "matched_diversity_gain.png", _pair_rows(summaries, primary))
    for pair in pairs:
        if pair == primary:
            continue
        target = analysis_dir / "robustness" / _pair_token(pair)
        _plot_methods_two_panel(target / "methods_qd_two_panel.png", _pair_rows(rows, pair), pair, display)
        _plot_fixed_alpha_gamma_two_panel(target / "fixed_alpha_gamma_qd_two_panel.png", _pair_rows(rows, pair), pair, display)
        _plot_matched_diversity_gain(target / "matched_diversity_gain.png", _pair_rows(summaries, pair))
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
        return
    axis.plot([row["diversity"] for row in series], [row["quality"] for row in series], color="black", marker="o", linewidth=2.2, label="baseline")


def _style_qd_axis(axis, pair: tuple[str, str], title: str) -> None:
    axis.set(xlabel=_metric_label(pair[1]), ylabel=_metric_label(pair[0]), title=title)
    axis.grid(alpha=.2)


def _plot_baseline_qd(target: Path, rows: list[dict[str, Any]], pair: tuple[str, str]) -> None:
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6, 4.5))
    _plot_baseline(axis, rows)
    for row in _series(rows, "baseline"):
        axis.annotate(f"α={_format_grid_value(float(row['alpha']))}", (row["diversity"], row["quality"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    _style_qd_axis(axis, pair, "Baseline Q–D curve")
    axis.legend(frameon=False)
    _save_figure(figure, target)


def _plot_family_panel(axis, rows: list[dict[str, Any]], family: str, pair: tuple[str, str], *, legend: bool = True) -> None:
    import matplotlib.pyplot as plt
    _plot_baseline(axis, rows)
    cmap = plt.colormaps["Blues" if family == "same_phase" else "Oranges"]
    curve_ids = sorted({row["curve_id"] for row in rows if row["family"] == family}, key=lambda value: float(value.rsplit("_", 1)[1]))
    curve_gammas = {curve_id: float(_series(rows, curve_id)[0]["gamma"]) for curve_id in curve_ids}
    gamma_values = tuple(curve_gammas[curve_id] for curve_id in curve_ids)
    for curve_id in curve_ids:
        series = _series(rows, curve_id)
        gamma = curve_gammas[curve_id]
        color = _gamma_colour(cmap, gamma, gamma_values)
        axis.plot([row["diversity"] for row in series], [row["quality"] for row in series], color=color, marker="o", markersize=3, linewidth=1.25, label=f"γ={_format_grid_value(gamma)}")
    _style_qd_axis(axis, pair, "Same-phase" if family == "same_phase" else "Independent-white")
    if legend:
        axis.legend(loc="lower left", bbox_to_anchor=(1.02, 0.), fontsize=7.5, frameon=False)
    # The baseline can span much more of the Q-D plane than the proposed
    # sweeps.  Retain that global context, but also show the eight alpha points
    # of every fixed-gamma curve at a readable scale.  Anchored bottom-left,
    # since the curves themselves tend to cluster toward the upper-right.
    proposed = [row for row in rows if row["family"] == family]
    if proposed:
        inset = axis.inset_axes([.04, .08, .43, .34])
        baseline = _series(rows, "baseline")
        inset.plot([row["diversity"] for row in baseline], [row["quality"] for row in baseline], color="black", marker="o", markersize=2.8, linewidth=1.5, zorder=1)
        for curve_id in curve_ids:
            series = _series(rows, curve_id)
            gamma = curve_gammas[curve_id]
            inset.plot([row["diversity"] for row in series], [row["quality"] for row in series], color=_gamma_colour(cmap, gamma, gamma_values), marker="o", markersize=2.5, linewidth=1.0, zorder=2)
        diversity = [float(row["diversity"]) for row in proposed]
        quality = [float(row["quality"]) for row in proposed]
        inset.set_xlim(*_padded_limits(diversity))
        inset.set_ylim(*_padded_limits(quality))
        inset.set_title("Zoom: baseline + proposed sweeps", fontsize=7)
        inset.tick_params(labelsize=6)
        inset.grid(alpha=.2)


def _gamma_colour(cmap, gamma: float, selected_gammas: tuple[float, ...]):
    """Map the displayed gamma range, not the absolute 0--1 interval, to colour."""
    low, high = min(selected_gammas), max(selected_gammas)
    position = .65 if np.isclose(low, high) else (gamma - low) / (high - low)
    return cmap(.18 + .78 * position)


def _padded_limits(values: list[float]) -> tuple[float, float]:
    lower, upper = min(values), max(values)
    padding = max((upper - lower) * .12, 1e-8)
    return lower - padding, upper + padding


def _plot_methods_two_panel(target, rows, pair, display):
    import matplotlib.pyplot as plt
    rows = _two_panel_rows(rows, display)
    families = [f for f in ("same_phase", "independent_white") if any(row["family"] == f for row in rows)]
    if not families:
        if rows:
            _plot_baseline_qd(target, rows, pair)
        return
    figure, axes = plt.subplots(1, len(families), figsize=(13.5 if len(families) == 2 else 6.75, 4.5),
                                sharex=False, sharey=False, squeeze=False, constrained_layout=True)
    for axis, family in zip(axes[0], families):
        _plot_family_panel(axis, rows, family, pair)
    figure.suptitle(f"Q–D comparison ({_display_title_fragment(display)}): {_metric_label(pair[0])} × {_metric_label(pair[1])}")
    _save_figure(figure, target)




def _plot_fixed_alpha_family_panel(axis, rows: list[dict[str, Any]], family: str, pair: tuple[str, str], *, legend: bool = True) -> None:
    """Diagnostic Q-D paths obtained by holding alpha fixed and sweeping gamma."""
    import matplotlib.pyplot as plt
    _plot_baseline_alpha_points(axis, rows, annotate=True)
    alphas = sorted({float(row["alpha"]) for row in rows if row["family"] == family})
    cmap = plt.colormaps["viridis"]
    for index, alpha in enumerate(alphas):
        series = sorted([row for row in rows if row["family"] == family and float(row["alpha"]) == alpha], key=lambda row: float(row["gamma"]))
        color = cmap(.12 + .78 * index / max(len(alphas) - 1, 1))
        axis.plot([row["diversity"] for row in series], [row["quality"] for row in series], color=color, marker="o", markersize=3, linewidth=1.25, label=f"α={_format_grid_value(alpha)}")
    _style_qd_axis(axis, pair, "Same-phase: fixed α, γ sweep" if family == "same_phase" else "Independent-white: fixed α, γ sweep")
    if legend:
        axis.legend(loc="lower left", bbox_to_anchor=(1.02, 0.), fontsize=7.5, frameon=False)
    proposed = [row for row in rows if row["family"] == family]
    if proposed:
        inset = axis.inset_axes([.04, .08, .43, .34])
        baseline = _series(rows, "baseline")
        inset.scatter([row["diversity"] for row in baseline], [row["quality"] for row in baseline], color="black", s=14, zorder=1)
        for index, alpha in enumerate(alphas):
            series = sorted([row for row in proposed if float(row["alpha"]) == alpha], key=lambda row: float(row["gamma"]))
            color = cmap(.12 + .78 * index / max(len(alphas) - 1, 1))
            inset.plot([row["diversity"] for row in series], [row["quality"] for row in series], color=color, marker="o", markersize=2.5, linewidth=1.0, zorder=2)
        inset.set_xlim(*_padded_limits([float(row["diversity"]) for row in proposed]))
        inset.set_ylim(*_padded_limits([float(row["quality"]) for row in proposed]))
        inset.set_title("Zoom: baseline + fixed-α sweeps", fontsize=7)
        inset.tick_params(labelsize=6)
        inset.grid(alpha=.2)


def _plot_baseline_alpha_points(axis, rows: list[dict[str, Any]], *, annotate: bool) -> None:
    """Show baseline as alpha-labelled reference points, not a gamma trajectory."""
    baseline = _series(rows, "baseline")
    if not baseline:
        return
    axis.scatter([row["diversity"] for row in baseline], [row["quality"] for row in baseline], color="black", s=32, label="baseline α points", zorder=3)
    if annotate:
        for row in baseline:
            axis.annotate(f"α={_format_grid_value(float(row['alpha']))}", (row["diversity"], row["quality"]), fontsize=7, xytext=(3, 3), textcoords="offset points")


def _plot_fixed_alpha_gamma_two_panel(target, rows, pair, display):
    import matplotlib.pyplot as plt
    rows = _two_panel_rows(rows, display)
    families = [f for f in ("same_phase", "independent_white") if any(row["family"] == f for row in rows)]
    if not families:
        if rows:
            _plot_baseline_qd(target, rows, pair)
        return
    figure, axes = plt.subplots(1, len(families), figsize=(13.5 if len(families) == 2 else 6.75, 4.5),
                                sharex=False, sharey=False, squeeze=False, constrained_layout=True)
    for axis, family in zip(axes[0], families):
        _plot_fixed_alpha_family_panel(axis, rows, family, pair)
    figure.suptitle(f"Diagnostic fixed-α γ sweeps ({_display_title_fragment(display)}): {_metric_label(pair[0])} × {_metric_label(pair[1])}")
    _save_figure(figure, target)


def _two_panel_display(config):
    return config["analysis"].get("two_panel_display") or display_selection(config)


def _two_panel_rows(rows, display):
    return [row for row in rows if selected_condition(Condition(row["family"], float(row["alpha"]),
            None if row["family"] == "baseline" else float(row["gamma"])), display)]


def _plot_matched_diversity_gain(target: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6, 4.5))
    gammas = _available_gammas(rows)
    positions = {gamma: index for index, gamma in enumerate(gammas)}
    for family, label, color in (("same_phase", "Same-phase", "#2166ac"), ("independent_white", "Independent-white", "#b35806")):
        series = sorted([row for row in rows if row.get("family") == family and row.get("available")], key=lambda row: float(row["gamma"]))
        if not series:
            continue
        series = [row for row in series if row.get("bootstrap_ci_low") not in (None, "")]
        if not series:
            continue
        gamma = [float(row["gamma"]) for row in series]
        gain = [float(row["mean_quality_improvement"]) for row in series]
        lower = [value - float(row["bootstrap_ci_low"]) for value, row in zip(gain, series)]
        upper = [float(row["bootstrap_ci_high"]) - value for value, row in zip(gain, series)]
        # Gamma grids may be deliberately non-uniform (for example the SDXL
        # finer sweep).  Use their ordered grid positions so each tested value
        # remains readable rather than clustering all low gamma values near 0.
        x = [positions[value] for value in gamma]
        if min(lower + upper) >= 0:
            axis.errorbar(x, gain, yerr=[lower, upper], color=color, marker="o", capsize=3, label=label)
        else:
            lows = [float(row["bootstrap_ci_low"]) for row in series]
            highs = [float(row["bootstrap_ci_high"]) for row in series]
            middle = [(lo + hi) / 2 for lo, hi in zip(lows, highs)]
            axis.errorbar(x, middle, yerr=[(hi-lo)/2 for lo, hi in zip(lows, highs)], color=color, fmt="none", capsize=3)
            axis.plot(x, gain, color=color, marker="o", label=label)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set(xlabel="γ", ylabel="Mean Δ quality at matched diversity", title="Matched-diversity quality gain")
    axis.set_xticks(range(len(gammas)))
    axis.set_xticklabels([_format_grid_value(gamma) for gamma in gammas])
    if gammas:
        axis.set_xlim(-.5, len(gammas) - .5)
    axis.grid(alpha=.2)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(frameon=False)
    _save_figure(figure, target)


def _available_gammas(rows: list[dict[str, Any]]) -> tuple[float, ...]:
    """Actual gamma values represented by the plotted summary rows."""
    return tuple(sorted({float(row["gamma"]) for row in rows if row.get("family") in {"same_phase", "independent_white"} and row.get("available")}))


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


def _display_title_fragment(display):
    def endpoint(values):
        return _format_grid_value(min(values)) if len(values) == 1 else f"{_format_grid_value(min(values))}–{_format_grid_value(max(values))}"
    grids = [grid for family, grid in display.items() if family != "baseline"]
    alphas = sorted({value for grid in grids for value in grid["alpha_values"]})
    gammas = sorted({value for grid in grids for value in grid["gamma_values"]})
    return f"α={endpoint(alphas)}, γ={endpoint(gammas)}" if grids else "baseline"


def validate_aggregate_rows(rows, contract, source):
    """Reject incomplete/ambiguous tables before plotting existing aggregates."""
    import math
    from result_integrity import validate_condition_metadata
    quality, pairwise, group_only = enabled_metric_names(contract.config)
    expected = {(c, q, d) for c in contract.condition_ids for q in quality for d in pairwise + group_only}
    seen = set()
    for row in rows:
        validate_condition_metadata(row, context=str(source))
        key = (row["condition_id"], row["quality_metric"], row["diversity_metric"])
        if key in seen or key not in expected or not all(math.isfinite(float(row[k])) for k in ("quality", "diversity")):
            raise RuntimeError(f"{source}: duplicate/unexpected/non-finite curve point {key}")
        seen.add(key)
    if seen != expected:
        raise RuntimeError(f"{source}: missing {len(expected - seen)} configured curve points; rerun analysis")
