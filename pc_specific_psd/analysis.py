"""Cross-block aggregation and paired-effect analysis over a run's computed
metrics (plan section on ``analysis.py``).

Deliberately does **not** reuse ``noise_init.analysis.analyze_run`` -- that
module's every row is keyed by ``family``/``alpha``/``gamma`` (parsed out of
a ``baseline/alpha_0p5``-style condition id) and its central machinery
(``pareto_frontier``, ``interpolate_within``, ``_bootstrap_matched_improvements``)
exists specifically to interpolate a *continuous* alpha/gamma Q-D curve at a
shared "matched diversity" target. Our PC-specific study has no such curve:
each run is a small, fixed set of directly-generated conditions (Reference
plus one or two candidate PC groups' +/- pair), and per plan
(PC-Specific-PSD-Implementation-Prompt.md:277) "初始五个点只提供局部证据,
不跨不同PC组硬连'同一条强度曲线', 不外推matched-Q/D, 不宣称完整Pareto优胜"
-- the five (or three) points are local evidence only, never linked into one
intensity curve across different PC groups, never extrapolated to a matched
Q/D target, never claimed as a full Pareto result. So this module has no
curve-fitting/interpolation function anywhere: every comparison is a direct
lookup among conditions actually generated for the run, and asking for one
that wasn't raises ``AnalysisError`` rather than approximating it.

The block-bootstrap recipe itself *is* reused byte-for-byte from
``noise_init.analysis._bootstrap`` (2000 replicates, seed 20260829, 95% CI --
frozen constants here, not user-configurable, per
PC-Specific-PSD-Implementation-Prompt.md:132 / PCA-Specific-PSD-Editing-Plan.md:620).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pc_specific_psd.review import ARTIFACT_FIELDS, GalleryReviewMappingEntry, GalleryReviewRow

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260829
CONFIDENCE_LEVEL = 0.95

REFERENCE_CONDITION_ID = "reference"
QUALITY_METRICS = ("clip_cosine", "hpsv3")
DIVERSITY_METRICS = ("dreamsim_mean_pair_distance", "lpips_alex_mean_pair_distance", "vendi_clip")


class AnalysisError(RuntimeError):
    """Raised for missing-metrics input and for an unmeasured-comparison request."""


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_per_group_scores(run_dir: str | Path) -> list[dict[str, Any]]:
    rows = _read_csv(Path(run_dir) / "metrics" / "per_group.csv")
    if not rows:
        raise AnalysisError(f"Metrics have not been computed for {run_dir} (metrics/per_group.csv is missing or empty)")
    return rows


def _block_values(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    """(block_id, condition_id) -> {metric: score}, ``status == "ok"`` rows
    only -- an ``incomplete``/``not_applicable`` gallery contributes no score,
    it is simply absent from this map rather than coerced to a placeholder.
    """
    values: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["block_id"], row["condition_id"])
        values.setdefault(key, {})[row["metric"]] = float(row["score"])
    return values


def _bootstrap(values: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(BOOTSTRAP_REPLICATES)])
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    return {
        "point_estimate": float(values.mean()),
        "standard_error": float(means.std(ddof=1)),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1 - alpha)),
        "block_count": len(values),
    }


def summarize_conditions(run_dir: str | Path) -> list[dict[str, Any]]:
    """Per-(condition, metric) block bootstrap over directly measured scores
    only -- no family/alpha/gamma parsing (this schema has none), no curve id.
    """
    values = _block_values(load_per_group_scores(run_dir))
    conditions = sorted({condition for _, condition in values})
    rows = []
    for condition in conditions:
        for metric in QUALITY_METRICS + DIVERSITY_METRICS:
            blocks = sorted(block for block, cand in values if cand == condition and metric in values[(block, cand)])
            if not blocks:
                continue
            scores = np.array([values[(block, condition)][metric] for block in blocks])
            rows.append({"condition_id": condition, "metric": metric, **_bootstrap(scores)})
    return rows


def paired_effects(run_dir: str | Path, *, reference_condition_id: str = REFERENCE_CONDITION_ID) -> list[dict[str, Any]]:
    """delta = condition score - reference score, computed per matched block
    first (paired), then bootstrapped by resampling those matched blocks --
    matching noise_init.analysis's own per-condition delta recipe.
    """
    values = _block_values(load_per_group_scores(run_dir))
    conditions = sorted({condition for _, condition in values if condition != reference_condition_id})
    rows = []
    for condition in conditions:
        for metric in QUALITY_METRICS + DIVERSITY_METRICS:
            common = sorted(
                block for block, cand in values
                if cand == condition
                and (block, reference_condition_id) in values
                and metric in values[(block, cand)]
                and metric in values[(block, reference_condition_id)]
            )
            if not common:
                continue
            diffs = np.array([values[(b, condition)][metric] - values[(b, reference_condition_id)][metric] for b in common])
            rows.append({
                "condition_id": condition, "reference_condition_id": reference_condition_id,
                "metric": metric, **_bootstrap(diffs),
            })
    return rows


def compare_conditions(run_dir: str | Path, condition_id: str, metric: str, *, reference_condition_id: str = REFERENCE_CONDITION_ID) -> dict[str, Any]:
    """A single directly-measured paired comparison. Raises ``AnalysisError``
    -- never approximates -- when the requested (condition, metric) pair was
    not actually generated/scored for this run: there is no curve to
    interpolate on.
    """
    for row in paired_effects(run_dir, reference_condition_id=reference_condition_id):
        if row["condition_id"] == condition_id and row["metric"] == metric:
            return row
    raise AnalysisError(
        f"No measured comparison for condition={condition_id!r} metric={metric!r} against "
        f"reference={reference_condition_id!r} in {run_dir} -- this module never "
        "interpolates or extrapolates an unmeasured operating point"
    )


def summarize_gallery_review(
    gallery_review_rows: Sequence[GalleryReviewRow] | None,
    mapping: Sequence[GalleryReviewMappingEntry] | None,
) -> dict[str, Any]:
    """The human-rated axis, reported *alongside* -- never in place of -- the
    automated metrics. Explicitly ``"incomplete"`` -- never silently omitted
    -- whenever gallery-review annotations have not yet been ingested for the
    run under analysis.
    """
    if not gallery_review_rows or not mapping:
        return {"status": "incomplete", "per_condition_preference_rate": {}, "per_condition_artifact_rate": {}, "group_count": 0}

    label_to_condition = {(m.group_key, m.blind_label): m.condition_id for m in mapping}
    preference_wins: dict[str, int] = {}
    preference_total: dict[str, int] = {}
    artifact_counts: dict[str, dict[str, list[int]]] = {}

    for row in gallery_review_rows:
        for label in row.blind_labels:
            condition_id = label_to_condition[(row.group_key, label)]
            preference_total[condition_id] = preference_total.get(condition_id, 0) + 1
            if row.preference == label:
                preference_wins[condition_id] = preference_wins.get(condition_id, 0) + 1

            fields = artifact_counts.setdefault(condition_id, {name: [0, 0] for name in ARTIFACT_FIELDS})
            for image_artifacts in row.ratings[label].per_image_artifacts:
                for field_name, level in image_artifacts.items():
                    if level == "NA":
                        continue
                    fields[field_name][1] += 1
                    if level == "present":
                        fields[field_name][0] += 1

    preference_rate = {cid: preference_wins.get(cid, 0) / total for cid, total in preference_total.items()}
    artifact_rate = {
        cid: {field_name: present / total for field_name, (present, total) in fields.items() if total > 0}
        for cid, fields in artifact_counts.items()
    }
    return {
        "status": "ok",
        "per_condition_preference_rate": preference_rate,
        "per_condition_artifact_rate": artifact_rate,
        "group_count": len(gallery_review_rows),
    }


def analyze_run(
    run_dir: str | Path,
    *,
    reference_condition_id: str = REFERENCE_CONDITION_ID,
    gallery_review_rows: Sequence[GalleryReviewRow] | None = None,
    gallery_review_mapping: Sequence[GalleryReviewMappingEntry] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    condition_summaries = summarize_conditions(run_dir)
    effects = paired_effects(run_dir, reference_condition_id=reference_condition_id)
    gallery_axis = summarize_gallery_review(gallery_review_rows, gallery_review_mapping)

    tables_dir = run_dir / "analysis" / "tables"
    _write_csv(tables_dir / "condition_summaries.csv", condition_summaries)
    _write_csv(tables_dir / "paired_effects.csv", effects)
    _write_json(run_dir / "analysis" / "gallery_review_axis.json", gallery_axis)

    return {"condition_summaries": condition_summaries, "paired_effects": effects, "gallery_review": gallery_axis}
