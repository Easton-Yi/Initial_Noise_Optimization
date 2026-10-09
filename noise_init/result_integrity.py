"""Shared integrity checks for generated galleries and metric tables."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from io_utils import condition_id, file_hash, read_json, read_jsonl, sha256_text


METRIC_CACHE_VERSION = "prompt_aware_v1"


@dataclass(frozen=True)
class RunContract:
    config: dict[str, Any]
    blocks: dict[str, dict[str, Any]]
    condition_ids: tuple[str, ...]
    base_indices: tuple[int, ...]

    @property
    def expected_groups(self) -> set[tuple[str, str]]:
        return {(block_id, condition) for block_id in self.blocks for condition in self.condition_ids}


def expected_condition_ids(config: dict[str, Any]) -> tuple[str, ...]:
    """Expand the frozen grid without importing run_experiment (which imports analysis)."""
    values: list[str] = []
    baseline = config.get("baseline")
    if not isinstance(baseline, dict):
        raise RuntimeError("Run manifest has no baseline configuration")
    if baseline.get("enabled", True):
        values.extend(condition_id("baseline", float(alpha)) for alpha in baseline.get("alpha_values", []))
    for section_name, family in (("same_phase_floor", "same_phase"),
                                 ("independent_white", "independent_white")):
        section = config.get(section_name)
        if not isinstance(section, dict):
            raise RuntimeError(f"Run manifest has no {section_name} configuration")
        if section.get("enabled", False):
            values.extend(condition_id(family, float(alpha), float(gamma))
                          for alpha in section.get("alpha_values", [])
                          for gamma in section.get("gamma_values", []))
    if not values:
        raise RuntimeError("Frozen run configuration contains no enabled conditions")
    if len(set(values)) != len(values):
        raise RuntimeError("Frozen run configuration expands to duplicate condition IDs")
    return tuple(values)


def load_run_contract(run_dir: str | Path) -> RunContract:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    blocks_path = run_dir / "blocks.jsonl"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing run metadata: {manifest_path}")
    manifest = read_json(manifest_path)
    config = manifest.get("resolved_config")
    if not isinstance(config, dict) or not config:
        raise RuntimeError(f"Run manifest has no resolved_config: {manifest_path}")
    block_rows = read_jsonl(blocks_path)
    if not block_rows:
        raise RuntimeError(f"Run block metadata is missing or empty: {blocks_path}")
    required = {"block_id", "prompt_id", "prompt"}
    malformed = [index for index, row in enumerate(block_rows) if required - set(row)]
    if malformed:
        raise RuntimeError(f"Run block metadata has {len(malformed)} malformed rows; examples={malformed[:3]}")
    blocks = {str(row["block_id"]): row for row in block_rows}
    if len(blocks) != len(block_rows):
        raise RuntimeError("Run block metadata contains duplicate block_id values")
    gallery_size = int(config.get("experiment", {}).get("gallery_size", 0))
    if gallery_size != 4:
        raise RuntimeError(f"Frozen run configuration must use gallery_size=4, got {gallery_size}")
    return RunContract(config, blocks, expected_condition_ids(config), tuple(range(gallery_size)))


def load_generation_records(run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    return [row for path in sorted(run_dir.glob("generations/**/samples.jsonl")) for row in read_jsonl(path)]


def validate_generation_records(run_dir: str | Path, *, require_complete: bool) -> tuple[RunContract, list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    """Validate all present galleries; optionally require the frozen Cartesian product."""
    contract = load_run_contract(run_dir)
    rows = load_generation_records(run_dir)
    if not rows:
        raise RuntimeError("No generated sample records found")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    malformed: list[str] = []
    prompt_mismatches: list[str] = []
    missing_images: list[str] = []
    unexpected: list[str] = []
    model_ids: set[str] = set()
    required = {"block_id", "condition_id", "base_index", "image_path", "model_id", "prompt_id", "prompt"}
    for number, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            malformed.append(f"row {number}: missing {sorted(missing)}")
            continue
        block_id, candidate = str(row["block_id"]), str(row["condition_id"])
        try:
            base_index = int(row["base_index"])
        except (TypeError, ValueError):
            malformed.append(f"{block_id}/{candidate}: base_index={row['base_index']!r}")
            continue
        key = (block_id, candidate)
        groups.setdefault(key, []).append(row)
        model_ids.add(str(row["model_id"]))
        if key not in contract.expected_groups:
            unexpected.append(f"{block_id}/{candidate}/{base_index}")
        block = contract.blocks.get(block_id)
        if block is not None and (row["prompt_id"] != block["prompt_id"] or row["prompt"] != block["prompt"]):
            prompt_mismatches.append(f"{block_id}/{candidate}/{base_index}")
        if not Path(row["image_path"]).is_file():
            missing_images.append(f"{block_id}/{candidate}/{base_index}")

    bad_groups: list[str] = []
    expected_indices = set(contract.base_indices)
    for (block_id, candidate), gallery in groups.items():
        indices = [int(row["base_index"]) for row in gallery]
        if len(gallery) != len(contract.base_indices) or set(indices) != expected_indices or len(set(indices)) != len(indices):
            bad_groups.append(f"{block_id}/{candidate}: count={len(gallery)}, indices={indices}")

    missing_groups = sorted(contract.expected_groups - set(groups)) if require_complete else []
    issues = []
    _add_issue(issues, "malformed generation records", malformed)
    _add_issue(issues, "unexpected block/condition records", unexpected)
    _add_issue(issues, "prompt metadata mismatches", prompt_mismatches)
    _add_issue(issues, "missing PNG files", missing_images)
    _add_issue(issues, "incomplete or duplicate-index galleries", bad_groups)
    _add_issue(issues, "missing expected galleries", [f"{block}/{condition}" for block, condition in missing_groups])
    if len(model_ids) > 1:
        issues.append(f"mixed model_id values: {sorted(model_ids)[:4]}")
    if issues:
        raise RuntimeError("Generation integrity check failed: " + "; ".join(issues))
    return contract, rows, groups


def enabled_metric_names(config: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    quality = tuple(metric for metric, section in (
        ("clip_cosine", config.get("quality_metrics", {}).get("clip", {})),
        ("hpsv3", config.get("quality_metrics", {}).get("hpsv3", {})),
    ) if section.get("enabled", False))
    pairwise = tuple(metric for metric, section in (
        ("dreamsim_mean_pair_distance", config.get("diversity_metrics", {}).get("dreamsim", {})),
        ("lpips_alex_mean_pair_distance", config.get("diversity_metrics", {}).get("lpips", {})),
    ) if section.get("enabled", False))
    group_only = (("vendi_clip",) if config.get("diversity_metrics", {}).get("vendi_clip", {}).get("enabled", False) else ())
    return quality, pairwise, group_only


def validate_analysis_inputs(run_dir: str | Path) -> RunContract:
    """Require a complete frozen run and exact, prompt-aware metric coverage."""
    run_dir = Path(run_dir)
    contract, generation_rows, groups = validate_generation_records(run_dir, require_complete=True)
    quality, pairwise, group_only = enabled_metric_names(contract.config)
    manifest_path = run_dir / "metrics" / "metric_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Metric integrity check failed: missing {manifest_path}; rerun metrics")
    manifest = read_json(manifest_path)
    metric_hash = manifest.get("metric_config_hash")
    if (manifest.get("metric_cache_version") != METRIC_CACHE_VERSION or
            manifest.get("complete") is not True or not metric_hash):
        raise RuntimeError("Metric integrity check failed: old or incomplete metric manifest; rerun metrics")

    generation_by_key = {
        (str(row["block_id"]), str(row["condition_id"]), int(row["base_index"])): row
        for row in generation_rows
    }
    per_image = _read_csv(run_dir / "metrics" / "per_image.csv")
    per_pair = _read_csv(run_dir / "metrics" / "per_pair.csv")
    per_group = _read_csv(run_dir / "metrics" / "per_group.csv")
    issues: list[str] = []
    _validate_per_image(per_image, generation_by_key, quality, str(metric_hash), issues)
    _validate_per_pair(per_pair, groups, pairwise, str(metric_hash), contract.base_indices, issues)
    _validate_per_group(per_group, groups, quality, pairwise, group_only, str(metric_hash), issues)
    if issues:
        raise RuntimeError("Metric integrity check failed: " + "; ".join(issues) + "; rerun metrics")
    return contract


def _validate_per_image(rows: list[dict[str, Any]], generation: dict[tuple[str, str, int], dict[str, Any]],
                        metrics: tuple[str, ...], metric_hash: str, issues: list[str]) -> None:
    expected = {(block, condition, index, metric) for block, condition, index in generation for metric in metrics}
    counts: dict[tuple[str, str, int, str], int] = {}
    image_hashes: dict[str, str] = {}
    invalid: list[str] = []
    for row in rows:
        try:
            key = (str(row["block_id"]), str(row["condition_id"]), int(row["base_index"]), str(row["metric"]))
            counts[key] = counts.get(key, 0) + 1
            generated = generation.get(key[:3])
            image_path = "" if generated is None else str(generated["image_path"])
            if image_path and image_path not in image_hashes:
                image_hashes[image_path] = file_hash(image_path)
            if (key not in expected or generated is None or row.get("metric_config_hash") != metric_hash or
                    row.get("image_hash") != image_hashes.get(image_path) or
                    row.get("prompt_hash") != sha256_text(generated["prompt"]) or
                    not _finite(row.get("score"))):
                invalid.append(_format_key(key))
        except (KeyError, TypeError, ValueError, OSError):
            invalid.append(f"malformed row {len(invalid)}")
    missing = sorted(expected - set(counts))
    duplicate = sorted(key for key, count in counts.items() if count != 1)
    _add_issue(issues, "invalid per_image rows", invalid)
    _add_issue(issues, "missing per_image rows", [_format_key(key) for key in missing])
    _add_issue(issues, "duplicate per_image logical keys", [_format_key(key) for key in duplicate])


def _validate_per_pair(rows: list[dict[str, Any]], groups: dict[tuple[str, str], list[dict[str, Any]]],
                       metrics: tuple[str, ...], metric_hash: str, base_indices: tuple[int, ...],
                       issues: list[str]) -> None:
    pairs = tuple(combinations(base_indices, 2))
    expected = {(block, condition, metric, left, right)
                for block, condition in groups for metric in metrics for left, right in pairs}
    counts: dict[tuple[str, str, str, int, int], int] = {}
    invalid: list[str] = []
    for row in rows:
        try:
            left, right = sorted((int(row["left_base_index"]), int(row["right_base_index"])))
            key = (str(row["block_id"]), str(row["condition_id"]), str(row["metric"]), left, right)
            counts[key] = counts.get(key, 0) + 1
            if (left == right or key not in expected or row.get("metric_config_hash") != metric_hash or not _finite(row.get("score"))):
                invalid.append(_format_key(key))
        except (KeyError, TypeError, ValueError):
            invalid.append(f"malformed row {len(invalid)}")
    missing = sorted(expected - set(counts))
    duplicate = sorted(key for key, count in counts.items() if count != 1)
    _add_issue(issues, "invalid per_pair rows", invalid)
    _add_issue(issues, "missing per_pair rows", [_format_key(key) for key in missing])
    _add_issue(issues, "duplicate per_pair logical keys", [_format_key(key) for key in duplicate])


def _validate_per_group(rows: list[dict[str, Any]], groups: dict[tuple[str, str], list[dict[str, Any]]],
                        quality: tuple[str, ...], pairwise: tuple[str, ...], group_only: tuple[str, ...],
                        metric_hash: str, issues: list[str]) -> None:
    expected_n = {**{metric: 4 for metric in quality}, **{metric: 6 for metric in pairwise},
                  **{metric: 4 for metric in group_only}}
    expected = {(block, condition, metric) for block, condition in groups for metric in expected_n}
    counts: dict[tuple[str, str, str], int] = {}
    invalid: list[str] = []
    for row in rows:
        try:
            key = (str(row["block_id"]), str(row["condition_id"]), str(row["metric"]))
            counts[key] = counts.get(key, 0) + 1
            if (key not in expected or row.get("metric_config_hash") != metric_hash or
                    int(row.get("n", -1)) != expected_n.get(key[2]) or not _finite(row.get("score"))):
                invalid.append(_format_key(key))
        except (KeyError, TypeError, ValueError):
            invalid.append(f"malformed row {len(invalid)}")
    missing = sorted(expected - set(counts))
    duplicate = sorted(key for key, count in counts.items() if count != 1)
    _add_issue(issues, "invalid per_group rows", invalid)
    _add_issue(issues, "missing per_group rows", [_format_key(key) for key in missing])
    _add_issue(issues, "duplicate per_group logical keys", [_format_key(key) for key in duplicate])


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _format_key(key: tuple[Any, ...]) -> str:
    return "/".join(map(str, key))


def _add_issue(issues: list[str], label: str, examples: list[str]) -> None:
    if examples:
        issues.append(f"{label}={len(examples)} examples={examples[:3]}")
