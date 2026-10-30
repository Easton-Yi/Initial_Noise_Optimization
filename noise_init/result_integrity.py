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
    from run_support import conditions
    try:
        return tuple(c.identifier for c in conditions(config))
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


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
    prompts = {}
    for block in blocks.values():
        pid = str(block["prompt_id"])
        if pid in prompts and prompts[pid] != block["prompt"]:
            raise RuntimeError(f"Conflicting prompt text for prompt_id={pid}")
        prompts[pid] = block["prompt"]
    gallery_size = int(config.get("experiment", {}).get("gallery_size", 0))
    if gallery_size != 4:
        raise RuntimeError(f"Frozen run configuration must use gallery_size=4, got {gallery_size}")
    return RunContract(config, blocks, expected_condition_ids(config), tuple(range(gallery_size)))


def load_generation_records(run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir).resolve()
    rows = []
    for path in sorted(run_dir.glob("generations/**/samples.jsonl")):
        for saved in read_jsonl(path):
            row = dict(saved)
            if row.get("image_path"):
                resolved = resolve_recorded_image(run_dir, row)
                if str(resolved) != row["image_path"]:
                    row["recorded_image_path"] = row["image_path"]
                    row["image_path"] = str(resolved)
            rows.append(row)
    return rows


def resolve_recorded_image(run_dir: Path, row: dict[str, Any]) -> Path:
    """Rebase only an identity-matching generation suffix, never search by basename.

    Prefer the explicitly selected run over an old absolute root, even when the
    old root still exists. Alias rows refer to their recorded source condition.
    Resolution changes only the in-memory row; saved samples/metric hashes remain
    unchanged and analysis still verifies each image against the metric record.
    """
    recorded = Path(row["image_path"])
    try:
        index = row["base_index"]
        if type(index) is not int or index < 0:
            return recorded
        source = row.get("alias_of_condition_id") or row["condition_id"]
        relative = Path("generations") / str(row["model_id"]) / str(row["block_id"]) / str(source) / f"image_{index:02d}.png"
        root = Path(run_dir).resolve()
        candidate = (root / relative).resolve()
        # Reject traversal and require the stored suffix to match the full
        # model/block/condition/sample identity, not merely image_00.png.
        if candidate.is_relative_to(root) and recorded.parts[-len(relative.parts):] == relative.parts:
            return candidate
    except (KeyError, TypeError, ValueError):
        pass  # The normal record-integrity checks report malformed metadata.
    if not recorded.is_absolute():
        candidate = (Path(run_dir).resolve() / recorded).resolve()
        if candidate.is_relative_to(Path(run_dir).resolve()):
            return candidate
    return recorded


def validate_generation_records(run_dir: str | Path, *, require_complete: bool, condition_ids=None) -> tuple[RunContract, list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    """Validate all present galleries; optionally require the frozen Cartesian product."""
    contract = load_run_contract(run_dir)
    if condition_ids is not None:
        from dataclasses import replace
        if not condition_ids or not set(condition_ids) <= set(contract.condition_ids):
            raise RuntimeError("Selected conditions must belong to the frozen run")
        contract = replace(contract, condition_ids=tuple(condition_ids))
    expected_groups = contract.expected_groups
    rows = load_generation_records(run_dir)
    if condition_ids is not None:
        rows = [r for r in rows if r.get("condition_id") in condition_ids]
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
        if type(row["base_index"]) is not int:
            malformed.append(f"row {number}: base_index must be an integer")
            continue
        validate_condition_metadata(row, context=f"generation row {number}")
        key = (block_id, candidate)
        groups.setdefault(key, []).append(row)
        model_ids.add(str(row["model_id"]))
        if key not in expected_groups:
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

    missing_groups = sorted(expected_groups - set(groups)) if require_complete else []
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
    from coupled_sampling import enabled
    if enabled(contract.config):
        from coupled_run import validate_saved_groups
        validate_saved_groups(Path(run_dir), contract, groups)
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


def validate_analysis_inputs(run_dir: str | Path, *, condition_ids=None) -> RunContract:
    """Require a complete frozen run and exact, prompt-aware metric coverage."""
    run_dir = Path(run_dir)
    contract, generation_rows, groups = validate_generation_records(run_dir, require_complete=True, condition_ids=condition_ids)
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
    if condition_ids is not None:
        per_image, per_pair, per_group = ([r for r in rows if r.get("condition_id") in condition_ids]
                                        for rows in (per_image, per_pair, per_group))
    issues: list[str] = []
    _validate_per_image(per_image, generation_by_key, quality, str(metric_hash), issues)
    _validate_per_pair(per_pair, groups, pairwise, str(metric_hash), contract.base_indices, issues)
    _validate_per_group(per_group, groups, quality, pairwise, group_only, str(metric_hash), issues)
    for filename, rows in (("per_image.csv", per_image), ("per_pair.csv", per_pair), ("per_group.csv", per_group)):
        for row in rows:
            validate_condition_metadata(row, context=filename)
            block = contract.blocks.get(str(row.get("block_id")))
            if block is None or any(row.get(key) != str(block[key]) for key in ("prompt_id", "prompt")):
                issues.append(f"{filename}: prompt identity mismatch for {row.get('block_id')}")
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


def validate_condition_metadata(row, *, context):
    try:
        family = row.get("family") or row["condition_id"].split("/")[0]
        if family in {"antithetic", "repulsive"}:
            if not {"alpha", "gamma"} <= set(row):
                raise ValueError("Coupled metadata must include null alpha/gamma fields")
            if row.get("alpha") not in (None, "") or row.get("gamma") not in (None, ""):
                raise ValueError("Coupled methods must have null alpha/gamma")
            if row["condition_id"] != condition_id(family, None, None):
                raise ValueError("Coupled condition ID mismatch")
            if row.get("method") != family or row.get("normalization_profile") != "none":
                raise ValueError("Contradictory coupled method/normalization metadata")
            return
        alpha = float(row["alpha"])
        gamma = None if family == "baseline" else float(row["gamma"])
        if family not in {"baseline", "same_phase", "independent_white"} or not math.isfinite(alpha) or alpha < 0:
            raise ValueError("invalid family/alpha")
        if gamma is not None and (not math.isfinite(gamma) or not 0 <= gamma <= 1):
            raise ValueError("invalid gamma")
        if row["condition_id"] != condition_id(family, alpha, gamma):
            raise ValueError("condition ID does not match family/alpha/gamma")
        if family == "baseline" and row.get("gamma") not in (None, ""):
            raise ValueError("baseline must have null gamma")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{context}: invalid condition metadata {row.get('condition_id')}: {exc}") from exc
