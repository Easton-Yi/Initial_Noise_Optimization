"""Quality/diversity metric stage over a generated run (plan section on
``metrics.py``).

Deliberately does **not** call ``noise_init.metric_runner.evaluate_run`` and
does **not** import ``compat_generation`` at all -- two separate reasons:

1. ``evaluate_run`` filters rows to ``Path(row["image_path"]).name.startswith
   ("image_")``, but ``runner.py`` always names a sample image literally
   ``"image.png"``; that filter would silently discard every one of our
   records. Its ``_common()`` helper also hardcodes ``alpha``/``gamma``/
   ``prompt`` fields that don't exist in our PC-condition schema (we have
   ``pc_group_id``/``tau``/``gate_id``/``basis_hash``/``calibration_hash``,
   and no stored ``prompt`` text -- only ``prompt_id``, resolved here via
   ``manifests.prompt_by_id``). And its diversity/Vendi loops silently
   ``continue`` past any gallery whose size isn't 4, which would swallow the
   preview track's single-image galleries without a trace.
2. ``compat_generation`` eagerly imports ``noise_init``'s ``model_adapters``/
   ``noise_methods`` at module top. Even though (as of this writing) neither
   actually imports ``diffusers``/``transformers`` eagerly, keeping this
   module's import graph structurally independent of ``compat_generation`` --
   using a private, stdlib-only JSONL reader instead of importing
   ``read_jsonl`` from it -- means a future change to the generation shim can
   never silently force the metrics venv to have the generation stack
   importable, and vice versa. ``compat_metrics`` (the only ``noise_init``
   contact point this module has) is imported lazily inside
   ``_build_metric_runner``, never at module top.

Every non-4-image gallery -- in particular every preview-track gallery,
which is single-image by design -- gets an explicit ``not_applicable``/
``incomplete`` status in ``per_group`` rather than being silently skipped.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from pc_specific_psd import manifests

QUALITY_METRICS_CONFIG: dict[str, dict[str, Any]] = {
    "clip": {"enabled": True, "checkpoint": "openai/clip-vit-large-patch14"},
    "hpsv3": {"enabled": True},
}
DIVERSITY_METRICS_CONFIG: dict[str, dict[str, Any]] = {
    "dreamsim": {"enabled": True},
    "lpips": {"enabled": True, "backbone": "alex"},
    "vendi_clip": {"enabled": True},
}

_COMMON_FIELDS = (
    "run_id", "condition_id", "prompt_id", "block_id", "base_index",
    "pc_group_id", "tau", "gate_id", "basis_hash", "calibration_hash",
)

GALLERY_SIZE_OK = 4
GALLERY_SIZE_PREVIEW = 1


class MetricsError(RuntimeError):
    """Raised for missing-input failures analogous to runner.py's RunnerError."""


def collect_sample_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    """Reads every ``generations/**/sample.jsonl`` record under ``run_dir``.

    A plain stdlib JSONL reader (matching ``noise_init.io_utils.read_jsonl``'s
    semantics -- blank lines skipped, missing file yields no rows for that
    path) so this module never imports ``compat_generation``.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(run_dir).glob("generations/**/sample.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def gallery_status(size: int) -> str:
    """``4`` -> ``"ok"`` (a full-pilot/full-gallery 4-image group); ``1`` ->
    ``"not_applicable"`` (a preview-track single-image "gallery", where
    pairwise diversity is structurally meaningless, not merely missing);
    anything else -> ``"incomplete"`` (an unexpectedly partial run).
    """
    if size == GALLERY_SIZE_OK:
        return "ok"
    if size == GALLERY_SIZE_PREVIEW:
        return "not_applicable"
    return "incomplete"


def group_by_gallery(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Groups sample rows by ``(block_id, condition_id)`` -- the same unit a
    4-image gallery or a single preview draw occupies -- sorted by
    ``base_index`` within each group.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["block_id"], row["condition_id"]), []).append(row)
    for gallery in groups.values():
        gallery.sort(key=lambda row: row["base_index"])
    return groups


def _common(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _COMMON_FIELDS if key in row}


def _metric_config_hash(versions: dict[str, str]) -> str:
    payload = {"quality_metrics": QUALITY_METRICS_CONFIG, "diversity_metrics": DIVERSITY_METRICS_CONFIG, "versions": versions}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


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


def _build_metric_runner(device: str):
    """Lazily reaches into ``noise_init`` -- never at module top -- so
    importing ``pc_specific_psd.metrics`` never requires the metrics venv's
    ``transformers==4.45.2``/``lpips``/``dreamsim``/``hpsv3``/``vendi-score``
    stack; only actually calling this function does.
    """
    from pc_specific_psd.compat_metrics import MetricRunner

    config = {"quality_metrics": QUALITY_METRICS_CONFIG, "diversity_metrics": DIVERSITY_METRICS_CONFIG, "cache_dir": None}
    return MetricRunner(config=config, device=torch.device(device))


def _aggregate_quality_per_group(per_image: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean/std/min/max per ``(block_id, condition_id, metric)``. Gallery-size
    agnostic by construction -- a size-1 preview "gallery" just yields
    ``n=1``, ``std=0.0``, so quality metrics never need a not_applicable
    status the way pairwise diversity metrics do.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in per_image:
        groups.setdefault((row["block_id"], row["condition_id"], row["metric"]), []).append(row)
    result = []
    for (_, _, metric), values in groups.items():
        scores = [float(v["score"]) for v in values]
        result.append({
            **_common(values[0]), "metric": metric, "status": "ok",
            "score": float(np.mean(scores)), "n": len(scores),
            "std": float(np.std(scores, ddof=0)), "minimum": float(np.min(scores)), "maximum": float(np.max(scores)),
            "metric_config_hash": values[0]["metric_config_hash"],
        })
    return result


@dataclass(frozen=True)
class MetricsSummary:
    per_image: list[dict[str, Any]]
    per_pair: list[dict[str, Any]]
    per_group: list[dict[str, Any]]
    metric_config_hash: str
    versions: dict[str, str]


def run_metrics(
    run_dir: str | Path,
    *,
    metric_runner=None,
    device: str = "cpu",
    force: bool = False,
) -> MetricsSummary:
    """Computes quality metrics per-image and diversity metrics per-gallery,
    writing ``metrics/{per_image,per_pair,per_group}.csv`` and
    ``metrics/metric_manifest.json`` under ``run_dir`` -- the same on-disk
    shape ``noise_init.metric_runner.evaluate_run`` produces, but built by a
    schema-compatible reimplementation (see module docstring) rather than by
    calling it.

    ``metric_runner`` mirrors ``runner.generate_manifest``'s own ``adapter``
    override: tests inject a fake one; production leaves it ``None`` so a
    real ``MetricRunner`` is built lazily against ``device``.
    """
    run_dir = Path(run_dir)
    rows = collect_sample_rows(run_dir)
    if not rows:
        raise MetricsError(f"No generated sample records found under {run_dir}/generations")

    metric_runner = metric_runner or _build_metric_runner(device)
    versions = metric_runner.metric_versions()
    metric_config_hash = _metric_config_hash(versions)
    metrics_dir = run_dir / "metrics"

    existing = [] if force else _read_csv(metrics_dir / "per_image.csv")
    score_cache = {(r["image_hash"], r["metric_config_hash"], r["metric"]): float(r["score"]) for r in existing}
    current_keys = {(row["block_id"], row["condition_id"], int(row["base_index"])) for row in rows}
    per_image: list[dict[str, Any]] = [
        r for r in existing
        if (r["block_id"], r["condition_id"], int(r.get("base_index", -1))) in current_keys
        and r.get("metric_config_hash") == metric_config_hash
    ]
    existing_keys = {
        (r["block_id"], r["condition_id"], int(r.get("base_index", -1)), r["image_hash"], r["metric_config_hash"], r["metric"])
        for r in per_image
    }

    quality_metrics: tuple[tuple[str, bool], ...] = (
        ("clip_cosine", QUALITY_METRICS_CONFIG["clip"]["enabled"]),
        ("hpsv3", QUALITY_METRICS_CONFIG["hpsv3"]["enabled"]),
    )
    for metric_name, enabled in quality_metrics:
        if not enabled:
            continue
        for row in rows:
            record_key = (row["block_id"], row["condition_id"], row["base_index"], row["image_hash"], metric_config_hash, metric_name)
            if record_key in existing_keys:
                continue
            cache_key = (row["image_hash"], metric_config_hash, metric_name)
            score = score_cache.get(cache_key)
            if score is None:
                prompt_text = manifests.prompt_by_id(row["prompt_id"]).text
                path = Path(row["image_path"])
                score = metric_runner.clip_cosine(path, prompt_text) if metric_name == "clip_cosine" else metric_runner.hpsv3(path, prompt_text)
                score_cache[cache_key] = score
            per_image.append({**_common(row), "metric": metric_name, "score": score, "image_hash": row["image_hash"], "metric_config_hash": metric_config_hash})
            existing_keys.add(record_key)
        metric_runner.release_models()

    galleries = group_by_gallery(rows)
    per_pair: list[dict[str, Any]] = []
    per_group: list[dict[str, Any]] = _aggregate_quality_per_group(per_image)

    pairwise_metrics: tuple[tuple[str, bool, Callable[[Path, Path], float]], ...] = (
        ("dreamsim_mean_pair_distance", DIVERSITY_METRICS_CONFIG["dreamsim"]["enabled"], metric_runner.dreamsim_distance),
        ("lpips_alex_mean_pair_distance", DIVERSITY_METRICS_CONFIG["lpips"]["enabled"], metric_runner.lpips_distance),
    )
    for metric_name, enabled, distance_fn in pairwise_metrics:
        if not enabled:
            continue
        for gallery in galleries.values():
            status = gallery_status(len(gallery))
            if status != "ok":
                per_group.append({**_common(gallery[0]), "metric": metric_name, "status": status, "n": len(gallery)})
                continue
            values = []
            for left, right in combinations(gallery, 2):
                value = distance_fn(Path(left["image_path"]), Path(right["image_path"]))
                values.append(value)
                per_pair.append({
                    **_common(left), "metric": metric_name,
                    "left_base_index": left["base_index"], "right_base_index": right["base_index"],
                    "score": value, "metric_config_hash": metric_config_hash,
                })
            per_group.append({
                **_common(gallery[0]), "metric": metric_name, "status": "ok",
                "score": float(np.mean(values)), "n": 6, "metric_config_hash": metric_config_hash,
            })
        metric_runner.release_models()

    if DIVERSITY_METRICS_CONFIG["vendi_clip"]["enabled"]:
        try:
            from vendi_score import vendi
        except ImportError as error:
            raise ImportError("vendi-score is required for vendi_clip") from error
        for gallery in galleries.values():
            status = gallery_status(len(gallery))
            if status != "ok":
                per_group.append({**_common(gallery[0]), "metric": "vendi_clip", "status": status, "n": len(gallery)})
                continue
            vectors = np.stack([metric_runner.clip_embedding(Path(item["image_path"])) for item in gallery])
            kernel = vectors @ vectors.T
            kernel = (kernel + kernel.T) / 2
            if not np.allclose(np.diag(kernel), 1.0, atol=1e-4):
                raise MetricsError("Vendi CLIP kernel diagonal is not one")
            value = float(vendi.score_K(kernel))
            per_group.append({
                **_common(gallery[0]), "metric": "vendi_clip", "status": "ok",
                "score": value, "n": 4, "metric_config_hash": metric_config_hash,
            })
        metric_runner.release_models()

    _write_csv(metrics_dir / "per_image.csv", per_image)
    _write_csv(metrics_dir / "per_pair.csv", per_pair)
    _write_csv(metrics_dir / "per_group.csv", per_group)
    _write_json(metrics_dir / "metric_manifest.json", {"metric_config_hash": metric_config_hash, "versions": versions})
    return MetricsSummary(per_image=per_image, per_pair=per_pair, per_group=per_group, metric_config_hash=metric_config_hash, versions=versions)
