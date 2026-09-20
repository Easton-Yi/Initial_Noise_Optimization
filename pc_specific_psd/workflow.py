"""Single-entry-point orchestrator for the small-scope workflow-consolidation
pass (see the approved plan). ``run_workflow`` chains the 14 stages the
existing 15-command CLI already supports, one call each, and STOPs twice for
the two required human-review moments.

Deliberately imports only lower-level modules, never ``cli`` -- ``cli.py``
must import this module (to register the ``workflow`` subcommand), so the
reverse import would close a cycle. Any small CLI-layer helper this module
needs (``_default_path``, the calibration/validation bank builders, the
dataclass-to-JSON coercion) is therefore duplicated here rather than
imported, matching the codebase's existing "duplicated twin" pattern (see
``runner.basis_hash_for``/``probing.probe_basis_hash_for`` and
``adapters.smoke_check_cache_key``'s own docstring).

Resumability is deliberately thin: ``probing.generate_probes``,
``runner.generate_manifest``/``generate_preview``/``generate_full_pilot`` are
already individually idempotent (skip a complete sample dir, refuse to
silently overwrite an incomplete one), so this module never tracks "has
stage N completed" for those -- it just always calls them and lets that
built-in resume behavior do the right thing. The only state this module
persists (in ``<config_dir>/<run.name>_workflow_state.json``) is the
smoke-check cache and a content hash of each review file at the moment it
was last consumed, so a review file edited after being consumed is caught as
a conflict rather than silently mixed into a stale downstream artifact.
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from pc_specific_psd import (
    analysis,
    basis as basis_module,
    calibration,
    config,
    manifests,
    metrics,
    probing,
    review,
    runner,
)
from pc_specific_psd.adapters import SDXLTurboAdapterPCA, smoke_check, smoke_check_cache_key
from pc_specific_psd.compat_generation import file_hash, read_json, write_json


class WorkflowError(ValueError):
    """Raised for workflow-specific refusals/hard-stops. A ``ValueError``
    subclass so ``cli.main()``'s existing exception tuple catches it without
    needing an edit beyond the new ``workflow`` subparser/handler.
    """


NO_CANDIDATES_MESSAGE = "本轮无候选"
NO_APPROVED_CONDITIONS_MESSAGE = "本轮无通过预览的编辑条件"
GROUP_REVIEW_CAVEAT = "这是多 seed 支持的探索性筛选，不额外宣称做过独立的候选确认实验."


# ---------------------------------------------------------------------------
# Small CLI-layer helpers, duplicated from cli.py rather than imported (see
# module docstring for why this module can never import cli.py).
# ---------------------------------------------------------------------------


def _default_path(cfg: config.PCASpecificPSDConfig, suffix: str) -> Path:
    return Path(cfg.config_path).resolve().parent / f"{cfg.run.name}_{suffix}"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _calibration_bank(cfg: config.PCASpecificPSDConfig, *, height: int, width: int) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(cfg.psd.calibration_bank_seed)
    return torch.randn((cfg.psd.calibration_bank_size, cfg.basis.channels, height, width), generator=generator, dtype=torch.float32)


def _validation_bank(cfg: config.PCASpecificPSDConfig, *, height: int, width: int) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(cfg.psd.validation_bank_seed)
    return torch.randn((cfg.psd.validation_bank_size, cfg.basis.channels, height, width), generator=generator, dtype=torch.float32)


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    return read_json(path)


def _save_state(path: Path, state: dict) -> None:
    write_json(path, state)


def _continue_command(config_path: str | Path) -> str:
    return f"python -m pc_specific_psd workflow --config {config_path}"


def _maybe_subprocess(python_path: str, argv: list[str]) -> Optional[dict]:
    """Dispatches ``argv`` (an existing CLI command + flags) to a different
    interpreter via ``subprocess.run`` when ``python_path`` differs from the
    one currently running this process; returns ``None`` (meaning "run it
    in-process instead") when they match. Never used for ``analyze`` -- that
    stage has no separate-venv requirement.
    """
    if python_path == sys.executable:
        return None
    completed = subprocess.run(
        [python_path, "-m", "pc_specific_psd", *argv],
        capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise WorkflowError(
            f"subprocess `{python_path} -m pc_specific_psd {' '.join(argv)}` failed "
            f"(exit {completed.returncode}): {completed.stderr.strip()}"
        )
    return {"status": "ok", "dispatched_via": python_path, "argv": argv}


def _check_review_conflict(state: dict, state_key: str, review_path: Path, downstream_path: Path) -> None:
    if not downstream_path.exists():
        return
    recorded = state.get(state_key)
    if recorded is None:
        return
    current_hash = file_hash(review_path)
    if current_hash != recorded:
        raise WorkflowError(
            f"{review_path} changed after it was already used to build {downstream_path} -- "
            "re-run the affected downstream stages (delete/regenerate them) or restore the "
            "previously reviewed file before continuing"
        )


# ---------------------------------------------------------------------------
# Dry run: reports the planned stage sequence and current resume point
# without generating anything or loading a model.
# ---------------------------------------------------------------------------


def _dry_run_report(
    cfg: config.PCASpecificPSDConfig,
    *,
    candidates_path: Path,
    probe_review_path: Path,
    preview_review_path: Path,
    approved_conditions_path: Path,
) -> dict:
    basis_path = cfg.resolve_root(cfg.basis.basis_output_path)
    probe_run_dir = cfg.resolve_root(cfg.run.outputs_root) / f"{cfg.run.name}_probe"
    preview_run_dir = cfg.resolve_root(cfg.run.outputs_root) / f"{cfg.run.name}_preview"
    full_run_dir = cfg.resolve_root(cfg.run.outputs_root) / f"{cfg.run.name}_full"

    stages = [
        {"stage": "validate_config", "done": True},
        {"stage": "build_basis", "done": basis_path is not None and basis_path.exists()},
        {"stage": "smoke_check_pre_probe", "done": None},
        {"stage": "probe", "done": probe_run_dir.exists()},
        {"stage": "review_1_probe_group_review", "done": probe_review_path.exists()},
        {"stage": "select_candidates", "done": candidates_path.exists()},
        {"stage": "calibrate", "done": cfg.resolve_root(cfg.psd.calibration_result_path).exists() if cfg.psd.calibration_result_path else False},
        {"stage": "generate_preview", "done": preview_run_dir.exists()},
        {"stage": "review_2_preview_exclusion_review", "done": preview_review_path.exists()},
        {"stage": "approve_conditions", "done": approved_conditions_path.exists()},
        {"stage": "smoke_check_pre_full_pilot", "done": None},
        {"stage": "generate_full_pilot", "done": full_run_dir.exists()},
        {"stage": "metrics", "done": (full_run_dir / "metrics").exists()},
        {"stage": "analyze", "done": (full_run_dir / "analysis").exists()},
    ]
    next_stage = next((s["stage"] for s in stages if s["done"] is False), "complete")
    return {"status": "dry-run", "stages": stages, "next_stage": next_stage}


# ---------------------------------------------------------------------------
# Individual stage bodies.
# ---------------------------------------------------------------------------


def _stage_build_basis(cfg: config.PCASpecificPSDConfig, config_path: str, *, generation_python: str) -> dict:
    output_path = cfg.resolve_root(cfg.basis.basis_output_path)
    if output_path is not None and output_path.exists():
        return {"status": "skipped", "basis_output_path": str(output_path)}

    dispatched = _maybe_subprocess(generation_python, ["build-basis", "--config", str(config_path)])
    if dispatched is not None:
        return dispatched

    if cfg.basis.dataset_manifest_path is None:
        raise WorkflowError(
            "basis.dataset_manifest_path is not set -- real basis construction requires a dataset "
            "manifest path (never auto-downloaded); supply one in the config before running workflow for real"
        )
    manifest_path = cfg.resolve_root(cfg.basis.dataset_manifest_path)
    manifest = basis_module.load_dataset_manifest(manifest_path)
    basis_adapter = SDXLTurboAdapterPCA(cfg.model.as_model_config_dict())
    built = basis_module.build_pca_basis(
        manifest, basis_adapter.encode_images_for_basis,
        patch_size=cfg.basis.patch_size, channels=cfg.basis.channels,
        patches_per_image=cfg.basis.patches_per_image, sampling_seed=cfg.basis.sampling_seed,
        synthetic=False,
    )
    basis_module.split_half_stability(
        manifest, basis_adapter.encode_images_for_basis,
        patch_size=cfg.basis.patch_size, channels=cfg.basis.channels,
        patches_per_image=cfg.basis.patches_per_image, sampling_seed=cfg.basis.sampling_seed,
        split_seed=cfg.basis.split_seed, synthetic=False,
        num_leading_components=cfg.basis.num_leading_components,
    )
    basis_module.save_basis(built, output_path)
    if cfg.generation.release_model_after_generation:
        basis_adapter.close()
    return {"status": "ok", "basis_output_path": str(output_path), "num_samples": built.num_samples}


def _run_smoke_check(cfg: config.PCASpecificPSDConfig, state: dict, adapter_obj, *, allow_synthetic_basis: bool) -> dict:
    key = list(smoke_check_cache_key(cfg))
    cached = state.get("smoke_check")
    if cached and cached.get("key") == key and cached.get("passed"):
        return {"status": "ok", "smoke_check": "cached"}
    codec = runner.load_codec(cfg, allow_synthetic=allow_synthetic_basis)
    result = smoke_check(cfg, adapter_obj, codec=codec)
    if not result.passed:
        return {"status": "smoke_check_failed", "reason": result.reason, "resume_command": _continue_command(cfg.config_path)}
    state["smoke_check"] = {"key": key, "passed": True, "reason": None}
    return {"status": "ok", "smoke_check": "ran"}


def _stage_calibrate(cfg: config.PCASpecificPSDConfig, candidate_group_ids: Sequence[str], *, allow_synthetic_basis: bool) -> dict:
    height = cfg.generation.config.height // 8
    width = cfg.generation.config.width // 8
    candidate_set = set(candidate_group_ids)
    codec = runner.load_codec(cfg, allow_synthetic=allow_synthetic_basis)
    calibration_bank = _calibration_bank(cfg, height=height, width=width)
    reference_power = calibration.compute_reference_power(calibration_bank, cfg.psd.num_bins, protocol=cfg.psd.protocol)
    gate_candidates = [calibration.GateCandidate(r_s=g.r_s, beta=g.beta) for g in cfg.psd.gate_candidates]
    group_specs = [
        calibration.GroupCandidateSpec(
            group_id=g.group_id,
            group_indices=tuple(manifests.pc_group_by_id(g.group_id).indices),
            tau_plus_candidates=g.tau_plus_candidates,
            tau_minus_candidates=g.tau_minus_candidates,
            target_rms=g.target_rms,
        )
        for g in cfg.psd.groups
        if g.group_id in candidate_set
    ]
    result = calibration.select_gate_and_taus(
        codec, calibration_bank, reference_power, gate_candidates, group_specs,
        protocol=cfg.psd.protocol, num_bins=cfg.psd.num_bins,
        psd_tolerance=cfg.psd.psd_tolerance, correction_gain_bound=cfg.psd.correction_gain_bound,
        condition_number_threshold=cfg.psd.condition_number_threshold,
    )

    if result.status != "SELECTED":
        path = config.write_calibration_registry(cfg, status="REJECTED_ALL_GATES", gate=None, protocol=cfg.psd.protocol, group_taus={})
        return {
            "status": "REJECTED_ALL_GATES",
            "calibration_result_path": str(path),
            "rejected_gates": [{"gate": {"r_s": g.r_s, "beta": g.beta}, "reason": reason} for g, reason in result.rejected_gates],
        }

    validation_bank = _validation_bank(cfg, height=height, width=width)
    validation_reference_power = calibration.compute_reference_power(validation_bank, cfg.psd.num_bins, protocol=cfg.psd.protocol)
    all_passed = True
    for group_id in result.group_selections:
        for sign in ("plus", "minus"):
            outcome = calibration.validate_on_bank(
                codec, validation_bank, result, group_id, sign, validation_reference_power,
                num_bins=cfg.psd.num_bins, psd_tolerance=cfg.psd.psd_tolerance,
            )
            all_passed = all_passed and outcome.passed

    if all_passed:
        group_taus = {gid: (sel.tau_plus, sel.tau_minus) for gid, sel in result.group_selections.items()}
        status = "SELECTED"
        path = config.write_calibration_registry(
            cfg, status=status,
            gate=config.GateCandidateConfig(r_s=result.gate.r_s, beta=result.gate.beta),
            protocol=result.protocol, group_taus=group_taus,
        )
    else:
        status = "FAIL"
        path = config.write_calibration_registry(cfg, status=status, gate=None, protocol=result.protocol, group_taus={})

    return {"status": status, "calibration_result_path": str(path)}


def _metrics_run_dir_name(full_run_dir: Path) -> str:
    return full_run_dir.name


# ---------------------------------------------------------------------------
# The orchestrator.
# ---------------------------------------------------------------------------


def run_workflow(
    config_path: str | Path,
    *,
    run_id: Optional[str] = None,
    force: bool = False,
    allow_synthetic_basis: bool = False,
    candidates_file: Optional[str] = None,
    probe_group_review_file: Optional[str] = None,
    preview_exclusion_file: Optional[str] = None,
    generation_python: Optional[str] = None,
    metrics_python: Optional[str] = None,
    metrics_device: str = "cuda",
    skip_smoke_check: bool = False,
    dry_run: bool = False,
    adapter: Optional[SDXLTurboAdapterPCA] = None,
    metric_runner: Optional[Any] = None,
) -> dict:
    config_path = str(config_path)
    generation_python = generation_python or sys.executable
    metrics_python = metrics_python or sys.executable

    cfg = config.resolve_config(config_path, "workflow")
    base_run_id = run_id or cfg.run.name

    candidates_path = Path(candidates_file) if candidates_file else _default_path(cfg, "candidates.json")
    # ``--probe-group-review-file``/``--preview-exclusion-file`` name a
    # *directory*, not an exact file path: ``export_probe_group_review``/
    # ``export_preview_exclusion_review`` always write a fixed basename inside
    # whatever ``output_dir`` they're given, so treating the override as an
    # arbitrary full path would let the user's chosen basename silently
    # diverge from the one export actually writes, and resume would then look
    # for a file that was never created.
    probe_review_dir = Path(probe_group_review_file) if probe_group_review_file else Path(cfg.config_path).resolve().parent
    probe_review_path = probe_review_dir / "probe_group_review.csv"
    preview_review_dir = Path(preview_exclusion_file) if preview_exclusion_file else Path(cfg.config_path).resolve().parent
    preview_review_path = preview_review_dir / "preview_exclusion_review.csv"
    approved_conditions_path = _default_path(cfg, "approved_conditions.json")

    if dry_run:
        return _dry_run_report(
            cfg, candidates_path=candidates_path, probe_review_path=probe_review_path,
            preview_review_path=preview_review_path, approved_conditions_path=approved_conditions_path,
        )

    state_path = _default_path(cfg, "workflow_state.json")
    state = _load_state(state_path)

    shared_adapter = adapter

    def get_adapter() -> SDXLTurboAdapterPCA:
        nonlocal shared_adapter
        if shared_adapter is None:
            shared_adapter = SDXLTurboAdapterPCA(cfg.model.as_model_config_dict())
        return shared_adapter

    # -- Stage 1-2: validate + build basis -----------------------------------
    cfg = config.resolve_config(config_path, "build-basis")
    build_basis_result = _stage_build_basis(cfg, config_path, generation_python=generation_python)

    # -- Stage 3: smoke check (pre-probe) ------------------------------------
    cfg = config.resolve_config(config_path, "probe")
    if not skip_smoke_check:
        smoke_result = _run_smoke_check(cfg, state, get_adapter(), allow_synthetic_basis=allow_synthetic_basis)
        _save_state(state_path, state)
        if smoke_result["status"] != "ok":
            return {"status": "smoke_check_failed", "stage": "pre_probe", **smoke_result}

    # -- Stage 4: probe -------------------------------------------------------
    probe_run_id = f"{base_run_id}_probe"
    entries = probing.build_probing_manifest(
        channels=cfg.basis.channels,
        height=cfg.generation.config.height // 8,
        width=cfg.generation.config.width // 8,
        patch_size=cfg.basis.patch_size,
        rho=cfg.probing.rho,
    )
    dispatched = _maybe_subprocess(generation_python, [
        "probe", "--config", config_path, "--run-id", probe_run_id,
        *(["--force"] if force else []),
        *(["--allow-synthetic-basis"] if allow_synthetic_basis else []),
    ])
    if dispatched is not None:
        probe_run_dir = cfg.resolve_root(cfg.run.outputs_root) / probe_run_id
    else:
        probe_run_dir = probing.generate_probes(
            cfg, entries, run_id=probe_run_id, force=force,
            adapter=get_adapter(), allow_synthetic_basis=allow_synthetic_basis,
        )

    # -- Stage 5-6: STOP for review 1, then select candidates ----------------
    probe_ingest = review.ingest_probe_group_review(probe_review_path)
    if probe_ingest.status != "complete":
        export_result = review.export_probe_group_review(
            cfg, probe_run_dir, output_dir=probe_review_dir,
        )
        return {
            "status": "awaiting_review_1",
            "review_status": probe_ingest.status,
            "incomplete_keys": list(probe_ingest.incomplete_keys),
            "csv_path": str(export_result["csv_path"]),
            "contact_sheet_paths": {k: str(v) for k, v in export_result["contact_sheet_paths"].items()},
            "continue_command": _continue_command(config_path),
            "note": GROUP_REVIEW_CAVEAT,
        }
    _check_review_conflict(state, "probe_group_review_hash", probe_review_path, candidates_path)

    select_result = review.select_candidates_from_group_review(probe_ingest.rows)
    write_json(candidates_path, _jsonable(select_result))
    state["probe_group_review_hash"] = file_hash(probe_review_path)
    _save_state(state_path, state)

    if not select_result.candidates:
        return {
            "status": "no_candidates",
            "message": NO_CANDIDATES_MESSAGE,
            "candidates_path": str(candidates_path),
            "reserve": list(select_result.reserve),
        }
    candidate_group_ids = tuple(select_result.candidates)

    # -- Stage 7: calibrate ----------------------------------------------------
    cfg = config.resolve_config(config_path, "calibrate", candidate_group_ids)
    calibrate_result = _stage_calibrate(cfg, candidate_group_ids, allow_synthetic_basis=allow_synthetic_basis)
    if calibrate_result["status"] != "SELECTED":
        return {"status": "calibration_failed", **calibrate_result}

    # -- Stage 8: generate preview ---------------------------------------------
    cfg = config.resolve_config(config_path, "generate-psd", candidate_group_ids)
    preview_run_id = f"{base_run_id}_preview"
    dispatched = _maybe_subprocess(generation_python, [
        "generate-psd", "--config", config_path, "--stage", "preview", "--run-id", preview_run_id,
        "--candidates-file", str(candidates_path),
        *(["--force"] if force else []),
        *(["--allow-synthetic-basis"] if allow_synthetic_basis else []),
    ])
    if dispatched is not None:
        preview_run_dir = cfg.resolve_root(cfg.run.outputs_root) / preview_run_id
    else:
        preview_run_dir = runner.generate_preview(
            cfg, candidate_group_ids, run_id=preview_run_id, force=force,
            adapter=get_adapter(), allow_synthetic_basis=allow_synthetic_basis,
        )

    # -- Stage 9-10: STOP for review 2, then approve conditions ----------------
    preview_ingest = review.ingest_preview_exclusion_review(preview_review_path, candidate_group_ids)
    if preview_ingest.status != "complete":
        export_result = review.export_preview_exclusion_review(
            cfg, preview_run_dir, candidate_group_ids, output_dir=preview_review_dir,
        )
        return {
            "status": "awaiting_review_2",
            "review_status": preview_ingest.status,
            "incomplete_condition_ids": list(preview_ingest.incomplete_condition_ids),
            "csv_path": str(export_result["csv_path"]),
            "contact_sheet_paths": {k: str(v) for k, v in export_result["contact_sheet_paths"].items()},
            "continue_command": _continue_command(config_path),
        }
    _check_review_conflict(state, "preview_exclusion_review_hash", preview_review_path, approved_conditions_path)

    if any(row.condition_id == "reference" and row.excluded for row in preview_ingest.rows):
        raise WorkflowError(
            "the 'reference' condition was marked excluded in the preview-exclusion review -- a broken "
            "reference invalidates every paired comparison; investigate manually rather than proceeding"
        )

    approved = review.approved_condition_ids(preview_ingest.rows)
    write_json(approved_conditions_path, {"approved_condition_ids": list(approved)})
    state["preview_exclusion_review_hash"] = file_hash(preview_review_path)
    _save_state(state_path, state)

    if approved == ("reference",):
        return {
            "status": "no_approved_conditions",
            "message": NO_APPROVED_CONDITIONS_MESSAGE,
            "approved_conditions_path": str(approved_conditions_path),
        }

    # -- Stage 11: smoke check (pre-full-pilot) --------------------------------
    if not skip_smoke_check:
        smoke_result = _run_smoke_check(cfg, state, get_adapter(), allow_synthetic_basis=allow_synthetic_basis)
        _save_state(state_path, state)
        if smoke_result["status"] != "ok":
            return {"status": "smoke_check_failed", "stage": "pre_full_pilot", **smoke_result}

    # -- Stage 12: generate full pilot ------------------------------------------
    full_run_id = f"{base_run_id}_full"
    dispatched = _maybe_subprocess(generation_python, [
        "generate-psd", "--config", config_path, "--stage", "full", "--run-id", full_run_id,
        "--candidates-file", str(candidates_path),
        "--approved-conditions-file", str(approved_conditions_path),
        *(["--force"] if force else []),
        *(["--allow-synthetic-basis"] if allow_synthetic_basis else []),
    ])
    if dispatched is not None:
        full_run_dir = cfg.resolve_root(cfg.run.outputs_root) / full_run_id
    else:
        full_run_dir = runner.generate_full_pilot(
            cfg, candidate_group_ids, run_id=full_run_id, force=force,
            approved_condition_ids=approved, adapter=get_adapter(), allow_synthetic_basis=allow_synthetic_basis,
        )
    executed_image_count = len(approved) * manifests.NUM_PROMPT_BLOCK_PAIRS * manifests.NUM_BASE_INDICES_PER_BLOCK
    ceiling_image_count = manifests.full_pilot_image_count(len(candidate_group_ids))

    # -- Stage 13: metrics -------------------------------------------------------
    cfg = config.resolve_config(config_path, "metrics", candidate_group_ids)
    dispatched = _maybe_subprocess(metrics_python, [
        "metrics", "--config", config_path, "--run-id", _metrics_run_dir_name(full_run_dir),
        "--candidates-file", str(candidates_path),
        "--device", metrics_device,
        *(["--force"] if force else []),
    ])
    if dispatched is None:
        metrics.run_metrics(full_run_dir, metric_runner=metric_runner, device=metrics_device, force=force)

    # -- Stage 14: analyze ---------------------------------------------------------
    cfg = config.resolve_config(config_path, "analyze", candidate_group_ids)
    analysis_result = analysis.analyze_run(full_run_dir)

    return {
        "status": "ok",
        "candidates": list(candidate_group_ids),
        "approved_condition_ids": list(approved),
        "probe_run_dir": str(probe_run_dir),
        "preview_run_dir": str(preview_run_dir),
        "full_run_dir": str(full_run_dir),
        "executed_image_count": executed_image_count,
        "pre_exclusion_image_count_ceiling": ceiling_image_count,
        "condition_summary_count": len(analysis_result["condition_summaries"]),
        "paired_effect_count": len(analysis_result["paired_effects"]),
    }
