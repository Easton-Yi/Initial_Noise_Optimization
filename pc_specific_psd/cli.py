"""``python -m pc_specific_psd <command>`` entry point.

Every command resolves its config centrally through ``config.resolve_config``
(load + per-command tier validation), so a command whose ``needs_calibration``
fields aren't yet resolved refuses with the exact missing-field list rather
than failing deeper inside some other module. ``--dry-run`` reports planned
actions (image counts, required inputs, unresolved fields) without touching
disk, a model, or either compat shim's heavy dependencies -- it never calls
``resolve_config`` at the "full"/"calibration" tiers in a way that would raise,
since the whole point of a dry run is to show *why* a command isn't ready yet.

Human-annotation gate (see docs/PCA-Specific-PSD-Editing-Plan.md): this
module never invents, guesses, or partially auto-fills an annotation value.
Every review-ingesting command takes the completed annotation file as an
explicit ``--review-file``/``--mapping-file`` input and refuses if it is
missing, malformed, or fails schema validation.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from pc_specific_psd import (
    analysis,
    basis as basis_module,
    calibration,
    calibration_v2,
    expected_rms,
    config,
    manifests,
    metrics,
    probing,
    review,
    runner,
    spectral,
    psd_editor,
    workflow,
)
from pc_specific_psd.adapters import SDXLTurboAdapterPCA
from pc_specific_psd.compat_generation import file_hash, read_json, write_json


class CliError(RuntimeError):
    """Raised for CLI-specific refusals (bad flag combinations, missing
    review/mapping files, unknown ids) -- never for algorithmic failures,
    which raise their own module's error type instead.
    """


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


def _load_mapping(path: Path, cls) -> tuple:
    if not path.exists():
        raise CliError(f"mapping file not found: {path}")
    return tuple(cls(**entry) for entry in read_json(path))


def _gallery_row_from_dict(raw: dict) -> "review.GalleryReviewRow":
    ratings = {label: review.GalleryRatings(**value) for label, value in raw["ratings"].items()}
    return review.GalleryReviewRow(
        group_key=raw["group_key"],
        blind_labels=tuple(raw["blind_labels"]),
        ratings=ratings,
        preference=raw["preference"],
        evidence=raw["evidence"],
        confidence=raw["confidence"],
    )


def _load_and_ingest_probe_review(review_file: str | None, mapping_file: str | None):
    if not review_file or not mapping_file:
        raise CliError("this action requires both --review-file and --mapping-file")
    review_path = Path(review_file)
    if not review_path.exists():
        raise CliError(f"review file not found: {review_path}")
    mapping = _load_mapping(Path(mapping_file), review.ProbeReviewMappingEntry)
    raw_rows = read_json(review_path)
    rows = review.ingest_probe_review(raw_rows, expected_pair_ids=tuple(m.pair_id for m in mapping))
    return rows, mapping


def _parse_group_ids(raw: str) -> tuple[str, ...]:
    group_ids = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not group_ids:
        raise CliError("--group-ids must name at least one PC group id")
    for group_id in group_ids:
        try:
            manifests.pc_group_by_id(group_id)
        except KeyError as exc:
            raise CliError(str(exc)) from exc
    return group_ids


def _default_path(cfg: config.PCASpecificPSDConfig, suffix: str) -> Path:
    return Path(cfg.config_path).resolve().parent / f"{cfg.run.name}_{suffix}"


def _calibration_bank(cfg: config.PCASpecificPSDConfig, *, height: int, width: int) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(cfg.psd.calibration_bank_seed)
    return torch.randn((cfg.psd.calibration_bank_size, cfg.basis.channels, height, width), generator=generator, dtype=torch.float32)


def _validation_bank(cfg: config.PCASpecificPSDConfig, *, height: int, width: int) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(cfg.psd.validation_bank_seed)
    return torch.randn((cfg.psd.validation_bank_size, cfg.basis.channels, height, width), generator=generator, dtype=torch.float32)


def _read_candidates(cfg: config.PCASpecificPSDConfig, override: str | None) -> tuple[str, ...]:
    path = Path(override) if override else _default_path(cfg, "candidates.json")
    if not path.exists():
        raise CliError(f"candidates file not found: {path}; run select-candidates first")
    payload = read_json(path)
    return tuple(payload["candidates"])


# ---------------------------------------------------------------------------
# Per-command handlers. Each returns a JSON-serializable dict describing what
# happened (or, for --dry-run, what would happen).
# ---------------------------------------------------------------------------


def _cmd_validate_config(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    return {
        "status": "ok",
        "run_name": cfg.run.name,
        "config_path": str(cfg.config_path),
        "provenance": cfg.provenance(),
    }


def _cmd_build_basis(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    manifest_path = cfg.resolve_root(cfg.basis.dataset_manifest_path)
    output_path = cfg.resolve_root(cfg.basis.basis_output_path)
    if args.dry_run:
        return {
            "status": "dry_run",
            "dataset_manifest_path": str(manifest_path) if manifest_path else None,
            "basis_output_path": str(output_path) if output_path else None,
            "ready": manifest_path is not None,
            "note": (
                "basis.dataset_manifest_path is not set -- real basis construction requires a dataset "
                "manifest path (never auto-downloaded); supply one in the config before running build-basis "
                "for real" if manifest_path is None else "ready to build a real basis from this manifest"
            ),
        }
    if cfg.basis.dataset_manifest_path is None:
        raise CliError(
            "basis.dataset_manifest_path is not set -- real basis construction requires a dataset manifest "
            "path (never auto-downloaded); supply one in the config before running build-basis for real"
        )
    manifest = basis_module.load_dataset_manifest(manifest_path)
    adapter = SDXLTurboAdapterPCA(cfg.model.as_model_config_dict())
    built = basis_module.build_pca_basis(
        manifest, adapter.encode_images_for_basis,
        patch_size=cfg.basis.patch_size, channels=cfg.basis.channels,
        patches_per_image=cfg.basis.patches_per_image, sampling_seed=cfg.basis.sampling_seed,
        synthetic=False,
    )
    stability = basis_module.split_half_stability(
        manifest, adapter.encode_images_for_basis,
        patch_size=cfg.basis.patch_size, channels=cfg.basis.channels,
        patches_per_image=cfg.basis.patches_per_image, sampling_seed=cfg.basis.sampling_seed,
        split_seed=cfg.basis.split_seed, synthetic=False,
        num_leading_components=cfg.basis.num_leading_components,
        bands={group.group_id: tuple(group.indices) for group in manifests.PC_GROUPS},
    )
    basis_module.save_basis(built, output_path)
    if cfg.generation.release_model_after_generation:
        adapter.close()
    return {
        "status": "ok",
        "basis_output_path": str(output_path),
        "num_samples": built.num_samples,
        "orthogonality_error": basis_module.orthogonality_error(built.components),
        "split_half_stability": _jsonable(stability),
    }


def _cmd_inspect_basis(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    path = cfg.resolve_root(cfg.basis.basis_output_path)
    if args.dry_run:
        exists = path is not None and path.exists()
        return {
            "status": "dry_run",
            "basis_output_path": str(path) if path else None,
            "ready": exists,
            "note": "basis file exists and can be inspected" if exists else "basis file does not exist; run build-basis first",
        }
    if path is None or not path.exists():
        raise CliError(f"basis file does not exist: {path}; run build-basis first")
    loaded = basis_module.load_basis(
        path, allow_synthetic=args.allow_synthetic_basis,
        expected_patch_size=cfg.basis.patch_size, expected_channels=cfg.basis.channels,
    )
    centroids = basis_module.basis_frequency_centroids(loaded.components, loaded.patch_size, loaded.channels)
    group_centroids = {
        group.group_id: float(centroids[list(group.indices)].mean())
        for group in manifests.PC_GROUPS
        if group.end_1based <= centroids.shape[0]
    }
    return {
        "status": "ok",
        "metadata": loaded.metadata,
        "num_samples": loaded.num_samples,
        "eigenvalues_leading": loaded.eigenvalues[:10].tolist(),
        "orthogonality_error": basis_module.orthogonality_error(loaded.components),
        "group_frequency_centroids": group_centroids,
    }


def _cmd_basis_stability(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    """Image-disjoint split-half diagnostic; never overwrites the formal basis."""
    output_path = Path(args.output) if args.output else _default_path(cfg, "basis_stability_v2.json")
    manifest_path = cfg.resolve_root(cfg.basis.dataset_manifest_path)
    if args.dry_run:
        return {
            "status": "dry-run",
            "manifest_path": str(manifest_path) if manifest_path else None,
            "output_path": str(output_path),
            "split_seed": cfg.basis.split_seed,
            "note": "fits two diagnostic half-bases only; does not rebuild the formal basis",
        }
    if manifest_path is None or not manifest_path.exists():
        raise CliError(f"dataset manifest not found: {manifest_path}")
    manifest = basis_module.load_dataset_manifest(manifest_path)
    adapter = SDXLTurboAdapterPCA(cfg.model.as_model_config_dict())
    bands = {
        group.group_id: tuple(group.indices)
        for group in manifests.PC_GROUPS
        if group.end_1based <= cfg.basis.channels * cfg.basis.patch_size * cfg.basis.patch_size
    }
    result = basis_module.split_half_stability(
        manifest,
        adapter.encode_images_for_basis,
        patch_size=cfg.basis.patch_size,
        channels=cfg.basis.channels,
        patches_per_image=cfg.basis.patches_per_image,
        sampling_seed=cfg.basis.sampling_seed,
        split_seed=cfg.basis.split_seed,
        synthetic=False,
        num_leading_components=cfg.basis.num_leading_components or 32,
        bands=bands,
    )
    if cfg.generation.release_model_after_generation:
        adapter.close()
    payload = {
        "status": "ok",
        "basis_output_path": str(cfg.resolve_root(cfg.basis.basis_output_path)),
        "formal_basis_was_modified": False,
        "result": _jsonable(result),
    }
    write_json(output_path, payload)
    return {**payload, "output_path": str(output_path)}


def _cmd_probe(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    channels = cfg.basis.channels
    height = cfg.generation.config.height // 8
    width = cfg.generation.config.width // 8
    patch_size = cfg.basis.patch_size
    rho = cfg.probing.rho

    if args.rho020_followup and args.candidate_recheck:
        raise CliError("--rho020-followup and --candidate-recheck are mutually exclusive")

    if args.rho020_followup:
        if args.dry_run and (not args.review_file or not args.mapping_file):
            return {"status": "dry-run", "requires": ["--review-file", "--mapping-file"]}
        rows, mapping = _load_and_ingest_probe_review(args.review_file, args.mapping_file)
        trigger = review.compute_rho020_trigger(rows, mapping)
        if args.dry_run:
            return {
                "status": "dry-run", "permitted": trigger.permitted, "reason": trigger.reason,
                "prompt_count": trigger.prompt_count,
                "prompts_with_clear_change": sorted(trigger.prompts_with_clear_change),
            }
        if not trigger.permitted:
            raise CliError(
                f"rho=0.20 follow-up not permitted: {trigger.reason} "
                f"(prompt_count={trigger.prompt_count}, "
                f"prompts_with_clear_change={sorted(trigger.prompts_with_clear_change)})"
            )
        entries = probing.build_rho020_supplement_manifest(
            trigger, channels=channels, height=height, width=width, patch_size=patch_size,
        )
        run_id = args.run_id or f"{cfg.run.name}_probe_rho020"
    elif args.candidate_recheck:
        if args.dry_run and not args.group_ids:
            return {"status": "dry-run", "requires": ["--group-ids"]}
        if not args.group_ids:
            raise CliError("--candidate-recheck requires --group-ids")
        group_ids = _parse_group_ids(args.group_ids)
        entries = probing.build_candidate_recheck_manifest(
            group_ids, channels=channels, height=height, width=width, patch_size=patch_size, rho=rho,
        )
        run_id = args.run_id or f"{cfg.run.name}_probe_recheck"
        if args.dry_run:
            return {"status": "dry-run", "run_id": run_id, "image_count": len(entries)}
    else:
        entries = probing.build_probing_manifest(
            channels=channels, height=height, width=width, patch_size=patch_size, rho=rho,
        )
        run_id = args.run_id or f"{cfg.run.name}_probe"
        if args.dry_run:
            return {"status": "dry-run", "run_id": run_id, "image_count": len(entries)}

    run_dir = probing.generate_probes(
        cfg, entries, run_id=run_id, force=args.force, allow_synthetic_basis=args.allow_synthetic_basis,
    )
    return {"status": "ok", "run_dir": str(run_dir), "image_count": len(entries)}


def _cmd_export_review(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    entries = probing.build_probing_manifest(
        channels=cfg.basis.channels,
        height=cfg.generation.config.height // 8,
        width=cfg.generation.config.width // 8,
        patch_size=cfg.basis.patch_size,
        rho=cfg.probing.rho,
    )
    review_path = Path(args.review_output) if args.review_output else _default_path(cfg, "probe_review_template.json")
    mapping_path = Path(args.mapping_output) if args.mapping_output else _default_path(cfg, "probe_review_mapping.json")
    if args.dry_run:
        return {"status": "dry-run", "pair_count": len(entries), "review_output": str(review_path), "mapping_output": str(mapping_path)}

    blind_rows, mapping = review.export_probe_review(entries)
    write_json(review_path, list(blind_rows))
    write_json(mapping_path, _jsonable(mapping))
    return {"status": "ok", "review_template_path": str(review_path), "mapping_path": str(mapping_path), "pair_count": len(blind_rows)}


def _cmd_select_candidates(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    output_path = Path(args.output) if args.output else _default_path(cfg, "candidates.json")
    if args.dry_run:
        return {"status": "dry-run", "output": str(output_path), "requires": ["--review-file", "--mapping-file"]}

    rows, mapping = _load_and_ingest_probe_review(args.review_file, args.mapping_file)
    result = review.select_candidates(rows, mapping)
    write_json(output_path, _jsonable(result))
    return {"status": "ok", "candidates": list(result.candidates), "reserve": list(result.reserve), "output_path": str(output_path)}


def _cmd_calibrate(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    if cfg.psd.calibration_profile == config.EXPECTED_RMS_CALIBRATION_PROFILE:
        return _cmd_calibrate_expected_rms(cfg, args)
    if cfg.psd.calibration_profile == config.EFFECT_CALIBRATION_PROFILE:
        return _cmd_calibrate_effect_v2(cfg, args)
    height = cfg.generation.config.height // 8
    width = cfg.generation.config.width // 8
    candidate_group_ids = set(_read_candidates(cfg, args.candidates_file)) if args.candidates_file else None
    if args.dry_run:
        return {
            "status": "dry-run",
            "gate_candidate_count": len(cfg.psd.gate_candidates),
            "group_ids": [
                group.group_id for group in cfg.psd.groups
                if candidate_group_ids is None or group.group_id in candidate_group_ids
            ],
        }

    codec = runner.load_codec(cfg, allow_synthetic=args.allow_synthetic_basis)
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
        if candidate_group_ids is None or g.group_id in candidate_group_ids
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
    validation_report = {}
    all_passed = True
    for group_id in result.group_selections:
        for sign in ("plus", "minus"):
            outcome = calibration.validate_on_bank(
                codec, validation_bank, result, group_id, sign, validation_reference_power,
                num_bins=cfg.psd.num_bins, psd_tolerance=cfg.psd.psd_tolerance,
            )
            validation_report[f"{group_id}_{sign}"] = {"passed": outcome.passed, "reason": outcome.reason}
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

    return {
        "status": status,
        "calibration_result_path": str(path),
        "gate": {"r_s": result.gate.r_s, "beta": result.gate.beta},
        "protocol": result.protocol,
        "group_taus": {gid: {"tau_plus": sel.tau_plus, "tau_minus": sel.tau_minus} for gid, sel in result.group_selections.items()},
        "validation": validation_report,
    }


def _cmd_calibrate_effect_v2(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    """Fit corrections on one bank and select independent tau values by final effect."""
    height = cfg.generation.config.height // 8
    width = cfg.generation.config.width // 8
    if args.dry_run:
        return {
            "status": "dry-run",
            "calibration_profile": config.EFFECT_CALIBRATION_PROFILE,
            "fixed_gate": _jsonable(cfg.psd.gate_candidates[0]),
            "groups": [group.group_id for group in cfg.psd.groups],
            "targets": {group.group_id: list(group.target_relative_l2) for group in cfg.psd.groups},
        }
    codec = runner.load_codec(cfg, allow_synthetic=args.allow_synthetic_basis)
    bank = _calibration_bank(cfg, height=height, width=width)
    reference_power = calibration.compute_reference_power(bank, cfg.psd.num_bins, protocol=cfg.psd.protocol)
    declared_gate = cfg.psd.gate_candidates[0]
    gate = calibration.GateCandidate(r_s=declared_gate.r_s, beta=declared_gate.beta)
    specs = [
        calibration_v2.EffectGroupSpec(
            group_id=group.group_id,
            group_indices=tuple(manifests.pc_group_by_id(group.group_id).indices),
            tau_plus_candidates=group.tau_plus_candidates,
            tau_minus_candidates=group.tau_minus_candidates,
            target_relative_l2=group.target_relative_l2,
            effect_target_tolerance=group.effect_target_tolerance,
        )
        for group in cfg.psd.groups
    ]
    result = calibration_v2.select_effect_size_targets(
        codec, bank, reference_power, gate, specs,
        protocol=cfg.psd.protocol, num_bins=cfg.psd.num_bins,
        psd_tolerance=cfg.psd.psd_tolerance,
        correction_gain_bound=cfg.psd.correction_gain_bound,
        condition_number_threshold=cfg.psd.condition_number_threshold,
        minimum_covariance_distance=cfg.psd.minimum_covariance_distance,
    )
    conditions = _effect_condition_selections(result)
    payload = {
        "status": result.status,
        "protocol": result.protocol,
        "gate": _jsonable(result.gate),
        "bank": {"role": "calibration", "seed": cfg.psd.calibration_bank_seed, "size": cfg.psd.calibration_bank_size},
        "basis_hash": runner.basis_hash_for(cfg),
        "config_hash": file_hash(cfg.config_path),
        "config_path": str(cfg.config_path),
        "selections": _jsonable(result.selections),
        "candidate_diagnostics": _jsonable(result.candidate_diagnostics),
        "tau_zero_diagnostics": _jsonable(result.tau_zero_diagnostics),
        "condition_selections": conditions,
        "all_targets_reached": result.all_targets_reached,
        "unreachable_target_count": result.unreachable_target_count,
    }
    path = config.write_effect_calibration_registry(cfg, payload)
    return {
        "status": result.status,
        "calibration_profile": result.profile,
        "calibration_result_path": str(path),
        "condition_ids": sorted(conditions),
        "selections": _jsonable(result.selections),
    }


def _effect_condition_selections(result: calibration_v2.EffectCalibrationResult) -> dict[str, dict[str, Any]]:
    """Freeze only reachable selections, deduplicating targets that share tau."""
    conditions: dict[str, dict[str, Any]] = {}
    for selection in result.selections:
        if selection.status != "SELECTED" or selection.evaluation is None or selection.condition_id is None:
            continue
        diagnostic = selection.evaluation
        condition = conditions.setdefault(selection.condition_id, {
            "condition_id": selection.condition_id,
            "group_id": selection.group_id,
            "sign": selection.sign,
            "tau": selection.selected_tau,
            "actual_relative_l2": selection.actual_relative_l2,
            "target_relative_l2": [],
            "correction": _jsonable(diagnostic.correction),
            "diagnostics": _jsonable(diagnostic),
        })
        condition["target_relative_l2"].append(selection.target_relative_l2)
    return conditions


def _cmd_calibrate_expected_rms(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    """Freeze deterministic expected-energy scales and matched controls."""
    height = cfg.generation.config.height // 8
    width = cfg.generation.config.width // 8
    if args.dry_run:
        return {
            "status": "dry-run",
            "calibration_profile": expected_rms.CALIBRATION_PROFILE,
            "energy_constraint": expected_rms.ENERGY_CONSTRAINT,
            "fixed_gate": _jsonable(cfg.psd.gate_candidates[0]),
            "groups": [group.group_id for group in cfg.psd.groups],
            "targets": {group.group_id: list(group.target_relative_l2) for group in cfg.psd.groups},
            "processing_batch_size": cfg.psd.processing_batch_size,
            "note": "CPU-only deterministic transfer response plus calibration-bank diagnostics",
        }
    codec = runner.load_codec(cfg, allow_synthetic=args.allow_synthetic_basis)
    bank = _calibration_bank(cfg, height=height, width=width)
    gate = cfg.psd.gate_candidates[0]
    results = [
        expected_rms.calibrate(
            codec, bank,
            group_id=group.group_id,
            group_indices=tuple(manifests.pc_group_by_id(group.group_id).indices),
            tau_plus_candidates=group.tau_plus_candidates,
            tau_minus_candidates=group.tau_minus_candidates,
            targets=group.target_relative_l2,
            gate_r_s=gate.r_s,
            gate_beta=gate.beta,
            num_bins=cfg.psd.num_bins,
            batch_size=cfg.psd.processing_batch_size,
            injection_dtype=cfg.model.dtype,
            condition_number_threshold=cfg.psd.condition_number_threshold,
            minimum_covariance_distance=cfg.psd.minimum_covariance_distance,
            configured_tolerance=group.effect_target_tolerance,
        )
        for group in cfg.psd.groups
    ]
    operator_objects = tuple(operator for result in results for operator in result.operators)
    try:
        pairs = expected_rms.validate_operator_pairs(operator_objects) if operator_objects else ()
    except ValueError as exc:
        raise CliError(f"calibration produced an invalid candidate/control registry: {exc}") from exc
    frozen = {
        operator.condition_id: expected_rms.operator_to_payload(operator)
        for operator in operator_objects
    }
    candidates = [expected_rms.operator_to_payload(pair.candidate) for pair in pairs]
    controls = [expected_rms.operator_to_payload(pair.control) for pair in pairs]
    pair_condition_ids = [
        [pair.candidate.condition_id, pair.control.condition_id]
        for pair in pairs
    ]
    exclusions = [item for result in results for item in _jsonable(result.exclusions)]
    status = "SELECTED" if pairs and all(result.status == "SELECTED" for result in results) else "FAIL"
    _, bin_index = spectral.radial_bin_index(height, width, cfg.psd.num_bins, rfft=True)
    counts = torch.zeros(cfg.psd.num_bins, dtype=torch.float64).scatter_add_(
        0, bin_index.reshape(-1),
        spectral.rfft_conjugate_weights(width).view(1, -1).expand(height, -1).reshape(-1),
    )
    payload = {
        "status": status,
        "protocol": cfg.psd.protocol,
        "operator_schema_version": expected_rms.OPERATOR_SCHEMA_VERSION,
        "diagnostic_version": expected_rms.DIAGNOSTIC_VERSION,
        "binning": {
            **spectral.radial_binning_metadata(height, width, cfg.psd.num_bins),
            "counts": counts.tolist(),
            "count_sum": float(counts.sum()),
        },
        "gate": _jsonable(gate),
        "latent_shape": [cfg.basis.channels, height, width],
        "operator_dtype": "float32",
        "injection_dtype": cfg.model.dtype,
        "bank": {
            "role": "calibration", "seed": cfg.psd.calibration_bank_seed,
            "size": cfg.psd.calibration_bank_size,
            "processing_batch_size": cfg.psd.processing_batch_size,
        },
        "basis_hash": runner.basis_hash_for(cfg),
        "config_hash": file_hash(cfg.config_path),
        "config_path": str(cfg.config_path),
        "pc_groups": {
            group.group_id: list(manifests.pc_group_by_id(group.group_id).indices)
            for group in cfg.psd.groups
        },
        "reference": {
            "same_phase_alpha": cfg.frozen.same_phase_alpha,
            "same_phase_gamma": cfg.frozen.same_phase_gamma,
            "reference_scale_profile": cfg.frozen.reference_scale_profile,
        },
        "selections": [item for result in results for item in _jsonable(result.selections)],
        "attempted_taus": {group.group_id: _jsonable(result.attempted_taus) for group, result in zip(cfg.psd.groups, results)},
        "operator_diagnostics": [item for result in results for item in _jsonable(result.diagnostics)],
        "frozen_operators": frozen,
        "valid_operator_pairs": pair_condition_ids,
        "condition_exclusions": exclusions,
        "all_targets_reached": all(result.all_targets_reached for result in results),
        "unreachable_target_count": sum(result.unreachable_target_count for result in results),
        "code_source": {
            "module": "pc_specific_psd.expected_rms",
            "operator_schema_version": expected_rms.OPERATOR_SCHEMA_VERSION,
            "files": {
                "expected_rms.py": file_hash(Path(expected_rms.__file__)),
                "psd_editor.py": file_hash(Path(psd_editor.__file__)),
                "spectral.py": file_hash(Path(spectral.__file__)),
                "runner.py": file_hash(Path(runner.__file__)),
            },
        },
    }
    path = config.write_expected_rms_calibration_registry(cfg, payload)
    return {
        "status": status,
        "calibration_profile": expected_rms.CALIBRATION_PROFILE,
        "calibration_result_path": str(path),
        "candidate_condition_ids": sorted(raw["condition_id"] for raw in candidates),
        "control_condition_ids": sorted(raw["condition_id"] for raw in controls),
        "valid_operator_pairs": pair_condition_ids,
        "condition_exclusions": exclusions,
        "selections": payload["selections"],
    }


def _cmd_validate_noise_expected_rms(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    """Validate frozen candidate/control operators on the independent bank."""
    output_path = cfg.resolve_root(cfg.psd.validation_result_path)
    if args.dry_run:
        return {
            "status": "dry-run", "output_path": str(output_path),
            "bank_seed": cfg.psd.validation_bank_seed, "bank_size": cfg.psd.validation_bank_size,
            "processing_batch_size": cfg.psd.processing_batch_size,
            "note": "CPU-only; no model load and no operator refitting",
        }
    registry_path = cfg.resolve_root(cfg.psd.calibration_result_path)
    registry = read_json(registry_path)
    current_config_hash = file_hash(cfg.config_path)
    if registry.get("config_hash") != current_config_hash:
        raise CliError("expected_rms_v1 calibration registry is stale relative to the current config; run calibrate again")
    codec = runner.load_codec(cfg, allow_synthetic=args.allow_synthetic_basis)
    current_basis_hash = runner.basis_hash_for(cfg)
    if registry.get("basis_hash") != current_basis_hash:
        raise CliError("expected_rms_v1 calibration registry is stale relative to the formal basis")
    operators = runner.load_expected_rms_operators(cfg)
    pairs = expected_rms.validate_operator_pairs(tuple(operators.values()))
    height = cfg.generation.config.height // 8
    width = cfg.generation.config.width // 8
    bank = _validation_bank(cfg, height=height, width=width)
    reference_response = spectral.linear_operator_frequency_response(
        psd_editor.apply_psd_edit_tau_zero, cfg.basis.channels, height, width
    )
    reference_energy = spectral.expected_mean_square_from_frequency_response(reference_response, width=width)
    reference = expected_rms.reference_operator(cfg.basis.channels, height, width, cfg.psd.num_bins)
    reference_construction = expected_rms.OperatorConstruction(
        operator=reference,
        response=reference_response,
        radial_psd=spectral.expected_radial_psd_from_frequency_response(
            reference_response, width=width, num_bins=cfg.psd.num_bins
        ),
    )
    reference_diagnostic = expected_rms.diagnose_operator(
        codec, bank, reference_construction, reference_response,
        batch_size=cfg.psd.processing_batch_size, injection_dtype=cfg.model.dtype,
        condition_number_threshold=cfg.psd.condition_number_threshold,
        minimum_covariance_distance=cfg.psd.minimum_covariance_distance,
    )
    diagnostics: dict[str, Any] = {}
    constructions: dict[str, expected_rms.OperatorConstruction] = {}
    for condition_id, operator in operators.items():
        construction = expected_rms.construction_for_frozen_operator(codec, operator)
        constructions[condition_id] = construction
        diagnostic = expected_rms.diagnose_operator(
            codec, bank, construction, reference_response,
            batch_size=cfg.psd.processing_batch_size,
            injection_dtype=cfg.model.dtype,
            condition_number_threshold=cfg.psd.condition_number_threshold,
            minimum_covariance_distance=cfg.psd.minimum_covariance_distance,
        )
        energy_relative_error = abs(construction.radial_psd.total_expected_mean_square - reference_energy) / reference_energy
        group = next(group for group in cfg.psd.groups if group.group_id == operator.group_id)
        target_checks = [
            {
                "target": target,
                "tolerance": calibration_v2.target_tolerance(target, group.effect_target_tolerance),
                "target_within_tolerance": abs(diagnostic.paired_relative_l2_final_fp32 - target)
                <= calibration_v2.target_tolerance(target, group.effect_target_tolerance),
            }
            for target in operator.target_relative_l2
        ] if operator.operator_type == "pca_candidate" else []
        numerically_valid = diagnostic.status == "accepted" and energy_relative_error <= 1e-5
        diagnostics[condition_id] = {
            "numerically_valid": numerically_valid,
            "operator_hash": expected_rms.operator_hash(operator),
            "actual_relative_l2": diagnostic.paired_relative_l2_final_fp32,
            "target_checks": target_checks,
            "expected_energy_relative_error": energy_relative_error,
            "diagnostics": _jsonable(diagnostic),
        }
    for condition_id, operator in operators.items():
        if operator.operator_type != "fourier_control":
            continue
        candidate = constructions[operator.candidate_id]
        control = constructions[condition_id]
        nonempty = candidate.radial_psd.counts > 0
        denominator = candidate.radial_psd.power[nonempty].abs().clamp(min=1e-30)
        match_error = float(((control.radial_psd.power[nonempty] - candidate.radial_psd.power[nonempty]).abs() / denominator).max())
        diagnostics[condition_id]["candidate_annular_psd_match_max_relative_error"] = match_error
        diagnostics[condition_id]["numerically_valid"] = diagnostics[condition_id]["numerically_valid"] and match_error <= expected_rms.DEFAULT_ANNULAR_MATCH_TOLERANCE
    valid_condition_ids: list[str] = []
    valid_operator_pairs: list[list[str]] = []
    exclusions: list[dict[str, Any]] = []

    def own_failure_reason(condition_id: str) -> str:
        row = diagnostics[condition_id]
        diagnostic = row["diagnostics"]
        if diagnostic["status"] != "accepted":
            return diagnostic["failure_reason"] or "operator_numerical_validation_failed"
        if row["expected_energy_relative_error"] > 1e-5:
            return (
                "expected_energy_relative_error_exceeded: "
                f"{row['expected_energy_relative_error']:.6g}"
            )
        match_error = row.get("candidate_annular_psd_match_max_relative_error")
        if match_error is not None and match_error > expected_rms.DEFAULT_ANNULAR_MATCH_TOLERANCE:
            return f"candidate_annular_psd_match_failed: {match_error:.6g}"
        return "operator_numerical_validation_failed"

    for pair in pairs:
        candidate_id = pair.candidate.condition_id
        control_id = pair.control.condition_id
        candidate_ok = diagnostics[candidate_id]["numerically_valid"]
        control_ok = diagnostics[control_id]["numerically_valid"]
        if candidate_ok and control_ok:
            valid_operator_pairs.append([candidate_id, control_id])
            valid_condition_ids.extend((candidate_id, control_id))
            diagnostics[candidate_id]["preview_eligible"] = True
            diagnostics[control_id]["preview_eligible"] = True
            continue
        candidate_reason = (
            own_failure_reason(candidate_id)
            if not candidate_ok
            else f"paired control {control_id!r} failed numerical validation"
        )
        control_reason = (
            own_failure_reason(control_id)
            if not control_ok
            else f"paired candidate {candidate_id!r} failed numerical validation"
        )
        diagnostics[candidate_id]["preview_eligible"] = False
        diagnostics[candidate_id]["pair_exclusion_reason"] = candidate_reason
        diagnostics[control_id]["preview_eligible"] = False
        diagnostics[control_id]["pair_exclusion_reason"] = control_reason
        exclusions.extend((
            {
                "condition_id": candidate_id,
                "paired_condition_id": control_id,
                "stage": "validation",
                "reason": candidate_reason,
            },
            {
                "condition_id": control_id,
                "paired_condition_id": candidate_id,
                "stage": "validation",
                "reason": control_reason,
            },
        ))
    validation = {
        "status": "PASS" if valid_operator_pairs else "FAIL",
        "calibration_profile": expected_rms.CALIBRATION_PROFILE,
        "energy_constraint": expected_rms.ENERGY_CONSTRAINT,
        "diagnostic_version": expected_rms.DIAGNOSTIC_VERSION,
        "diagnostic_versions": spectral.diagnostic_definition_versions(),
        "bank": {
            "role": "validation", "seed": cfg.psd.validation_bank_seed,
            "size": cfg.psd.validation_bank_size,
            "processing_batch_size": cfg.psd.processing_batch_size,
        },
        "calibration_result_path": str(registry_path),
        "config_hash": current_config_hash,
        "basis_hash": current_basis_hash,
        "calibration_hash": file_hash(registry_path),
        "reference_diagnostics": _jsonable(reference_diagnostic),
        "valid_operator_pairs": valid_operator_pairs,
        "preview_condition_ids": valid_condition_ids,
        "condition_exclusions": exclusions,
        "conditions": diagnostics,
    }
    write_json(output_path, validation)
    return {**validation, "output_path": str(output_path)}


def _cmd_validate_noise(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    """Cache-only replay-and-check: reloads the frozen calibration registry
    and re-runs ``calibration.evaluate_candidate`` against a freshly
    reconstructed calibration bank for every declared group/sign, confirming
    the frozen correction still reproduces an accepted candidate -- no model
    load, no new generation, mirroring ``noise_init/noise_validation.py``'s
    standalone pattern.
    """
    if cfg.psd.calibration_profile == config.EXPECTED_RMS_CALIBRATION_PROFILE:
        return _cmd_validate_noise_expected_rms(cfg, args)
    if cfg.psd.calibration_profile == config.EFFECT_CALIBRATION_PROFILE:
        return _cmd_validate_noise_effect_v2(cfg, args)
    height = cfg.generation.config.height // 8
    width = cfg.generation.config.width // 8
    if args.dry_run:
        return {"status": "dry-run", "note": "replays the frozen calibration registry against a reconstructed calibration bank"}

    codec = runner.load_codec(cfg, allow_synthetic=args.allow_synthetic_basis)
    try:
        corrections = runner.build_frozen_corrections(cfg, codec, height=height, width=width)
    except runner.RunnerError as exc:
        return {"status": "FAIL", "reason": str(exc)}

    return {
        "status": "ok",
        "checked": [f"{group_id}_{sign}" for (group_id, sign) in corrections],
    }


def _cmd_validate_noise_effect_v2(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    """Evaluate the frozen v2 operators once on the disjoint validation bank."""
    output_path = cfg.resolve_root(cfg.psd.validation_result_path)
    if args.dry_run:
        return {
            "status": "dry-run", "output_path": str(output_path),
            "bank_seed": cfg.psd.validation_bank_seed, "bank_size": cfg.psd.validation_bank_size,
        }
    registry_path = cfg.resolve_root(cfg.psd.calibration_result_path)
    payload = read_json(registry_path)
    if payload.get("status") != "SELECTED" or payload.get("calibration_profile") != config.EFFECT_CALIBRATION_PROFILE:
        raise CliError("effect_size_v2 calibration registry is not SELECTED")
    current_config_hash = file_hash(cfg.config_path)
    if payload.get("config_hash") != current_config_hash:
        raise CliError("effect_size_v2 calibration registry is stale relative to the current config; run calibrate again")
    codec = runner.load_codec(cfg, allow_synthetic=args.allow_synthetic_basis)
    current_basis_hash = runner.basis_hash_for(cfg)
    if payload.get("basis_hash") != current_basis_hash:
        raise CliError("effect_size_v2 calibration registry is stale relative to the formal basis")
    height = cfg.generation.config.height // 8
    width = cfg.generation.config.width // 8
    bank = _validation_bank(cfg, height=height, width=width)
    gate = calibration.GateCandidate(**payload["gate"])
    results: dict[str, Any] = {}
    all_passed = True
    for condition_id, frozen in payload["condition_selections"].items():
        diagnostic = calibration_v2.diagnose_fixed_operator(
            codec, bank, frozen["group_id"], tuple(manifests.pc_group_by_id(frozen["group_id"]).indices),
            gate, float(frozen["tau"]), torch.tensor(frozen["correction"], dtype=torch.float64),
            num_bins=cfg.psd.num_bins, psd_tolerance=cfg.psd.psd_tolerance,
            correction_gain_bound=cfg.psd.correction_gain_bound,
            condition_number_threshold=cfg.psd.condition_number_threshold,
            minimum_covariance_distance=cfg.psd.minimum_covariance_distance,
        )
        target_checks = [
            {
                "target": float(target),
                "tolerance": calibration_v2.target_tolerance(float(target), next(
                    group.effect_target_tolerance for group in cfg.psd.groups if group.group_id == frozen["group_id"]
                )),
                "passed": abs(diagnostic.paired_relative_l2_final - float(target)) <= calibration_v2.target_tolerance(
                    float(target), next(group.effect_target_tolerance for group in cfg.psd.groups if group.group_id == frozen["group_id"])
                ),
            }
            for target in frozen["target_relative_l2"]
        ]
        passed = diagnostic.status == "accepted" and all(check["passed"] for check in target_checks)
        all_passed = all_passed and passed
        results[condition_id] = {"passed": passed, "target_checks": target_checks, "diagnostics": _jsonable(diagnostic)}
    validation = {
        "status": "PASS" if all_passed and results else "FAIL",
        "calibration_profile": config.EFFECT_CALIBRATION_PROFILE,
        "diagnostic_versions": spectral.diagnostic_definition_versions(),
        "bank": {"role": "validation", "seed": cfg.psd.validation_bank_seed, "size": cfg.psd.validation_bank_size},
        "calibration_result_path": str(registry_path),
        "config_hash": current_config_hash,
        "basis_hash": current_basis_hash,
        "calibration_hash": file_hash(registry_path),
        "conditions": results,
    }
    write_json(output_path, validation)
    return {**validation, "output_path": str(output_path)}


def _cmd_generate_psd(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    if cfg.psd.calibration_profile == config.EXPECTED_RMS_CALIBRATION_PROFILE:
        if args.stage != "preview":
            raise CliError(
                "expected_rms_v1 is limited to the development preview; freeze confirmatory "
                "metrics and success criteria before enabling a full run"
            )
        registry = config.load_calibration_registry(cfg)
        validation_path = cfg.resolve_root(cfg.psd.validation_result_path)
        validation = read_json(validation_path) if validation_path is not None and validation_path.exists() else {}
        condition_ids = tuple(validation.get("preview_condition_ids", registry.frozen_operators))
        entries = manifests.build_expected_rms_preview_manifest_entries(condition_ids)
        if args.dry_run:
            return {
                "status": "dry-run", "stage": "preview",
                "condition_ids": list(condition_ids), "image_count": len(entries),
                "maximum_image_count": 108,
            }
        run_dir = runner.generate_expected_rms_preview(
            cfg, condition_ids, run_id=args.run_id, force=args.force,
            allow_synthetic_basis=args.allow_synthetic_basis,
        )
        return {"status": "ok", "stage": "preview", "run_dir": str(run_dir), "image_count": len(entries)}
    if cfg.psd.calibration_profile == config.EFFECT_CALIBRATION_PROFILE:
        if args.stage != "preview":
            raise CliError(
                "effect_size_v2 is limited to the development preview; freeze confirmatory primary "
                "metrics and success criteria before enabling a full run"
            )
        registry = config.load_calibration_registry(cfg)
        condition_ids = tuple(registry.condition_selections)
        entries = manifests.build_effect_preview_manifest_entries(condition_ids)
        if args.dry_run:
            return {
                "status": "dry-run", "stage": "preview", "condition_ids": list(condition_ids),
                "image_count": len(entries), "maximum_image_count": 60,
            }
        run_dir = runner.generate_effect_preview(
            cfg, condition_ids, run_id=args.run_id, force=args.force,
            allow_synthetic_basis=args.allow_synthetic_basis,
        )
        return {"status": "ok", "stage": "preview", "run_dir": str(run_dir), "image_count": len(entries)}
    if args.dry_run:
        if args.stage == "full" and args.approved_conditions_file:
            approved_path = Path(args.approved_conditions_file)
            if not approved_path.exists():
                return {"status": "dry-run", "stage": args.stage, "note": f"approved-conditions file not found: {approved_path}"}
            approved_condition_ids = tuple(read_json(approved_path)["approved_condition_ids"])
            entries = manifests.build_full_pilot_manifest_entries_for_conditions(approved_condition_ids)
            return {"status": "dry-run", "stage": args.stage, "approved_conditions": list(approved_condition_ids), "image_count": len(entries)}
        try:
            candidate_group_ids = _read_candidates(cfg, args.candidates_file)
        except CliError:
            return {"status": "dry-run", "stage": args.stage, "note": "no candidates file yet; run select-candidates first"}
        image_count = (
            manifests.preview_image_count(len(candidate_group_ids)) if args.stage == "preview"
            else manifests.full_pilot_image_count(len(candidate_group_ids))
        )
        return {"status": "dry-run", "stage": args.stage, "candidates": list(candidate_group_ids), "image_count": image_count}

    candidate_group_ids = _read_candidates(cfg, args.candidates_file)
    if args.stage == "preview":
        run_dir = runner.generate_preview(
            cfg, candidate_group_ids, run_id=args.run_id, force=args.force,
            allow_synthetic_basis=args.allow_synthetic_basis,
        )
    else:
        approved_condition_ids = None
        if args.approved_conditions_file:
            approved_path = Path(args.approved_conditions_file)
            if not approved_path.exists():
                raise CliError(f"approved-conditions file not found: {approved_path}")
            approved_condition_ids = tuple(read_json(approved_path)["approved_condition_ids"])
        run_dir = runner.generate_full_pilot(
            cfg, candidate_group_ids, run_id=args.run_id, force=args.force,
            approved_condition_ids=approved_condition_ids, allow_synthetic_basis=args.allow_synthetic_basis,
        )

    if run_dir is None:
        return {"status": "ok", "stage": args.stage, "run_dir": None, "note": "no candidates -- nothing to generate"}
    return {"status": "ok", "stage": args.stage, "run_dir": str(run_dir)}


def _cmd_export_preview_review(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    review_path = Path(args.review_output) if args.review_output else _default_path(cfg, "preview_review_template.json")
    mapping_path = Path(args.mapping_output) if args.mapping_output else _default_path(cfg, "preview_review_mapping.json")
    if cfg.psd.calibration_profile == config.EXPECTED_RMS_CALIBRATION_PROFILE:
        validation_path = cfg.resolve_root(cfg.psd.validation_result_path)
        validation = read_json(validation_path) if validation_path is not None and validation_path.exists() else {}
        condition_ids = tuple(validation.get("preview_condition_ids", config.load_calibration_registry(cfg).frozen_operators))
        entries = manifests.build_expected_rms_preview_manifest_entries(condition_ids)
        grid_path = review_path.with_name(f"{cfg.run.name}_preview_grid.png")
        if args.dry_run:
            return {"status": "dry-run", "image_count": len(entries), "grid_path": str(grid_path)}
        blind_rows, mapping = review.export_expected_rms_preview_review(entries)
        write_json(review_path, list(blind_rows))
        write_json(mapping_path, _jsonable(mapping))
        run_dir = cfg.resolve_root(cfg.run.outputs_root) / (args.run_id or f"{cfg.run.name}_preview")
        registry_payload = read_json(cfg.resolve_root(cfg.psd.calibration_result_path))
        labels = {}
        for condition_id in condition_ids:
            operator = registry_payload["frozen_operators"][condition_id]
            actual = validation.get("conditions", {}).get(condition_id, {}).get("actual_relative_l2")
            detail = (
                f"{operator['operator_type']} tau={operator['tau']:.6g} "
                f"cal={operator.get('calibration_relative_l2')} val={actual}"
            )
            if operator["operator_type"] == "pca_candidate":
                detail += f" scale={operator['scale']:.6g}"
            else:
                detail += f" matched={operator['candidate_id']}"
            labels[condition_id] = f"{condition_id}\n{detail}"
        review.export_effect_preview_grid(run_dir, condition_ids, grid_path, condition_labels=labels)
        return {
            "status": "ok", "review_template_path": str(review_path),
            "mapping_path": str(mapping_path), "grid_path": str(grid_path),
            "image_count": len(blind_rows),
        }
    if cfg.psd.calibration_profile == config.EFFECT_CALIBRATION_PROFILE:
        condition_ids = tuple(config.load_calibration_registry(cfg).condition_selections)
        entries = manifests.build_effect_preview_manifest_entries(condition_ids)
        grid_path = review_path.with_name(f"{cfg.run.name}_preview_grid.png")
        if args.dry_run:
            return {"status": "dry-run", "image_count": len(entries), "grid_path": str(grid_path)}
        blind_rows, mapping = review.export_preview_review(entries)
        write_json(review_path, list(blind_rows))
        write_json(mapping_path, _jsonable(mapping))
        run_dir = cfg.resolve_root(cfg.run.outputs_root) / (args.run_id or f"{cfg.run.name}_preview")
        review.export_effect_preview_grid(run_dir, condition_ids, grid_path)
        return {
            "status": "ok", "review_template_path": str(review_path),
            "mapping_path": str(mapping_path), "grid_path": str(grid_path),
            "image_count": len(blind_rows),
        }
    if args.dry_run:
        try:
            candidate_group_ids = _read_candidates(cfg, args.candidates_file)
        except CliError:
            return {"status": "dry-run", "note": "no candidates file yet; run select-candidates first"}
        entries = manifests.build_preview_manifest_entries(candidate_group_ids)
        return {"status": "dry-run", "image_count": len(entries)}

    candidate_group_ids = _read_candidates(cfg, args.candidates_file)
    entries = manifests.build_preview_manifest_entries(candidate_group_ids)
    blind_rows, mapping = review.export_preview_review(entries)
    write_json(review_path, list(blind_rows))
    write_json(mapping_path, _jsonable(mapping))
    return {"status": "ok", "review_template_path": str(review_path), "mapping_path": str(mapping_path), "image_count": len(blind_rows)}


def _cmd_ingest_preview_review(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    output_path = Path(args.output) if args.output else _default_path(cfg, "preview_review_ingested.json")
    if args.dry_run:
        return {"status": "dry-run", "output": str(output_path)}

    if not args.review_file or not args.mapping_file:
        raise CliError("ingest-preview-review requires --review-file and --mapping-file")
    mapping = _load_mapping(Path(args.mapping_file), review.PreviewReviewMappingEntry)
    raw_rows = read_json(Path(args.review_file))
    ingest = (
        review.ingest_expected_rms_preview_review
        if cfg.psd.calibration_profile == config.EXPECTED_RMS_CALIBRATION_PROFILE
        else review.ingest_preview_review
    )
    rows = ingest(raw_rows, expected_image_ids=tuple(m.image_id for m in mapping))
    write_json(output_path, _jsonable(rows))
    return {"status": "ok", "output_path": str(output_path), "row_count": len(rows)}


def _cmd_export_gallery_review(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    review_path = Path(args.review_output) if args.review_output else _default_path(cfg, "gallery_review_template.json")
    mapping_path = Path(args.mapping_output) if args.mapping_output else _default_path(cfg, "gallery_review_mapping.json")
    if args.dry_run:
        try:
            candidate_group_ids = _read_candidates(cfg, args.candidates_file)
        except CliError:
            return {"status": "dry-run", "note": "no candidates file yet; run select-candidates first"}
        entries = manifests.build_full_pilot_manifest_entries(candidate_group_ids)
        return {"status": "dry-run", "image_count": len(entries)}

    candidate_group_ids = _read_candidates(cfg, args.candidates_file)
    entries = manifests.build_full_pilot_manifest_entries(candidate_group_ids)
    blind_rows, mapping = review.export_gallery_review(entries)
    write_json(review_path, list(blind_rows))
    write_json(mapping_path, _jsonable(mapping))
    return {"status": "ok", "review_template_path": str(review_path), "mapping_path": str(mapping_path), "gallery_count": len(blind_rows)}


def _cmd_ingest_gallery_review(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    output_path = Path(args.output) if args.output else _default_path(cfg, "gallery_review_ingested.json")
    if args.dry_run:
        return {"status": "dry-run", "output": str(output_path)}

    if not args.review_file or not args.mapping_file:
        raise CliError("ingest-gallery-review requires --review-file and --mapping-file")
    mapping = _load_mapping(Path(args.mapping_file), review.GalleryReviewMappingEntry)
    raw_rows = read_json(Path(args.review_file))
    expected_group_keys = sorted({m.group_key for m in mapping})
    blind_labels_by_group: dict[str, list[str]] = {}
    for m in mapping:
        blind_labels_by_group.setdefault(m.group_key, [])
        if m.blind_label not in blind_labels_by_group[m.group_key]:
            blind_labels_by_group[m.group_key].append(m.blind_label)
    rows = review.ingest_gallery_review(raw_rows, expected_group_keys, blind_labels_by_group)
    write_json(output_path, [
        {
            "group_key": row.group_key,
            "blind_labels": list(row.blind_labels),
            "ratings": {label: _jsonable(rating) for label, rating in row.ratings.items()},
            "preference": row.preference,
            "evidence": row.evidence,
            "confidence": row.confidence,
        }
        for row in rows
    ])
    return {"status": "ok", "output_path": str(output_path), "row_count": len(rows)}


def _cmd_metrics(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    run_dir = cfg.resolve_root(cfg.run.outputs_root) / (args.run_id or cfg.run.name)
    if args.dry_run:
        return {"status": "dry-run", "run_dir": str(run_dir)}
    if not run_dir.exists():
        raise CliError(f"run directory not found: {run_dir}")

    summary = metrics.run_metrics(run_dir, device=args.device, force=args.force)
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "per_image_count": len(summary.per_image),
        "per_pair_count": len(summary.per_pair),
        "per_group_count": len(summary.per_group),
        "metric_config_hash": summary.metric_config_hash,
        "versions": summary.versions,
    }


def _cmd_analyze(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    run_dir = cfg.resolve_root(cfg.run.outputs_root) / (args.run_id or cfg.run.name)
    if args.dry_run:
        return {"status": "dry-run", "run_dir": str(run_dir)}
    if not run_dir.exists():
        raise CliError(f"run directory not found: {run_dir}")

    gallery_review_rows = None
    gallery_review_mapping = None
    if args.gallery_review_file:
        raw_rows = read_json(Path(args.gallery_review_file))
        gallery_review_rows = tuple(_gallery_row_from_dict(row) for row in raw_rows)
        if not args.gallery_mapping_file:
            raise CliError("--gallery-review-file requires --gallery-mapping-file")
        gallery_review_mapping = _load_mapping(Path(args.gallery_mapping_file), review.GalleryReviewMappingEntry)

    result = analysis.analyze_run(run_dir, gallery_review_rows=gallery_review_rows, gallery_review_mapping=gallery_review_mapping)
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "condition_summary_count": len(result["condition_summaries"]),
        "paired_effect_count": len(result["paired_effects"]),
        "gallery_review_status": result["gallery_review"]["status"],
    }


def _cmd_workflow(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    if cfg.psd.calibration_profile in (config.EFFECT_CALIBRATION_PROFILE, config.EXPECTED_RMS_CALIBRATION_PROFILE):
        raise CliError(
            f"workflow does not support {cfg.psd.calibration_profile}; run validate-config, "
            "basis-stability, calibrate, validate-noise, generate-psd --stage preview, "
            "and export-preview-review in order"
        )
    return workflow.run_workflow(
        args.config,
        run_id=args.run_id,
        force=args.force,
        allow_synthetic_basis=args.allow_synthetic_basis,
        candidates_file=args.candidates_file,
        probe_group_review_file=args.probe_group_review_file,
        preview_exclusion_file=args.preview_exclusion_file,
        generation_python=args.generation_python,
        metrics_python=args.metrics_python,
        skip_smoke_check=args.skip_smoke_check,
        dry_run=args.dry_run,
    )


_HANDLERS = {
    "validate-config": _cmd_validate_config,
    "build-basis": _cmd_build_basis,
    "inspect-basis": _cmd_inspect_basis,
    "basis-stability": _cmd_basis_stability,
    "probe": _cmd_probe,
    "export-review": _cmd_export_review,
    "select-candidates": _cmd_select_candidates,
    "calibrate": _cmd_calibrate,
    "validate-noise": _cmd_validate_noise,
    "generate-psd": _cmd_generate_psd,
    "export-preview-review": _cmd_export_preview_review,
    "ingest-preview-review": _cmd_ingest_preview_review,
    "export-gallery-review": _cmd_export_gallery_review,
    "ingest-gallery-review": _cmd_ingest_gallery_review,
    "metrics": _cmd_metrics,
    "analyze": _cmd_analyze,
    "workflow": _cmd_workflow,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pc_specific_psd", description="PCA-specific PSD noise-editing pipeline for SDXL-Turbo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="path to the PCASpecificPSDConfig YAML file")
    common.add_argument("--run-id", default=None, help="override the run directory / manifest id")
    common.add_argument("--dry-run", action="store_true", help="report planned actions without touching disk or a model")

    synthetic = argparse.ArgumentParser(add_help=False)
    synthetic.add_argument("--allow-synthetic-basis", action="store_true", help="permit a synthetic basis (tests only)")

    force = argparse.ArgumentParser(add_help=False)
    force.add_argument("--force", action="store_true", help="regenerate/recompute even if a complete resume point exists")

    candidates_flag = argparse.ArgumentParser(add_help=False)
    candidates_flag.add_argument("--candidates-file", default=None, help="path to select-candidates' output JSON")

    subparsers.add_parser("validate-config", parents=[common])
    subparsers.add_parser("build-basis", parents=[common])
    subparsers.add_parser("inspect-basis", parents=[common, synthetic])
    stability_parser = subparsers.add_parser("basis-stability", parents=[common])
    stability_parser.add_argument("--output", default=None)

    probe_parser = subparsers.add_parser("probe", parents=[common, synthetic, force])
    followup_group = probe_parser.add_mutually_exclusive_group()
    followup_group.add_argument("--rho020-followup", action="store_true")
    followup_group.add_argument("--candidate-recheck", action="store_true")
    probe_parser.add_argument("--group-ids", default=None, help="comma-separated PC group ids, for --candidate-recheck")
    probe_parser.add_argument("--review-file", default=None, help="ingested probe-review rows, for --rho020-followup")
    probe_parser.add_argument("--mapping-file", default=None, help="probe-review mapping, for --rho020-followup")

    export_review_parser = subparsers.add_parser("export-review", parents=[common])
    export_review_parser.add_argument("--review-output", default=None)
    export_review_parser.add_argument("--mapping-output", default=None)

    select_parser = subparsers.add_parser("select-candidates", parents=[common])
    select_parser.add_argument("--review-file", default=None)
    select_parser.add_argument("--mapping-file", default=None)
    select_parser.add_argument("--output", default=None)

    subparsers.add_parser("calibrate", parents=[common, synthetic, candidates_flag])
    subparsers.add_parser("validate-noise", parents=[common, synthetic])

    generate_parser = subparsers.add_parser("generate-psd", parents=[common, synthetic, force, candidates_flag])
    generate_parser.add_argument("--stage", choices=["preview", "full"], required=True)
    generate_parser.add_argument(
        "--approved-conditions-file", default=None,
        help="preview-exclusion-review output (--stage full only): narrows the full-pilot manifest to exactly these condition ids",
    )

    export_preview_parser = subparsers.add_parser("export-preview-review", parents=[common, candidates_flag])
    export_preview_parser.add_argument("--review-output", default=None)
    export_preview_parser.add_argument("--mapping-output", default=None)

    ingest_preview_parser = subparsers.add_parser("ingest-preview-review", parents=[common, candidates_flag])
    ingest_preview_parser.add_argument("--review-file", default=None)
    ingest_preview_parser.add_argument("--mapping-file", default=None)
    ingest_preview_parser.add_argument("--output", default=None)

    export_gallery_parser = subparsers.add_parser("export-gallery-review", parents=[common, candidates_flag])
    export_gallery_parser.add_argument("--review-output", default=None)
    export_gallery_parser.add_argument("--mapping-output", default=None)

    ingest_gallery_parser = subparsers.add_parser("ingest-gallery-review", parents=[common, candidates_flag])
    ingest_gallery_parser.add_argument("--review-file", default=None)
    ingest_gallery_parser.add_argument("--mapping-file", default=None)
    ingest_gallery_parser.add_argument("--output", default=None)

    metrics_parser = subparsers.add_parser("metrics", parents=[common, force, candidates_flag])
    metrics_parser.add_argument("--device", default="cpu")

    analyze_parser = subparsers.add_parser("analyze", parents=[common])
    analyze_parser.add_argument("--gallery-review-file", default=None)
    analyze_parser.add_argument("--gallery-mapping-file", default=None)

    workflow_parser = subparsers.add_parser("workflow", parents=[common, synthetic, force, candidates_flag])
    workflow_parser.add_argument("--generation-python", default=sys.executable, help="interpreter used for build-basis/probe/preview/full-pilot stages")
    workflow_parser.add_argument("--metrics-python", default=sys.executable, help="interpreter used for the metrics stage")
    workflow_parser.add_argument("--skip-smoke-check", action="store_true", help="skip the pre-probe/pre-full-pilot smoke check (CPU/test runs only)")
    workflow_parser.add_argument("--probe-group-review-file", default=None, help="directory or path override for the probe-group review CSV (default: alongside --config)")
    workflow_parser.add_argument("--preview-exclusion-file", default=None, help="directory or path override for the preview-exclusion review CSV (default: alongside --config)")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    candidate_group_ids = None
    explicit_candidates_file = getattr(args, "candidates_file", None)
    if explicit_candidates_file:
        candidates_path = Path(explicit_candidates_file)
        if candidates_path.exists():
            candidate_group_ids = tuple(read_json(candidates_path)["candidates"])

    try:
        cfg = config.resolve_config(args.config, args.command, candidate_group_ids)
    except config.ConfigValidationError as exc:
        if args.dry_run:
            print(f"dry-run: {args.command} is not yet resolvable -- missing: {', '.join(exc.missing_fields)}")
            return 0
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except config.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    handler = _HANDLERS[args.command]
    try:
        result = handler(cfg, args)
    except (CliError, config.ConfigError, runner.RunnerError, probing.ProbingError, metrics.MetricsError, analysis.AnalysisError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    import json

    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
