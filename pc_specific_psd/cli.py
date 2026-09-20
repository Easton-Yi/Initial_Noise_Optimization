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
    config,
    manifests,
    metrics,
    probing,
    review,
    runner,
    workflow,
)
from pc_specific_psd.adapters import SDXLTurboAdapterPCA
from pc_specific_psd.compat_generation import read_json, write_json


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
    return {"status": "ok", "run_name": cfg.run.name, "config_path": str(cfg.config_path)}


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
    )
    basis_module.save_basis(built, output_path)
    if cfg.generation.release_model_after_generation:
        adapter.close()
    return {
        "status": "ok",
        "basis_output_path": str(output_path),
        "num_samples": built.num_samples,
        "orthogonality_error": basis_module.orthogonality_error(built.components),
        "split_half_max_angle_leading": stability.max_angle_leading,
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


def _cmd_probe(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    channels = cfg.basis.channels
    height, width = cfg.generation.config.height, cfg.generation.config.width
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
        channels=cfg.basis.channels, height=cfg.generation.config.height, width=cfg.generation.config.width,
        patch_size=cfg.basis.patch_size, rho=cfg.probing.rho,
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
    height, width = cfg.generation.config.height, cfg.generation.config.width
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


def _cmd_validate_noise(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
    """Cache-only replay-and-check: reloads the frozen calibration registry
    and re-runs ``calibration.evaluate_candidate`` against a freshly
    reconstructed calibration bank for every declared group/sign, confirming
    the frozen correction still reproduces an accepted candidate -- no model
    load, no new generation, mirroring ``noise_init/noise_validation.py``'s
    standalone pattern.
    """
    height, width = cfg.generation.config.height, cfg.generation.config.width
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


def _cmd_generate_psd(cfg: config.PCASpecificPSDConfig, args: argparse.Namespace) -> dict:
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
    rows = review.ingest_preview_review(raw_rows, expected_image_ids=tuple(m.image_id for m in mapping))
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

    metrics_parser = subparsers.add_parser("metrics", parents=[common, force])
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
