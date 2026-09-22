"""Generation runner for the PSD-edited PC-condition study (plan sections 6-8).

Mirrors ``noise_init.run_experiment``'s resume/immutability pattern exactly
(``ensure_immutable_run``, a per-sample-dir completeness check, refusing
rather than overwriting an incomplete existing sample dir) rather than
inventing a new one, but replaces its ``alpha``/``gamma`` condition schema
with the PC-condition schema this study actually needs: ``pc_group_id``,
``tau``, ``gate_id``, ``basis_hash``, ``calibration_hash``.

Every image is generated through exactly one call site,
``adapters.SDXLTurboAdapterPCA.generate`` -- never the raw pipeline -- so the
paired-generator/revision rules stay structurally enforced. That adapter
already loops one image at a time internally regardless of how many
``pair_keys`` are passed in a single call, so this module always calls it
with a batch of 1 (matching ``run_experiment.py``'s per-image-directory
resume granularity) rather than trying to batch for a speed that does not
actually exist underneath.

Frozen corrections (plan section 5.4/6.1) are profile-specific. V1 stores
only the selected gate/tau and deterministically reconstructs the correction
once per run from the seeded calibration bank. Effect-size v2 instead
persists the selected frozen correction, so held-out validation and preview
use exactly the calibrated operator without refitting. Neither profile
recomputes a correction per image. The reference condition never needs one:
it uses
``psd_editor.apply_psd_edit_tau_zero`` directly, which is
``same_phase_floor`` times the frozen analytic expected-unit-RMS scale and
carries an implicit all-ones correction by
definition (``calibration.zero_tau_correction``), so no calibration-bank
reconstruction is required for it at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from pc_specific_psd import basis as basis_module
from pc_specific_psd import calibration, manifests, patch_codec, psd_editor
from pc_specific_psd.adapters import SDXLTurboAdapterPCA
from pc_specific_psd.compat_generation import (
    ensure_immutable_run,
    file_hash,
    read_json,
    read_jsonl,
    tensor_hash,
    write_json,
    write_jsonl,
)
from pc_specific_psd.config import PCASpecificPSDConfig, load_calibration_registry
from pc_specific_psd.patch_codec import OverlapCodec

REFERENCE_CONDITION_ID = "reference"


class RunnerError(RuntimeError):
    """Raised for resume/provenance failures analogous to run_experiment.py's."""


def _run_dir(config: PCASpecificPSDConfig, run_id: str | None) -> Path:
    root = config.resolve_root(config.run.outputs_root)
    return root / (run_id or config.run.name)


def _sample_dir(run_dir: Path, condition_id: str, entry: manifests.ConditionDrawEntry) -> Path:
    return run_dir / "generations" / condition_id / entry.block_id / f"b{entry.base_index}"


def _is_complete(sample_dir: Path, expected_final_hash: str) -> bool:
    records = read_jsonl(sample_dir / "sample.jsonl")
    if len(records) != 1:
        return False
    row = records[0]
    return Path(row["image_path"]).exists() and row["final_noise_hash"] == expected_final_hash


def basis_hash_for(config: PCASpecificPSDConfig) -> str:
    path = config.resolve_root(config.basis.basis_output_path)
    if path is None or not path.exists():
        raise RunnerError(f"basis file does not exist: {path}; run build-basis first")
    return file_hash(path)


def calibration_hash_for(config: PCASpecificPSDConfig) -> str:
    path = config.resolve_root(config.psd.calibration_result_path)
    if path is None or not path.exists():
        raise RunnerError(f"calibration_result file does not exist: {path}; run calibrate first")
    return file_hash(path)


def load_codec(config: PCASpecificPSDConfig, *, allow_synthetic: bool = False) -> OverlapCodec:
    """``allow_synthetic`` defaults to False so a production run can never
    silently substitute a synthetic basis; CPU tests pass True explicitly.
    """
    path = config.resolve_root(config.basis.basis_output_path)
    loaded = basis_module.load_basis(
        path, allow_synthetic=allow_synthetic,
        expected_patch_size=config.basis.patch_size, expected_channels=config.basis.channels,
    )
    return OverlapCodec(loaded.components, loaded.patch_size, loaded.channels)


def _calibration_bank(config: PCASpecificPSDConfig, *, height: int, width: int) -> torch.Tensor:
    seed = config.psd.calibration_bank_seed
    size = config.psd.calibration_bank_size
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn((size, config.basis.channels, height, width), generator=generator, dtype=torch.float32)


@dataclass(frozen=True)
class FrozenCorrection:
    group_id: str
    sign: str  # "plus" | "minus"
    tau: float
    r_s: float
    beta: float
    correction: torch.Tensor  # (num_bins,) float64


@dataclass(frozen=True)
class EffectFrozenCorrection:
    condition_id: str
    group_id: str
    sign: str
    tau: float
    r_s: float
    beta: float
    correction: torch.Tensor
    target_relative_l2: tuple[float, ...]
    actual_relative_l2: float


def load_effect_frozen_corrections(config: PCASpecificPSDConfig) -> dict[str, EffectFrozenCorrection]:
    """Load, without refitting, the exact corrections selected by v2."""
    path = config.resolve_root(config.psd.calibration_result_path)
    payload = read_json(path)
    if payload.get("status") != "SELECTED" or payload.get("calibration_profile") != "effect_size_v2":
        raise RunnerError("effect_size_v2 calibration registry is not SELECTED")
    if payload.get("reference_scale_profile") != config.frozen.reference_scale_profile:
        raise RunnerError("effect_size_v2 calibration registry has a stale reference scale profile")
    if payload.get("config_hash") != file_hash(config.config_path):
        raise RunnerError("effect_size_v2 calibration registry is stale relative to the current config; run calibrate again")
    if payload.get("basis_hash") != basis_hash_for(config):
        raise RunnerError("effect_size_v2 calibration registry is stale relative to the formal basis")
    gate = payload["gate"]
    return {
        condition_id: EffectFrozenCorrection(
            condition_id=condition_id,
            group_id=raw["group_id"], sign=raw["sign"], tau=float(raw["tau"]),
            r_s=float(gate["r_s"]), beta=float(gate["beta"]),
            correction=torch.tensor(raw["correction"], dtype=torch.float64),
            target_relative_l2=tuple(float(value) for value in raw["target_relative_l2"]),
            actual_relative_l2=float(raw["actual_relative_l2"]),
        )
        for condition_id, raw in payload.get("condition_selections", {}).items()
    }


def build_frozen_corrections(
    config: PCASpecificPSDConfig, codec: OverlapCodec, *, height: int, width: int,
) -> dict[tuple[str, str], FrozenCorrection]:
    """Recomputes each declared candidate group's tau+/tau- correction exactly
    once per run, reconstructing the calibration bank deterministically and
    replaying ``calibration.evaluate_candidate`` with the frozen gate/tau from
    ``load_calibration_registry`` -- never reading a persisted correction
    tensor, since one is never written to disk (see module docstring).
    """
    registry = load_calibration_registry(config)
    if not registry.loaded or registry.status != "SELECTED" or registry.gate is None:
        raise RunnerError("calibration registry is not SELECTED; run calibrate first")
    if registry.reference_scale_profile != config.frozen.reference_scale_profile:
        raise RunnerError(
            "calibration registry uses reference_scale_profile "
            f"{registry.reference_scale_profile!r}, expected {config.frozen.reference_scale_profile!r}; "
            "it is stale and must be regenerated -- run calibrate again"
        )
    bank = _calibration_bank(config, height=height, width=width)
    reference_power = calibration.compute_reference_power(bank, config.psd.num_bins, protocol=registry.protocol)

    corrections: dict[tuple[str, str], FrozenCorrection] = {}
    for group in config.psd.groups:
        if group.group_id not in registry.group_taus:
            continue
        tau_plus, tau_minus = registry.group_taus[group.group_id]
        group_indices = list(manifests.pc_group_by_id(group.group_id).indices)
        for sign, tau in (("plus", tau_plus), ("minus", tau_minus)):
            evaluation = calibration.evaluate_candidate(
                codec, bank, group_indices, registry.gate, tau, reference_power,
                num_bins=config.psd.num_bins, protocol=registry.protocol,
                psd_tolerance=config.psd.psd_tolerance, correction_gain_bound=config.psd.correction_gain_bound,
                condition_number_threshold=config.psd.condition_number_threshold,
            )
            if not evaluation.accepted:
                raise RunnerError(
                    f"replayed calibration for group {group.group_id!r} sign {sign!r} did not reproduce "
                    f"an accepted candidate (reason={evaluation.reason!r}); calibration_result.json may be "
                    "stale relative to the declared psd config or calibration bank settings"
                )
            corrections[(group.group_id, sign)] = FrozenCorrection(
                group_id=group.group_id, sign=sign, tau=tau,
                r_s=registry.gate.r_s, beta=registry.gate.beta, correction=evaluation.correction,
            )
    return corrections


def _parse_condition(condition_id: str) -> tuple[str | None, str | None]:
    """``"reference"`` -> (None, None); ``"{group_id}_plus"``/``"_minus"`` -> (group_id, sign)."""
    if condition_id == REFERENCE_CONDITION_ID:
        return None, None
    group_id, sign = condition_id.rsplit("_", 1)
    if sign not in ("plus", "minus"):
        raise RunnerError(f"malformed condition_id {condition_id!r}")
    return group_id, sign


def _seed_block_for_entry(entry: manifests.ConditionDrawEntry) -> manifests.SeedBlock:
    prompt = manifests.prompt_by_id(entry.prompt_id)
    for block in manifests.seed_blocks_for_prompt(prompt):
        if block.block_id == entry.block_id:
            return block
    raise RunnerError(f"no seed block {entry.block_id!r} for prompt {entry.prompt_id!r}")


def render_condition_latent(
    entry: manifests.ConditionDrawEntry,
    codec: OverlapCodec,
    corrections: dict[tuple[str, str], FrozenCorrection],
    *,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Renders the PSD-edited (or reference) latent for one condition draw.

    The reference condition uses the exact-scalar shortcut
    (``apply_psd_edit_tau_zero``), never the general codec path, for
    production runs -- the forced full-codec-path equivalence check at tau=0
    lives only in ``tests/test_psd_editor.py``.
    """
    block = _seed_block_for_entry(entry)
    base_white = psd_editor.base_white_for_draw(block, entry.base_index, channels=channels, height=height, width=width)
    group_id, sign = _parse_condition(entry.condition_id)
    if group_id is None:
        return psd_editor.apply_psd_edit_tau_zero(base_white)
    frozen = corrections[(group_id, sign)]
    group_indices = list(manifests.pc_group_by_id(group_id).indices)
    edited = psd_editor.apply_psd_edit(codec, base_white, group_indices, frozen.tau, frozen.r_s, frozen.beta)
    return calibration.apply_radial_correction(edited, frozen.correction, len(frozen.correction))


def render_effect_condition_latent(
    entry: manifests.ConditionDrawEntry,
    codec: OverlapCodec,
    corrections: dict[str, EffectFrozenCorrection],
    *,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    block = _seed_block_for_entry(entry)
    base_white = psd_editor.base_white_for_draw(
        block, entry.base_index, channels=channels, height=height, width=width
    )
    if entry.condition_id == REFERENCE_CONDITION_ID:
        return psd_editor.apply_psd_edit_tau_zero(base_white)
    try:
        frozen = corrections[entry.condition_id]
    except KeyError as exc:
        raise RunnerError(f"unknown frozen effect condition {entry.condition_id!r}") from exc
    group_indices = tuple(manifests.pc_group_by_id(frozen.group_id).indices)
    edited = psd_editor.apply_psd_edit(
        codec, base_white, group_indices, frozen.tau, frozen.r_s, frozen.beta
    )
    return calibration.apply_radial_correction(edited, frozen.correction, len(frozen.correction))


def _generation_hash(
    config: PCASpecificPSDConfig,
    basis_hash: str,
    calibration_hash: str,
    validation_hash: str | None = None,
) -> str:
    from pc_specific_psd.compat_generation import sha256_text

    base = (config.model.as_model_config_dict(), config.generation.config, basis_hash, calibration_hash)
    return sha256_text(*base) if validation_hash is None else sha256_text(*base, validation_hash)


def _run_provenance(
    config: PCASpecificPSDConfig,
    basis_hash: str,
    calibration_hash: str,
    validation_hash: str | None = None,
) -> dict:
    provenance = {
        "model": config.model.as_model_config_dict(),
        "generation": {
            "height": config.generation.config.height,
            "width": config.generation.config.width,
            "num_inference_steps": config.generation.config.num_inference_steps,
            "guidance_scale": config.generation.config.guidance_scale,
            "generation_batch_size": config.generation.config.generation_batch_size,
        },
        "basis_hash": basis_hash,
        "calibration_hash": calibration_hash,
        "reference_scale_profile": config.frozen.reference_scale_profile,
    }
    if validation_hash is not None:
        provenance["validation_hash"] = validation_hash
    return provenance


def generate_manifest(
    config: PCASpecificPSDConfig,
    entries: Sequence[manifests.ConditionDrawEntry],
    *,
    run_id: str | None,
    force: bool,
    adapter: SDXLTurboAdapterPCA | None = None,
    allow_synthetic_basis: bool = False,
) -> Path:
    """Generates every image in ``entries``, resuming like ``run_experiment.py``:
    an already-complete sample dir is skipped, an incomplete existing one
    raises rather than being overwritten (a new ``--run-id`` is required to
    retry), and the run directory's provenance is pinned by
    ``ensure_immutable_run`` including this run's ``basis_hash``/
    ``calibration_hash`` so a later resume against a different basis or
    calibration result is refused rather than silently mixed in.
    """
    if not entries:
        raise RunnerError("generate_manifest called with an empty manifest")
    run_dir = _run_dir(config, run_id)
    basis_hash = basis_hash_for(config)
    calibration_hash = calibration_hash_for(config)
    ensure_immutable_run(run_dir, _run_provenance(config, basis_hash, calibration_hash), force=force)

    codec = load_codec(config, allow_synthetic=allow_synthetic_basis)
    height = config.generation.config.height // 8
    width = config.generation.config.width // 8
    channels = config.basis.channels
    corrections = build_frozen_corrections(config, codec, height=height, width=width)

    adapter = adapter or SDXLTurboAdapterPCA(config.model.as_model_config_dict())
    for entry in entries:
        latent = render_condition_latent(entry, codec, corrections, channels=channels, height=height, width=width)
        final_hash = tensor_hash(latent[0])
        sample_dir = _sample_dir(run_dir, entry.condition_id, entry)
        if _is_complete(sample_dir, final_hash) and not force:
            continue
        if sample_dir.exists():
            raise RunnerError(f"Incomplete existing generation at {sample_dir}; refusing to overwrite source images")
        sample_dir.mkdir(parents=True, exist_ok=True)

        pair_key = (entry.prompt_id, entry.block_id, entry.base_index)
        prompt_text = manifests.prompt_by_id(entry.prompt_id).text
        images = adapter.generate(prompt_text, latent, [pair_key], seed=config.run.master_seed, generation_config=config.generation.config)
        if len(images) != 1:
            raise RunnerError(f"Adapter returned {len(images)} images; expected 1")
        group_id, sign = _parse_condition(entry.condition_id)
        tau = 0.0 if group_id is None else corrections[(group_id, sign)].tau
        gate_id = None if group_id is None else f"r_s={corrections[(group_id, sign)].r_s}_beta={corrections[(group_id, sign)].beta}"

        target = sample_dir / "image.png"
        images[0].save(target, format="PNG")
        record = {
            "run_id": run_dir.name, "condition_id": entry.condition_id,
            "prompt_id": entry.prompt_id, "block_id": entry.block_id, "base_index": entry.base_index,
            "pc_group_id": group_id, "tau": tau, "gate_id": gate_id,
            "basis_hash": basis_hash, "calibration_hash": calibration_hash,
            "final_noise_hash": final_hash,
            "adapter_prepared_latent_hash": adapter.last_generated_latent_hashes[0],
            "generator_seed": adapter.last_generator_seeds[0],
            "image_path": str(target.resolve()), "image_hash": file_hash(target),
            "generation_config_hash": _generation_hash(config, basis_hash, calibration_hash),
        }
        write_jsonl(sample_dir / "sample.jsonl", [record])
    if config.generation.release_model_after_generation:
        adapter.close()
    return run_dir


def generate_preview(
    config: PCASpecificPSDConfig, candidate_group_ids: Sequence[str], *,
    run_id: str | None = None, force: bool = False, adapter: SDXLTurboAdapterPCA | None = None,
    allow_synthetic_basis: bool = False,
) -> Path | None:
    """Zero candidates -> no preview manifest at all (returns ``None``); one
    or two candidates build and generate the preview manifest exactly as
    ``manifests.build_preview_manifest_entries`` sizes it -- a real branch on
    ``select_candidates()``'s actual output length, never a hardcoded count.
    ``adapter`` mirrors ``generate_manifest``'s own override (tests inject a
    fake one; production leaves it ``None`` to build the real adapter).
    """
    entries = manifests.build_preview_manifest_entries(candidate_group_ids)
    if not entries:
        return None
    return generate_manifest(
        config, entries, run_id=run_id or f"{config.run.name}_preview", force=force,
        adapter=adapter, allow_synthetic_basis=allow_synthetic_basis,
    )


def generate_effect_preview(
    config: PCASpecificPSDConfig,
    condition_ids: Sequence[str],
    *,
    run_id: str | None = None,
    force: bool = False,
    adapter: SDXLTurboAdapterPCA | None = None,
    allow_synthetic_basis: bool = False,
) -> Path:
    """Generate the bounded v2 preview with the persisted final operators."""
    entries = manifests.build_effect_preview_manifest_entries(condition_ids)
    validation_path = config.resolve_root(config.psd.validation_result_path)
    if validation_path is None or not validation_path.exists():
        raise RunnerError("effect_size_v2 noise validation result is missing")
    validation = read_json(validation_path)
    basis_hash = basis_hash_for(config)
    calibration_hash = calibration_hash_for(config)
    validation_hash = file_hash(validation_path)
    if (
        validation.get("status") != "PASS"
        or validation.get("calibration_profile") != "effect_size_v2"
        or validation.get("config_hash") != file_hash(config.config_path)
        or validation.get("basis_hash") != basis_hash
        or validation.get("calibration_hash") != calibration_hash
    ):
        raise RunnerError("effect_size_v2 noise validation is missing, failed, or stale")
    run_dir = _run_dir(config, run_id or f"{config.run.name}_preview")
    provenance = _run_provenance(config, basis_hash, calibration_hash, validation_hash)
    provenance.update({"calibration_profile": "effect_size_v2", "preview_only": True})
    ensure_immutable_run(run_dir, provenance, force=force)

    codec = load_codec(config, allow_synthetic=allow_synthetic_basis)
    height = config.generation.config.height // 8
    width = config.generation.config.width // 8
    channels = config.basis.channels
    corrections = load_effect_frozen_corrections(config)
    missing = set(condition_ids) - set(corrections)
    if missing:
        raise RunnerError(f"preview condition(s) missing from frozen registry: {sorted(missing)}")
    adapter = adapter or SDXLTurboAdapterPCA(config.model.as_model_config_dict())
    for entry in entries:
        latent = render_effect_condition_latent(
            entry, codec, corrections, channels=channels, height=height, width=width
        )
        final_hash = tensor_hash(latent[0])
        sample_dir = _sample_dir(run_dir, entry.condition_id, entry)
        if _is_complete(sample_dir, final_hash) and not force:
            continue
        if sample_dir.exists():
            raise RunnerError(f"Incomplete existing generation at {sample_dir}; refusing to overwrite source images")
        sample_dir.mkdir(parents=True, exist_ok=True)
        pair_key = (entry.prompt_id, entry.block_id, entry.base_index)
        images = adapter.generate(
            manifests.prompt_by_id(entry.prompt_id).text, latent, [pair_key],
            seed=config.run.master_seed, generation_config=config.generation.config,
        )
        if len(images) != 1:
            raise RunnerError(f"Adapter returned {len(images)} images; expected 1")
        frozen = corrections.get(entry.condition_id)
        target = sample_dir / "image.png"
        images[0].save(target, format="PNG")
        record = {
            "run_id": run_dir.name, "condition_id": entry.condition_id,
            "prompt_id": entry.prompt_id, "block_id": entry.block_id,
            "base_index": entry.base_index,
            "sample_seed": manifests.sample_seed_for_entry(entry),
            "pc_group_id": frozen.group_id if frozen else None,
            "sign": frozen.sign if frozen else None,
            "tau": frozen.tau if frozen else 0.0,
            "target_relative_l2": list(frozen.target_relative_l2) if frozen else [],
            "actual_relative_l2": frozen.actual_relative_l2 if frozen else 0.0,
            "gate_id": f"r_s={frozen.r_s}_beta={frozen.beta}" if frozen else None,
            "basis_hash": basis_hash, "calibration_hash": calibration_hash,
            "validation_hash": validation_hash,
            "calibration_profile": "effect_size_v2",
            "final_noise_hash": final_hash,
            "adapter_prepared_latent_hash": adapter.last_generated_latent_hashes[0],
            "generator_seed": adapter.last_generator_seeds[0],
            "image_path": str(target.resolve()), "image_hash": file_hash(target),
            "generation_config_hash": _generation_hash(
                config, basis_hash, calibration_hash, validation_hash
            ),
        }
        write_jsonl(sample_dir / "sample.jsonl", [record])
    if config.generation.release_model_after_generation:
        adapter.close()
    return run_dir


def generate_full_pilot(
    config: PCASpecificPSDConfig, candidate_group_ids: Sequence[str], *,
    run_id: str | None = None, force: bool = False, approved_condition_ids: Sequence[str] | None = None,
    adapter: SDXLTurboAdapterPCA | None = None, allow_synthetic_basis: bool = False,
) -> Path | None:
    """Zero candidates -> no full-pilot manifest at all (returns ``None``).
    Full-pilot draws use ``manifests.FULL_PILOT_BASE_INDICES``, a base_index
    range never touched by probing or preview -- so every full-pilot image is
    an independent, previously-unseen draw, never a reused preview sample.
    When ``approved_condition_ids`` is given (preview-exclusion review has
    run), the manifest is built over exactly that condition subset via
    ``manifests.build_full_pilot_manifest_entries_for_conditions`` instead of
    every candidate's both signs.
    """
    if approved_condition_ids is not None:
        entries = manifests.build_full_pilot_manifest_entries_for_conditions(approved_condition_ids)
    else:
        entries = manifests.build_full_pilot_manifest_entries(candidate_group_ids)
    if not entries:
        return None
    if candidate_group_ids:
        already_reviewed_entries = manifests.build_preview_manifest_entries(candidate_group_ids)
        manifests.assert_disjoint_sample_seeds(already_reviewed_entries, entries)
    return generate_manifest(
        config, entries, run_id=run_id or f"{config.run.name}_full", force=force,
        adapter=adapter, allow_synthetic_basis=allow_synthetic_basis,
    )
