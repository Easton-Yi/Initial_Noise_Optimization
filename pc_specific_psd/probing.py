"""Phase B: the 84-image probing manifest and role-intervention realization
(plan §4.2-§4.4, §14.2, §14.4).

Three structurally distinct builders live here, each gated independently:

- ``build_probing_manifest`` -- the primary 84-image manifest (rho=0.10).
- ``build_rho020_supplement_manifest`` -- the one-time, all-six-groups
  uniform supplement, gated by ``compute_rho020_trigger`` on ingested
  probe-review data (permitted only when fewer than 2 of the 4 prompts show
  a clear role change).
- ``build_candidate_recheck_manifest`` -- the confirmation recheck for
  groups ``select_candidates()`` has actually nominated or reserved, gated
  only by that candidate list being non-empty.

These two follow-ups share no trigger function and are not alternatives to
each other; both may fire, neither may fire, or either may fire alone,
depending on what the probe review actually shows.

Every probe's rotation is applied to raw, unnormalized Gaussian PCA
coefficients -- ``normalize()`` is never called here, since that legacy
per-sample std step belongs only to the final editor's ``legacy-matched``
protocol, applied after the PSD edit, not before probing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

from pc_specific_psd import manifests, patch_codec
from pc_specific_psd.adapters import SDXLTurboAdapterPCA
from pc_specific_psd.basis import PCABasis, load_basis
from pc_specific_psd.compat_generation import (
    ensure_immutable_run, file_hash, read_jsonl, sample_noise_batch, sha256_text, tensor_hash, write_jsonl,
)

DEFAULT_RHO = 0.10
RHO020_SUPPLEMENT_RHO = 0.20

# Donor-index convention: each follow-up gets its own donor draw at the same
# (block_id, base_index) so a supplement/recheck never silently replays the
# exact rotation direction of an earlier probe at a different rho.
PRIMARY_PROBE_DONOR_INDEX = 0
RHO020_SUPPLEMENT_DONOR_INDEX = 1
CANDIDATE_RECHECK_DONOR_INDEX = 2

ROLE_FIELDS: tuple[str, ...] = ("layout", "pose", "shape", "color", "light", "texture")
CLEAR_CHANGE_RATING = 2


def fair_budget_angle(rho: float, total_dimension: int, num_patches: int, group_size: int) -> float:
    """``theta_B = arccos(1 - rho^2 D / (2 N_patch |B|))`` (plan §4.3/§14.4).

    Raises ``ValueError`` rather than clamping when the computed angle would
    fall outside ``[0, pi/2]`` -- an out-of-range result means the configured
    rho/patch-grid/group-size combination is infeasible for a fair-budget
    probe, and silently clipping would hide that instead of surfacing it.
    """
    if rho < 0:
        raise ValueError("rho must be non-negative")
    if num_patches <= 0 or group_size <= 0:
        raise ValueError("num_patches and group_size must be positive")
    cos_theta = 1.0 - (rho ** 2 * total_dimension) / (2.0 * num_patches * group_size)
    if not 0.0 <= cos_theta <= 1.0:
        raise ValueError(
            f"infeasible fair-budget angle: cos(theta)={cos_theta} outside [0, 1] "
            f"(rho={rho}, D={total_dimension}, N_patch={num_patches}, |B|={group_size}); "
            "reduce rho or adjust the PC-group partition rather than clamping"
        )
    return math.acos(cos_theta)


@dataclass(frozen=True)
class ProbeEntry:
    """One of the 84 probing images: a fixed (prompt, seed-block,
    base_index=0) draw, either untouched (``condition_type == "reference"``)
    or with one PC group's coefficients rotated toward an independent donor
    (``condition_type == "intervention"``).
    """
    prompt_id: str
    prompt_text: str
    block_id: str
    batch_seed: int
    base_index: int
    sample_seed: int
    condition_type: str  # "reference" | "intervention"
    group_id: Optional[str] = None
    theta: Optional[float] = None
    donor_seed: Optional[int] = None

    @property
    def image_id(self) -> str:
        suffix = "reference" if self.condition_type == "reference" else str(self.group_id)
        return f"{self.block_id}_b{self.base_index}_{suffix}"


def _thetas_for_groups(rho: float, channels: int, height: int, width: int, patch_size: int, groups: Sequence[manifests.PCGroup]) -> dict[str, float]:
    grid = patch_codec.centered_grid(height, width, patch_size)
    total_dimension = channels * height * width
    return {group.group_id: fair_budget_angle(rho, total_dimension, grid.n_patches, group.size) for group in groups}


def build_probing_manifest(
    *,
    channels: int,
    height: int,
    width: int,
    patch_size: int,
    rho: float = DEFAULT_RHO,
    groups: tuple[manifests.PCGroup, ...] = manifests.PC_GROUPS,
) -> tuple[ProbeEntry, ...]:
    """The 84-entry probing manifest: 4 prompts x 3 seed blocks x (6 group
    interventions + 1 reference) = 4*3*6 + 4*3 = 84 (plan §4.4), matching
    ``manifests.probing_image_count()`` exactly. ``patch_size`` here must be
    the same patch size the basis used in ``realize_probe`` was fit with --
    it is a shared pipeline hyperparameter, not independently configurable
    per phase.
    """
    thetas = _thetas_for_groups(rho, channels, height, width, patch_size, groups)
    entries: list[ProbeEntry] = []
    for prompt in manifests.PROMPTS:
        for block in manifests.seed_blocks_for_prompt(prompt):
            base_index = manifests.PROBING_BASE_INDEX
            sample_seed = block.sample_seed(base_index)
            entries.append(ProbeEntry(
                prompt_id=prompt.prompt_id, prompt_text=prompt.text, block_id=block.block_id,
                batch_seed=block.batch_seed, base_index=base_index, sample_seed=sample_seed,
                condition_type="reference",
            ))
            donor = manifests.donor_seed(block.block_id, base_index, donor_index=PRIMARY_PROBE_DONOR_INDEX)
            for group in groups:
                entries.append(ProbeEntry(
                    prompt_id=prompt.prompt_id, prompt_text=prompt.text, block_id=block.block_id,
                    batch_seed=block.batch_seed, base_index=base_index, sample_seed=sample_seed,
                    condition_type="intervention", group_id=group.group_id,
                    theta=thetas[group.group_id], donor_seed=donor,
                ))
    expected = manifests.probing_image_count(len(manifests.PROMPTS), len(groups))
    if len(entries) != expected:
        raise AssertionError(f"probing manifest built {len(entries)} entries, expected {expected}")
    return tuple(entries)


def _draw_donor_latent(donor_seed: int, *, channels: int, height: int, width: int) -> torch.Tensor:
    """Independent standard-Gaussian donor draw (plan §14.2): a probing-only
    artifact distinct from the ``base_white`` base-draw convention -- one
    donor tensor per base draw, shared by every PC group probed off of it
    (``donor_seed`` already excludes ``group_id`` by construction).
    """
    generator = torch.Generator("cpu").manual_seed(donor_seed)
    return torch.randn((1, channels, height, width), generator=generator, dtype=torch.float32)


def draw_base_latent(entry: ProbeEntry, *, channels: int, height: int, width: int) -> torch.Tensor:
    """The exact same raw-white base-draw path used everywhere else in the
    project (``sample_noise_batch`` at ``sample_seed = batch_seed +
    base_index``); zero PCA-specific machinery. ``master_seed`` only affects
    the unused ``independent_eta`` draw, so the block's own ``batch_seed`` is
    reused there for determinism.
    """
    batch = sample_noise_batch(entry.batch_seed, entry.block_id, (1, channels, height, width), batch_seed=entry.batch_seed)
    return batch.base_white


def _boundary_matches_template(output: torch.Tensor, template: torch.Tensor, grid: patch_codec.GridSpec) -> bool:
    covered = torch.zeros(output.shape[-2:], dtype=torch.bool)
    p = grid.patch_size
    covered[grid.origin_row:grid.origin_row + grid.n_rows * p, grid.origin_col:grid.origin_col + grid.n_cols * p] = True
    boundary = ~covered
    if not boundary.any():
        return True
    return bool(torch.allclose(output[:, :, boundary], template[:, :, boundary]))


@dataclass(frozen=True)
class ProbeRealization:
    """A materialized probe entry plus *measured* realized statistics
    alongside the theoretical target, so a mismatch between intended and
    realized perturbation is visible rather than assumed (plan §4.4).
    """
    entry: ProbeEntry
    output_latent: torch.Tensor
    theoretical_delta_energy: Optional[float]
    measured_delta_energy: Optional[float]
    theoretical_relative_rms: Optional[float]
    measured_relative_rms: Optional[float]
    group_energy_ratio: Optional[float]
    max_abs_error_vs_closed_form: Optional[float]
    complement_and_boundary_unchanged: bool


def realize_probe(entry: ProbeEntry, basis: PCABasis, *, channels: int, height: int, width: int) -> ProbeRealization:
    """Materializes one probe entry's output latent. Operates directly on
    raw, unnormalized Gaussian PCA coefficients -- ``normalize()`` is never
    called here.
    """
    if basis.channels != channels:
        raise ValueError(f"basis channels {basis.channels} does not match requested channels {channels}")
    base_white = draw_base_latent(entry, channels=channels, height=height, width=width)
    grid = patch_codec.centered_grid(height, width, basis.patch_size)
    codec = patch_codec.NonOverlapCodec(grid, basis.components, channels)

    if entry.condition_type == "reference":
        return ProbeRealization(
            entry=entry, output_latent=base_white,
            theoretical_delta_energy=None, measured_delta_energy=None,
            theoretical_relative_rms=None, measured_relative_rms=None,
            group_energy_ratio=None, max_abs_error_vs_closed_form=None,
            complement_and_boundary_unchanged=True,
        )

    if entry.group_id is None or entry.theta is None or entry.donor_seed is None:
        raise ValueError("intervention probe entry is missing group_id/theta/donor_seed")
    group = manifests.pc_group_by_id(entry.group_id)
    donor_latent = _draw_donor_latent(entry.donor_seed, channels=channels, height=height, width=width)

    coefficients = codec.encode(base_white)
    donor_coefficients = codec.encode(donor_latent)
    rotated = codec.rotate_block(coefficients, donor_coefficients, group.indices, entry.theta)
    output_latent = codec.decode(rotated, template=base_white)

    idx = list(group.indices)
    complement_mask = torch.ones(basis.components.shape[1], dtype=torch.bool)
    complement_mask[idx] = False

    delta = rotated[..., idx] - coefficients[..., idx]
    measured_delta_energy = float(delta.square().sum())
    n_patch, group_size = grid.n_patches, group.size
    theoretical_delta_energy = 2.0 * n_patch * group_size * (1.0 - math.cos(entry.theta))
    measured_relative_rms = math.sqrt(measured_delta_energy / (n_patch * group_size))
    theoretical_relative_rms = math.sqrt(max(0.0, 2.0 * (1.0 - math.cos(entry.theta))))

    closed_form = math.cos(entry.theta) * coefficients[..., idx] + math.sin(entry.theta) * donor_coefficients[..., idx]
    max_abs_error_vs_closed_form = float((rotated[..., idx] - closed_form).abs().max())

    group_energy = float(rotated[..., idx].square().sum())
    complement_energy = float(rotated[..., complement_mask].square().sum())
    group_energy_ratio = group_energy / complement_energy if complement_energy > 0 else float("inf")

    complement_unchanged = torch.allclose(rotated[..., complement_mask], coefficients[..., complement_mask])
    boundary_unchanged = _boundary_matches_template(output_latent, base_white, grid)

    return ProbeRealization(
        entry=entry, output_latent=output_latent,
        theoretical_delta_energy=theoretical_delta_energy, measured_delta_energy=measured_delta_energy,
        theoretical_relative_rms=theoretical_relative_rms, measured_relative_rms=measured_relative_rms,
        group_energy_ratio=group_energy_ratio, max_abs_error_vs_closed_form=max_abs_error_vs_closed_form,
        complement_and_boundary_unchanged=bool(complement_unchanged and boundary_unchanged),
    )


# -- Follow-up 1: rho=0.20 uniform supplement, gated by probe-review evidence -

@dataclass(frozen=True)
class ProbeAnnotation:
    """One blind-reviewed (prompt, seed-block, PC-group) role-rating record
    (plan §14.5). ``role_ratings`` maps each of ``ROLE_FIELDS`` to ``0``,
    ``1``, ``2``, or ``"NA"``; ``rating == 2`` means a clear role change on
    that field. This is the shape ``review.ingest_probe_review()`` is
    expected to produce once that module exists.
    """
    prompt_id: str
    block_id: str
    group_id: str
    role_ratings: dict


@dataclass(frozen=True)
class Rho020TriggerResult:
    prompts_with_clear_change: frozenset
    prompt_count: int
    permitted: bool
    reason: str


def compute_rho020_trigger(
    annotations: Sequence[ProbeAnnotation],
    *,
    prompts: tuple[manifests.Prompt, ...] = manifests.PROMPTS,
    groups: tuple[manifests.PCGroup, ...] = manifests.PC_GROUPS,
) -> Rho020TriggerResult:
    """Plan §14.4's actual documented trigger, not a generic "review exists"
    flag: counts, across the 4 prompts, how many prompts show *at least one*
    group with a clear (``rating == 2``) role change on *any* annotated role
    field (the full ``ROLE_FIELDS`` set, not just structural fields). The
    supplement is permitted only when that count is strictly less than 2.

    Only evaluates the annotation-based half of the gate; refusing a second
    exercise of the supplement for the same run is enforced by the run
    manifest at the CLI/runner layer, not by this function.
    """
    expected_keys = {
        (prompt.prompt_id, block.block_id, group.group_id)
        for prompt in prompts
        for block in manifests.seed_blocks_for_prompt(prompt)
        for group in groups
    }
    seen_keys = {(a.prompt_id, a.block_id, a.group_id) for a in annotations}
    missing = expected_keys - seen_keys
    if missing:
        return Rho020TriggerResult(
            prompts_with_clear_change=frozenset(), prompt_count=0, permitted=False,
            reason=f"{len(missing)} probe-review annotation(s) missing; cannot evaluate the rho=0.20 trigger",
        )

    prompts_with_change: set[str] = set()
    for annotation in annotations:
        if any(annotation.role_ratings.get(field) == CLEAR_CHANGE_RATING for field in ROLE_FIELDS):
            prompts_with_change.add(annotation.prompt_id)

    count = len(prompts_with_change)
    permitted = count < 2
    reason = (
        f"{count} of {len(prompts)} prompts show a clear role change; "
        + ("supplement permitted" if permitted else "supplement not permitted (need fewer than 2 prompts)")
    )
    return Rho020TriggerResult(
        prompts_with_clear_change=frozenset(prompts_with_change), prompt_count=count,
        permitted=permitted, reason=reason,
    )


def build_rho020_supplement_manifest(
    trigger: Rho020TriggerResult,
    *,
    channels: int,
    height: int,
    width: int,
    patch_size: int,
    groups: tuple[manifests.PCGroup, ...] = manifests.PC_GROUPS,
) -> tuple[ProbeEntry, ...]:
    """All six groups x 4 prompts x 3 base draws = 72 images at rho=0.20
    (plan §14.4); the 12 existing per-prompt references are reused, not
    regenerated, so this manifest contains no reference entries at all.
    """
    if not trigger.permitted:
        raise ValueError(f"rho=0.20 supplement not permitted: {trigger.reason}")
    thetas = _thetas_for_groups(RHO020_SUPPLEMENT_RHO, channels, height, width, patch_size, groups)
    entries: list[ProbeEntry] = []
    for prompt in manifests.PROMPTS:
        for block in manifests.seed_blocks_for_prompt(prompt):
            base_index = manifests.PROBING_BASE_INDEX
            sample_seed = block.sample_seed(base_index)
            donor = manifests.donor_seed(block.block_id, base_index, donor_index=RHO020_SUPPLEMENT_DONOR_INDEX)
            for group in groups:
                entries.append(ProbeEntry(
                    prompt_id=prompt.prompt_id, prompt_text=prompt.text, block_id=block.block_id,
                    batch_seed=block.batch_seed, base_index=base_index, sample_seed=sample_seed,
                    condition_type="intervention", group_id=group.group_id,
                    theta=thetas[group.group_id], donor_seed=donor,
                ))
    expected = manifests.rho020_supplement_image_count(len(manifests.PROMPTS), len(groups))
    if len(entries) != expected:
        raise AssertionError(f"rho=0.20 supplement manifest built {len(entries)} entries, expected {expected}")
    return tuple(entries)


# -- Follow-up 2: candidate/reserve recheck, gated only by select_candidates() -

def build_candidate_recheck_manifest(
    candidate_group_ids: Sequence[str],
    *,
    channels: int,
    height: int,
    width: int,
    patch_size: int,
    rho: float = DEFAULT_RHO,
    donor_index: int = CANDIDATE_RECHECK_DONOR_INDEX,
) -> tuple[ProbeEntry, ...]:
    """New-seed/donor recheck for groups ``select_candidates()`` has actually
    nominated or reserved (plan §14.4): gated *only* by ``candidate_group_ids``
    being non-empty -- never by the rho=0.20 prompt-count condition. Shares
    no trigger function or code path with ``compute_rho020_trigger``, since
    one is a role-evidence-insufficiency fallback and the other is a
    confirmation step for groups already nominated.
    """
    if len(candidate_group_ids) == 0:
        raise ValueError("candidate recheck requires at least one nominated/reserved group; got none")
    groups = tuple(manifests.pc_group_by_id(group_id) for group_id in candidate_group_ids)
    thetas = _thetas_for_groups(rho, channels, height, width, patch_size, groups)
    entries: list[ProbeEntry] = []
    for prompt in manifests.PROMPTS:
        for block in manifests.seed_blocks_for_prompt(prompt):
            base_index = manifests.PROBING_BASE_INDEX
            sample_seed = block.sample_seed(base_index)
            donor = manifests.donor_seed(block.block_id, base_index, donor_index=donor_index)
            for group in groups:
                entries.append(ProbeEntry(
                    prompt_id=prompt.prompt_id, prompt_text=prompt.text, block_id=block.block_id,
                    batch_seed=block.batch_seed, base_index=base_index, sample_seed=sample_seed,
                    condition_type="intervention", group_id=group.group_id,
                    theta=thetas[group.group_id], donor_seed=donor,
                ))
    return tuple(entries)


# -- Orchestration: actually render and persist probe images -----------------


class ProbingError(RuntimeError):
    """Raised for probe-generation orchestration failures (missing basis,
    an empty manifest, or an existing sample dir that doesn't match the
    latent this entry would produce).
    """


def probe_basis_hash_for(config) -> str:
    """Independent twin of ``runner.basis_hash_for`` -- duplicated rather than
    imported, since ``runner.py`` imports ``config.py`` which itself imports
    this module, and importing ``runner`` from here would close that cycle.
    """
    path = config.resolve_root(config.basis.basis_output_path)
    if path is None or not path.exists():
        raise ProbingError(f"basis file does not exist: {path}; run build-basis first")
    return file_hash(path)


def _probe_condition_label(entry: ProbeEntry) -> str:
    return "reference" if entry.condition_type == "reference" else entry.group_id


def _probe_sample_dir(run_dir: Path, entry: ProbeEntry) -> Path:
    return run_dir / "probes" / _probe_condition_label(entry) / entry.block_id / f"b{entry.base_index}"


def _probe_is_complete(sample_dir: Path, expected_final_hash: str) -> bool:
    records = read_jsonl(sample_dir / "sample.jsonl")
    if len(records) != 1:
        return False
    row = records[0]
    return Path(row["image_path"]).exists() and row["final_noise_hash"] == expected_final_hash


def _probe_run_dir(config, run_id: Optional[str]) -> Path:
    root = config.resolve_root(config.run.outputs_root)
    return root / (run_id or f"{config.run.name}_probe")


def _probe_run_provenance(config, basis_hash: str) -> dict:
    return {
        "model": config.model.as_model_config_dict(),
        "generation": {
            "height": config.generation.config.height,
            "width": config.generation.config.width,
            "num_inference_steps": config.generation.config.num_inference_steps,
            "guidance_scale": config.generation.config.guidance_scale,
            "generation_batch_size": config.generation.config.generation_batch_size,
        },
        "basis_hash": basis_hash,
    }


def generate_probes(
    config,
    entries: Sequence[ProbeEntry],
    *,
    run_id: Optional[str] = None,
    force: bool = False,
    adapter: Optional[SDXLTurboAdapterPCA] = None,
    allow_synthetic_basis: bool = False,
) -> Path:
    """Renders and persists one probe manifest -- whichever of the three
    builders above ``entries`` came from -- mirroring
    ``runner.generate_manifest``'s exact resume/immutability/per-entry loop
    (an already-complete sample dir is skipped, an incomplete existing one
    raises rather than being overwritten, the run directory's provenance is
    pinned by ``ensure_immutable_run``), but keyed on ``ProbeEntry``/
    ``ProbeRealization`` fields instead of the PC-condition schema: probing
    never touches ``psd_editor``/``calibration`` at all, so there is no
    ``calibration_hash`` and no frozen-correction step here -- only the
    basis is required.
    """
    if not entries:
        raise ProbingError("generate_probes called with an empty manifest")
    run_dir = _probe_run_dir(config, run_id)
    basis_hash = probe_basis_hash_for(config)
    ensure_immutable_run(run_dir, _probe_run_provenance(config, basis_hash), force=force)

    height, width = config.generation.config.height, config.generation.config.width
    channels = config.basis.channels
    loaded_basis = load_basis(
        config.resolve_root(config.basis.basis_output_path),
        allow_synthetic=allow_synthetic_basis,
        expected_patch_size=config.basis.patch_size, expected_channels=channels,
    )

    adapter = adapter or SDXLTurboAdapterPCA(config.model.as_model_config_dict())
    for entry in entries:
        realization = realize_probe(entry, loaded_basis, channels=channels, height=height, width=width)
        latent = realization.output_latent
        final_hash = tensor_hash(latent[0])
        sample_dir = _probe_sample_dir(run_dir, entry)
        if _probe_is_complete(sample_dir, final_hash) and not force:
            continue
        if sample_dir.exists():
            raise ProbingError(f"Incomplete existing probe at {sample_dir}; refusing to overwrite source images")
        sample_dir.mkdir(parents=True, exist_ok=True)

        pair_key = (entry.prompt_id, entry.block_id, entry.base_index)
        images = adapter.generate(
            entry.prompt_text, latent, [pair_key], seed=config.run.master_seed,
            generation_config=config.generation.config,
        )
        if len(images) != 1:
            raise ProbingError(f"Adapter returned {len(images)} images; expected 1")

        target = sample_dir / "image.png"
        images[0].save(target, format="PNG")
        record = {
            "run_id": run_dir.name, "image_id": entry.image_id,
            "prompt_id": entry.prompt_id, "block_id": entry.block_id, "base_index": entry.base_index,
            "condition_type": entry.condition_type, "group_id": entry.group_id,
            "theta": entry.theta, "donor_seed": entry.donor_seed,
            "theoretical_delta_energy": realization.theoretical_delta_energy,
            "measured_delta_energy": realization.measured_delta_energy,
            "theoretical_relative_rms": realization.theoretical_relative_rms,
            "measured_relative_rms": realization.measured_relative_rms,
            "group_energy_ratio": realization.group_energy_ratio,
            "max_abs_error_vs_closed_form": realization.max_abs_error_vs_closed_form,
            "complement_and_boundary_unchanged": realization.complement_and_boundary_unchanged,
            "basis_hash": basis_hash,
            "final_noise_hash": final_hash,
            "adapter_prepared_latent_hash": adapter.last_generated_latent_hashes[0],
            "generator_seed": adapter.last_generator_seeds[0],
            "image_path": str(target.resolve()), "image_hash": file_hash(target),
            "generation_config_hash": sha256_text(config.model.as_model_config_dict(), config.generation.config, basis_hash),
        }
        write_jsonl(sample_dir / "sample.jsonl", [record])
    if config.generation.release_model_after_generation:
        adapter.close()
    return run_dir
