#!/usr/bin/env python3
"""Post-hoc frequency validation for completed white-floor noise-cache experiments.

Reconstructs every noise condition from the cached ``base_white``/``independent_eta``
tensors using the exact generation functions in ``noise_methods.py`` -- it never
samples replacement noise and never loads an image-generation model. Validates
empirical radial PSDs against the theoretical intervention PSDs, checks cached
final-noise hashes where available, and reports the baseline alpha with the
closest normalized log-PSD for every same-phase/independent-white condition.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from io_utils import condition_id, read_json, read_jsonl, tensor_hash
from noise_methods import NoiseBatch, construct_noise, noise_statistics, normalize, pink, radial_frequency_grid

ALPHAS = [round(0.1 * i, 1) for i in range(8)]
GAMMAS = [round(0.1 * i, 1) for i in range(1, 10)]
FAMILIES = ("same_phase", "independent_white")
# Resolved-config keys for the same two families -- the config section for "same_phase" is
# named "same_phase_floor" (matching run_experiment.py's YAML schema).
FAMILIES_CONFIG_KEYS = ("same_phase_floor", "independent_white")

PRIMARY_ALPHA_MIN, PRIMARY_ALPHA_MAX = 0.3, 0.7
PRIMARY_GAMMA_MAX = 0.6
CONTROL_ALPHA_MAX = 0.2
# Defaults used when --alphas/--gammas are not passed on the CLI.
DEFAULT_SELECTED_ALPHAS = (0.0, 0.3, 0.5, 0.7)
DEFAULT_SELECTED_GAMMAS = (0.1, 0.3, 0.6, 0.9)
NUM_RADIAL_BINS = 24
# Cycled by index (not looked up by value), so any --alphas/--gammas selection --
# a range, a single value, or an arbitrary discrete list -- gets a distinct style.
LINESTYLE_CYCLE = ("solid", (0, (4, 1.5)), (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1, 1, 1, 1, 1)), (0, (1, 1, 3, 1)))
SAME_PHASE_EXACT_TOLERANCE = 1e-3
PSD_SHAPE_TOLERANCE = 0.35
EPS = 1e-12
# SHA-256 final-noise hashes are not bit-reproducible across FFT library builds
# (confirmed: reconstructed std/l2_norm/psd_mean agree with the recorded
# noise_statistics.json to ~1e-7, i.e. float32 precision, even when the hash
# differs) -- this tolerance distinguishes that from an actual formula error.
NUMERIC_HASH_TOLERANCE = 1e-4
STATS_KEYS_FOR_NUMERIC_CHECK = ("std", "l2_norm", "psd_mean", "psd_high_frequency_mean")


@dataclass(frozen=True)
class ConditionSpec:
    family: str
    alpha: float
    gamma: float | None

    @property
    def identifier(self) -> str:
        return condition_id(self.family, self.alpha, self.gamma)

    @property
    def gamma_for_theory(self) -> float:
        return 0.0 if self.gamma is None else self.gamma

    @property
    def region(self) -> str:
        if self.family == "baseline":
            return "baseline"
        if self.alpha < CONTROL_ALPHA_MAX + 1e-9:
            return "control_alpha"
        if self.gamma is not None and self.gamma > PRIMARY_GAMMA_MAX + 1e-9:
            return "high_gamma_endpoint"
        return "primary"


def full_condition_grid(baseline_alphas: list[float], proposed_alphas: list[float],
                         proposed_gammas: list[float]) -> list[ConditionSpec]:
    conditions = [ConditionSpec("baseline", alpha, None) for alpha in baseline_alphas]
    for family in FAMILIES:
        conditions += [ConditionSpec(family, alpha, gamma) for alpha in proposed_alphas for gamma in proposed_gammas]
    return conditions


# --------------------------------------------------------------------------- #
# Cache / run discovery
# --------------------------------------------------------------------------- #

def _linestyle_for(index: int) -> Any:
    return LINESTYLE_CYCLE[index % len(LINESTYLE_CYCLE)]


def _format_grid_value(value: float) -> str:
    # Some runs use a finer grid (e.g. gamma=0.0125) where "%.1f" would collapse distinct values
    # to the same label -- strip trailing zeros instead of assuming one fixed decimal width.
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _parse_value_spec(spec: str, grid: list[float], flag_name: str) -> tuple[float, ...]:
    """Accepts an inclusive 'MIN:MAX' range, a single value, or a comma-separated list of
    discrete values -- always resolved against ``grid`` (this run's actual alpha/gamma grid),
    never interpolated."""
    if ":" in spec:
        low_text, high_text = spec.split(":", 1)
        low, high = float(low_text), float(high_text)
        selected = tuple(value for value in grid if low - 1e-9 <= value <= high + 1e-9)
    else:
        requested = [float(token) for token in spec.split(",") if token.strip()]
        missing = [req for req in requested if not any(abs(value - req) < 1e-9 for value in grid)]
        if missing:
            raise ValueError(f"--{flag_name}={spec} requests values not in the fixed grid {grid}: {missing}")
        selected = tuple(value for value in grid if any(abs(value - req) < 1e-9 for req in requested))
    if not selected:
        raise ValueError(f"--{flag_name}={spec} selects no values from the fixed grid {grid}")
    return selected


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def find_run_dir(run_id: str) -> Path:
    run_dir = script_dir() / "outputs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No such run directory: {run_dir}")
    return run_dir


def _run_resolved_config(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing run manifest, cannot recover the run configuration: {manifest_path}")
    return read_json(manifest_path)["resolved_config"]


def normalization_profile_for(run_dir: Path) -> str:
    return _run_resolved_config(run_dir)["experiment"]["normalization_profile"]


def load_condition_grids(run_dir: Path) -> tuple[list[float], list[float], list[float]]:
    """Reads this run's actual alpha/gamma grids from its manifest instead of assuming the
    standard 0.1-step sweep -- a 'finer' config (see configs/sdxl_turbo_full_finer.yaml) can use
    an entirely different, non-uniform alpha/gamma grid for the proposed methods only."""
    config = _run_resolved_config(run_dir)
    baseline_alphas = sorted(round(v, 6) for v in config["baseline"]["alpha_values"])
    proposed_alphas = sorted({round(v, 6) for family in FAMILIES_CONFIG_KEYS for v in config[family]["alpha_values"]})
    proposed_gammas = sorted({round(v, 6) for family in FAMILIES_CONFIG_KEYS for v in config[family]["gamma_values"]})
    return baseline_alphas, proposed_alphas, proposed_gammas


def _defaults_or_full(preferred: tuple[float, ...], grid: list[float]) -> tuple[float, ...]:
    """Uses the curated ``preferred`` selection where at least two of its values exist in this
    run's actual grid (the common case: the standard 0.1-step sweep). Otherwise -- e.g. a finer
    grid sharing at most one point with ``preferred`` -- shows the run's full (small) grid rather
    than a near-empty, effectively arbitrary selection."""
    filtered = tuple(v for v in preferred if v in grid)
    return filtered if len(filtered) >= 2 else tuple(sorted(grid))


def discover_blocks(run_dir: Path) -> list[str]:
    cache_root = run_dir / "noise_cache"
    if not cache_root.is_dir():
        raise RuntimeError(f"No noise cache found: {cache_root}")
    blocks = [p.name for p in sorted(cache_root.iterdir())
              if (p / "base_white.pt").exists() and (p / "independent_eta.pt").exists() and (p / "noise_metadata.json").exists()]
    if not blocks:
        raise RuntimeError(f"No usable cache blocks under {cache_root}")
    return blocks


def load_block_batch(run_dir: Path, block_id: str) -> tuple[NoiseBatch, bool]:
    cache_dir = run_dir / "noise_cache" / block_id
    base = torch.load(cache_dir / "base_white.pt", map_location="cpu", weights_only=True).to(torch.float32)
    eta = torch.load(cache_dir / "independent_eta.pt", map_location="cpu", weights_only=True).to(torch.float32)
    metadata = read_json(cache_dir / "noise_metadata.json")
    batch = NoiseBatch(base, eta, metadata["sample_seeds"], metadata["eta_sample_seeds"], metadata["base_hashes"], metadata["eta_hashes"])
    integrity_ok = ([tensor_hash(x) for x in base] == batch.base_hashes) and ([tensor_hash(x) for x in eta] == batch.eta_hashes)
    return batch, integrity_ok


def find_model_dir(run_dir: Path) -> Path | None:
    gen_root = run_dir / "generations"
    if not gen_root.is_dir():
        return None
    candidates = [p for p in gen_root.iterdir() if p.is_dir()]
    if len(candidates) > 1:
        raise RuntimeError(f"Ambiguous model directory under {gen_root}: {[p.name for p in candidates]} "
                            "-- pass a run with a single generation model, or extend this script to disambiguate")
    return candidates[0] if candidates else None


def load_final_hashes(model_dir: Path | None, block_id: str, condition: ConditionSpec) -> dict[int, str] | None:
    if model_dir is None:
        return None
    samples_path = model_dir / block_id / condition.identifier / "samples.jsonl"
    if not samples_path.exists():
        return None
    rows = read_jsonl(samples_path)
    if not rows:
        return None
    return {int(row["base_index"]): row["final_noise_hash"] for row in rows}


def load_saved_post_norm_stats(model_dir: Path | None, block_id: str, condition: ConditionSpec) -> list[dict[str, float]] | None:
    if model_dir is None:
        return None
    stats_path = model_dir / block_id / condition.identifier / "noise_statistics.json"
    if not stats_path.exists():
        return None
    return read_json(stats_path)["post_normalization"]


def stats_relative_error(saved_rows: list[dict[str, float]], mine_rows: list[dict[str, float]]) -> float:
    """Max relative error on scale-sensitive stats; excludes mean/psd_dc, which are ~0 after
    normalization and make relative error meaningless (dividing near-zero by near-zero)."""
    worst = 0.0
    for saved, mine in zip(saved_rows, mine_rows):
        for key in STATS_KEYS_FOR_NUMERIC_CHECK:
            worst = max(worst, abs(saved[key] - mine[key]) / max(abs(saved[key]), 1e-6))
    return worst


# --------------------------------------------------------------------------- #
# Frequency-domain helpers
# --------------------------------------------------------------------------- #

def radial_bin_layout(height: int, width: int, num_bins: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = radial_frequency_grid(height, width)
    r_max = float(r.max())
    edges = torch.linspace(0.0, r_max + 1e-6, num_bins + 1)
    bin_index_flat = torch.bucketize(r.reshape(-1), edges[1:-1], right=False)
    counts = torch.zeros(num_bins, dtype=torch.float64).index_add_(0, bin_index_flat, torch.ones_like(bin_index_flat, dtype=torch.float64))
    return r, bin_index_flat, counts


def radial_shell_mean(power: torch.Tensor, bin_index_flat: torch.Tensor, counts: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Mean power within each radial shell; power's last two dims are (H, Wf)."""
    flat = power.reshape(*power.shape[:-2], -1).to(torch.float64)
    out = torch.zeros(*power.shape[:-2], num_bins, dtype=torch.float64)
    out.index_add_(-1, bin_index_flat, flat)
    return out / counts.clamp(min=1.0)


def theoretical_shell_curve(family: str, alpha: float, gamma: float, r: torch.Tensor, bin_index_flat: torch.Tensor,
                             counts: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Theoretical expected (unnormalized) PSD shape, binned identically to the empirical curves. Valid for
    baseline/same_phase, whose closed-form multiplier applies deterministically to base_white's own spectrum
    (no independent random component involved). Independent-white mixes in an independently-drawn eta, so its
    theoretical curve is instead built from the real pink/eta component spectra -- see
    ``independent_white_theoretical_shell`` below."""
    h_squared = (1.0 + r).pow(-2.0 * alpha).to(torch.float64)
    theoretical = h_squared if family == "baseline" else (1.0 - gamma) * h_squared + gamma
    return radial_shell_mean(theoretical.unsqueeze(0).unsqueeze(0), bin_index_flat, counts, num_bins)[0, 0]


def independent_white_theoretical_shell(acc: "Accumulator", gamma: float) -> torch.Tensor:
    """(1-gamma)*E[pink power] + gamma*E[eta power], from the actual raw pink/eta components ``construct_noise``
    mixed for this condition's block(s) -- not assumed from the closed form. ``noise_methods.independent_white``
    never normalizes pink/eta before mixing, so this is expected to (and empirically does) match the closed-form
    ``(1-gamma)*H^2+gamma``, but deriving it from the real components makes that agreement a checked fact rather
    than an assumption baked into the validator itself."""
    n = max(acc.sample_count, 1)
    return (1.0 - gamma) * (acc.raw_pink_shell_power_sum / n) + gamma * (acc.raw_eta_shell_power_sum / n)


def normalize_log_curve(shell_mean: torch.Tensor) -> torch.Tensor:
    normalized = shell_mean / shell_mean.sum().clamp(min=EPS)
    return torch.log(normalized.clamp(min=EPS))


def frequency_column_weights(width: int, width_freq: int) -> torch.Tensor:
    """rfft2's half-spectrum omits the conjugate-mirror columns for 0 < fx < W/2; weight the DC and
    (even-width) Nyquist columns by 1 and every other column by 2 so column sums approximate the true
    full-spectrum energy instead of over-crediting the DC/Nyquist edge relative to the doubled interior."""
    weights = torch.full((width_freq,), 2.0, dtype=torch.float64)
    weights[0] = 1.0
    if width % 2 == 0:
        weights[-1] = 1.0
    return weights


def band_energy_fractions(power: torch.Tensor, r: torch.Tensor, column_weights: torch.Tensor
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r_max = float(r.max())
    flat_r = r.reshape(-1)
    weighted = (power.to(torch.float64) * column_weights.view(1, -1)).reshape(*power.shape[:-2], -1)
    total = weighted.sum(dim=-1).clamp(min=EPS)
    low_mask, mid_mask = flat_r <= r_max / 3.0, (flat_r > r_max / 3.0) & (flat_r <= 2.0 * r_max / 3.0)
    high_mask = flat_r > 2.0 * r_max / 3.0
    return (weighted[..., low_mask].sum(dim=-1) / total, weighted[..., mid_mask].sum(dim=-1) / total,
            weighted[..., high_mask].sum(dim=-1) / total)


def cosine_similarity(z: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    zf, rf = z.reshape(*z.shape[:2], -1).to(torch.float64), ref.reshape(*ref.shape[:2], -1).to(torch.float64)
    return (zf * rf).sum(dim=-1) / (zf.norm(dim=-1) * rf.norm(dim=-1)).clamp(min=EPS)


def same_phase_exact_ratio_error(raw: torch.Tensor, base_white: torch.Tensor, alpha: float, gamma: float, r: torch.Tensor) -> float:
    """Same-phase preserves Fourier phase, so |z_hat|^2/|eps_hat|^2 must equal G(r)^2 pointwise -- an exact,
    per-sample check that needs no ensemble averaging (unlike independent-white's random cross term). Takes the
    actual reconstructed ``raw`` from ``construct_noise`` (not a locally recomputed ``same_phase_floor``) so this
    check also catches a wrong family/gamma/normalization dispatch inside ``construct_noise`` itself."""
    eps_hat = torch.fft.rfft2(base_white, dim=(-2, -1))
    z_hat = torch.fft.rfft2(raw, dim=(-2, -1))
    mask = eps_hat.abs() > 1e-4
    if not bool(mask.any()):
        return 0.0
    ratio = z_hat[mask].abs().square().to(torch.float64) / eps_hat[mask].abs().square().to(torch.float64)
    theoretical = ((1.0 - gamma) * (1.0 + r).pow(-2.0 * alpha) + gamma).to(torch.float64)
    theoretical = theoretical.unsqueeze(0).unsqueeze(0).expand_as(eps_hat.abs())[mask]
    return float((ratio - theoretical).abs().max())


# --------------------------------------------------------------------------- #
# Per-condition accumulation across blocks
# --------------------------------------------------------------------------- #

@dataclass
class Accumulator:
    shell_power_sum: torch.Tensor
    raw_shell_power_sum: torch.Tensor
    raw_pink_shell_power_sum: torch.Tensor
    raw_eta_shell_power_sum: torch.Tensor
    sample_count: int = 0
    mean_sum: float = 0.0
    std_sum: float = 0.0
    l2_sum: float = 0.0
    low_sum: float = 0.0
    mid_sum: float = 0.0
    high_sum: float = 0.0
    cosine_sum: float = 0.0
    cross_sum: torch.Tensor | None = None
    sxx_sum: torch.Tensor | None = None
    syy_sum: torch.Tensor | None = None
    hash_checked: int = 0
    hash_matched: int = 0
    numeric_checked: int = 0
    numeric_matched: int = 0
    same_phase_exact_ok: bool = True
    same_phase_max_error: float = 0.0
    block_ids: list[str] = field(default_factory=list)


def new_accumulator(num_bins: int, height: int, width_freq: int) -> Accumulator:
    return Accumulator(shell_power_sum=torch.zeros(num_bins, dtype=torch.float64),
                        raw_shell_power_sum=torch.zeros(num_bins, dtype=torch.float64),
                        raw_pink_shell_power_sum=torch.zeros(num_bins, dtype=torch.float64),
                        raw_eta_shell_power_sum=torch.zeros(num_bins, dtype=torch.float64),
                        cross_sum=torch.zeros(height, width_freq, dtype=torch.complex128),
                        sxx_sum=torch.zeros(height, width_freq, dtype=torch.float64),
                        syy_sum=torch.zeros(height, width_freq, dtype=torch.float64))


def accumulate_block(acc: Accumulator, block_id: str, final: torch.Tensor, base_reference: torch.Tensor, r: torch.Tensor,
                      bin_index_flat: torch.Tensor, counts: torch.Tensor, num_bins: int, column_weights: torch.Tensor) -> None:
    """Accumulates real spectral/statistical analysis for the actual model-input tensor (``final``, i.e.
    post-normalization). ``base_reference`` must be normalized the same way (``normalize(base_white, profile)``),
    not the raw cached draw -- otherwise cosine/coherence pick up a spurious offset from comparing a zero-mean
    unit-std tensor against a reference with a different mean/std, which is largest exactly where it matters most:
    the alpha=0 same-phase sanity check, where final IS base_reference and should read cosine/coherence == 1 exactly."""
    b, c = final.shape[0], final.shape[1]
    n = b * c
    z_hat = torch.fft.rfft2(final, dim=(-2, -1))
    e_hat = torch.fft.rfft2(base_reference, dim=(-2, -1))
    power = z_hat.abs().square()
    acc.shell_power_sum += radial_shell_mean(power, bin_index_flat, counts, num_bins).sum(dim=(0, 1))
    low, mid, high = band_energy_fractions(power, r, column_weights)
    acc.low_sum += float(low.sum()); acc.mid_sum += float(mid.sum()); acc.high_sum += float(high.sum())
    flat = final.reshape(b, c, -1).to(torch.float64)
    acc.mean_sum += float(flat.mean(dim=-1).sum())
    acc.std_sum += float(flat.std(dim=-1, unbiased=False).sum())
    acc.l2_sum += float(torch.linalg.vector_norm(flat, dim=-1).sum())
    acc.cosine_sum += float(cosine_similarity(final, base_reference).sum())
    acc.cross_sum += (z_hat * e_hat.conj()).sum(dim=(0, 1)).to(torch.complex128)
    acc.sxx_sum += power.sum(dim=(0, 1)).to(torch.float64)
    acc.syy_sum += e_hat.abs().square().sum(dim=(0, 1)).to(torch.float64)
    acc.sample_count += n
    acc.block_ids.append(block_id)


def accumulate_raw_shell_power(acc: Accumulator, raw: torch.Tensor, bin_index_flat: torch.Tensor,
                                counts: torch.Tensor, num_bins: int) -> None:
    """Tracks the pre-normalization PSD separately from the final-noise PSD above, so the theoretical-formula
    check (which is derived pre-normalization) never gets compared against post-normalization statistics."""
    power = torch.fft.rfft2(raw, dim=(-2, -1)).abs().square()
    acc.raw_shell_power_sum += radial_shell_mean(power, bin_index_flat, counts, num_bins).sum(dim=(0, 1))


def accumulate_raw_component_shell_power(acc: Accumulator, pink_component: torch.Tensor, eta: torch.Tensor,
                                          bin_index_flat: torch.Tensor, counts: torch.Tensor, num_bins: int) -> None:
    """For independent-white only: tracks the raw pink and eta components' own PSDs, so the theoretical curve can
    be built from what ``construct_noise`` actually mixed (``noise_methods.independent_white`` never normalizes
    pink/eta before mixing, so this is expected to match the closed-form ``(1-gamma)*H^2+gamma`` formula, but
    deriving it from the real components makes that agreement a checked fact rather than an assumption)."""
    pink_power = torch.fft.rfft2(pink_component, dim=(-2, -1)).abs().square()
    eta_power = torch.fft.rfft2(eta, dim=(-2, -1)).abs().square()
    acc.raw_pink_shell_power_sum += radial_shell_mean(pink_power, bin_index_flat, counts, num_bins).sum(dim=(0, 1))
    acc.raw_eta_shell_power_sum += radial_shell_mean(eta_power, bin_index_flat, counts, num_bins).sum(dim=(0, 1))


def summarize(acc: Accumulator) -> dict[str, float]:
    n = max(acc.sample_count, 1)
    # Magnitude-squared coherence per frequency bin, |<Sxy>|^2 / (<Sxx><Syy>), bounded in [0, 1]
    # by Cauchy-Schwarz; averaged across bins weighted by their combined power so near-zero-power
    # bins (e.g. DC after zero-mean normalization) don't dominate with unstable 0/0 ratios.
    denom = (acc.sxx_sum * acc.syy_sum).clamp(min=EPS)
    coherence = float(acc.cross_sum.abs().square().sum() / denom.sum()) if float(acc.syy_sum.sum()) > 0 else 0.0
    return {"mean": acc.mean_sum / n, "std": acc.std_sum / n, "l2_norm": acc.l2_sum / n,
            "low_frac": acc.low_sum / n, "mid_frac": acc.mid_sum / n, "high_frac": acc.high_sum / n,
            "cosine_to_base": acc.cosine_sum / n, "coherence_to_base": coherence}


# --------------------------------------------------------------------------- #
# Main per-run validation
# --------------------------------------------------------------------------- #

def run_validation(run_id: str, alpha_spec: str | None = None, gamma_spec: str | None = None) -> dict[str, Any]:
    run_dir = find_run_dir(run_id)
    profile = normalization_profile_for(run_dir)
    blocks = discover_blocks(run_dir)
    model_dir = find_model_dir(run_dir)
    baseline_alphas, proposed_alphas, proposed_gammas = load_condition_grids(run_dir)
    conditions = full_condition_grid(baseline_alphas, proposed_alphas, proposed_gammas)

    alphas = (_parse_value_spec(alpha_spec, proposed_alphas, "alphas") if alpha_spec is not None
              else _defaults_or_full(DEFAULT_SELECTED_ALPHAS, proposed_alphas))
    gammas = (_parse_value_spec(gamma_spec, proposed_gammas, "gammas") if gamma_spec is not None
              else _defaults_or_full(DEFAULT_SELECTED_GAMMAS, proposed_gammas))

    batches: dict[str, NoiseBatch] = {}
    integrity_ok = True
    for block_id in blocks:
        batch, ok = load_block_batch(run_dir, block_id)
        batches[block_id] = batch
        integrity_ok = integrity_ok and ok
    shapes = {tuple(batch.base_white.shape) for batch in batches.values()}
    if len(shapes) != 1:
        raise RuntimeError(f"Cache blocks disagree on tensor shape: {shapes}")
    _, _, height, width = next(iter(shapes))
    width_freq = width // 2 + 1
    r, bin_index_flat, counts = radial_bin_layout(height, width, NUM_RADIAL_BINS)
    column_weights = frequency_column_weights(width, width_freq)

    accumulators: dict[ConditionSpec, Accumulator] = {c: new_accumulator(NUM_RADIAL_BINS, height, width_freq) for c in conditions}
    per_block_rows: list[dict[str, Any]] = []
    hash_totals = {"checked": 0, "matched": 0, "conditions_with_hashes": 0,
                   "numeric_checked": 0, "numeric_matched": 0}
    same_phase_condition_error: dict[ConditionSpec, float] = {}

    for block_id in blocks:
        batch = batches[block_id]
        # Same normalization as ``final`` below, applied to the reference draw: this makes cosine/coherence
        # compare like-for-like (both zero-mean unit-std) instead of picking up a spurious gap from comparing
        # a normalized tensor against a differently-scaled raw one -- most visibly at the alpha=0 same-phase
        # endpoint, where final IS base_reference and cosine/coherence should read exactly 1.
        base_reference = normalize(batch.base_white, profile)
        for condition in conditions:
            gamma = None if condition.family == "baseline" else condition.gamma
            # ``raw`` (pre-normalization) is used only to validate the intervention formula itself
            # (theoretical PSD / same-phase exactness); every statistic below analyzes ``final``, the
            # tensor actually fed to the model, since normalization is affine and materially changes
            # absolute-scale stats (mean/std/l2_norm) even though shape-based ratios are nearly invariant.
            raw = construct_noise(batch, condition.family, condition.alpha, gamma, "none")
            final = normalize(raw, profile)
            acc = accumulators[condition]
            accumulate_block(acc, block_id, final, base_reference, r, bin_index_flat, counts, NUM_RADIAL_BINS, column_weights)
            accumulate_raw_shell_power(acc, raw, bin_index_flat, counts, NUM_RADIAL_BINS)
            if condition.family == "independent_white":
                accumulate_raw_component_shell_power(acc, pink(batch.base_white, condition.alpha), batch.independent_eta,
                                                      bin_index_flat, counts, NUM_RADIAL_BINS)

            hash_matched = hash_checked = 0
            numeric_matched = 0
            final_hashes = load_final_hashes(model_dir, block_id, condition)
            if final_hashes is not None:
                for index, expected in final_hashes.items():
                    hash_checked += 1
                    if tensor_hash(final[index]) == expected:
                        hash_matched += 1
                acc.hash_checked += hash_checked
                acc.hash_matched += hash_matched
                hash_totals["checked"] += hash_checked
                hash_totals["matched"] += hash_matched
                hash_totals["conditions_with_hashes"] += 1

                saved_stats = load_saved_post_norm_stats(model_dir, block_id, condition)
                if saved_stats is not None:
                    numeric_matched = int(stats_relative_error(saved_stats, noise_statistics(final)) < NUMERIC_HASH_TOLERANCE)
                    acc.numeric_checked += 1
                    acc.numeric_matched += numeric_matched
                    hash_totals["numeric_checked"] += 1
                    hash_totals["numeric_matched"] += numeric_matched

            same_phase_error = None
            if condition.family == "same_phase" and condition.gamma is not None:
                same_phase_error = same_phase_exact_ratio_error(raw, batch.base_white, condition.alpha, condition.gamma, r)
                acc.same_phase_max_error = max(acc.same_phase_max_error, same_phase_error)
                acc.same_phase_exact_ok = acc.same_phase_exact_ok and same_phase_error < SAME_PHASE_EXACT_TOLERANCE
                same_phase_condition_error[condition] = acc.same_phase_max_error

            b, c = final.shape[0], final.shape[1]
            power = torch.fft.rfft2(final, dim=(-2, -1)).abs().square()
            low, mid, high = band_energy_fractions(power, r, column_weights)
            flat = final.reshape(b, c, -1).to(torch.float64)
            per_block_rows.append({
                "run_id": run_id, "block_id": block_id, "family": condition.family, "alpha": condition.alpha,
                "gamma": condition.gamma_for_theory if condition.family != "baseline" else "", "region": condition.region,
                "mean": float(flat.mean(dim=-1).mean()), "std": float(flat.std(dim=-1, unbiased=False).mean()),
                "l2_norm": float(torch.linalg.vector_norm(flat, dim=-1).mean()),
                "low_frac": float(low.mean()), "mid_frac": float(mid.mean()), "high_frac": float(high.mean()),
                "cosine_to_base": float(cosine_similarity(final, base_reference).mean()),
                "same_phase_exact_max_error": same_phase_error if same_phase_error is not None else "",
                "hash_available": final_hashes is not None,
                "hash_matched": hash_matched, "hash_checked": hash_checked,
                "numeric_matched": numeric_matched,
            })

    aggregate_rows, equivalent_alpha_map, coherence_map, psd_error_map = _aggregate_conditions(
        conditions, accumulators, run_id, r, bin_index_flat, counts)

    csv_rows = per_block_rows + aggregate_rows
    out_dir = run_dir / "analysis" / "frequency_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "noise_frequency_summary.csv", csv_rows)

    r_max = float(r.max())
    curves = _condition_shell_curves(conditions, accumulators)
    _plot_radial_psd_selected(out_dir / "radial_psd_selected.png", curves, run_id, r_max, alphas, gammas)
    _plot_theoretical_vs_empirical(out_dir / "theoretical_vs_empirical_psd.png", conditions, accumulators, r,
                                    bin_index_flat, counts, run_id, r_max, alphas, gammas)
    _plot_equivalent_alpha_heatmap(out_dir / "equivalent_alpha_heatmap.png", equivalent_alpha_map, run_id,
                                    list(alphas), list(gammas), baseline_alphas)
    _plot_coherence_heatmap(out_dir / "coherence_heatmap.png", coherence_map, run_id, list(alphas), list(gammas))

    return {
        "run_id": run_id, "run_dir": run_dir, "num_blocks": len(blocks), "num_conditions": len(conditions),
        "cache_integrity_ok": integrity_ok, "hash_totals": hash_totals, "out_dir": out_dir,
        "psd_error_map": psd_error_map,
    }


def _aggregate_conditions(conditions: list[ConditionSpec], accumulators: dict[ConditionSpec, Accumulator], run_id: str,
                           r: torch.Tensor, bin_index_flat: torch.Tensor, counts: torch.Tensor
                           ) -> tuple[list[dict[str, Any]], dict[tuple[str, float, float], float],
                                      dict[tuple[str, float, float], float], dict[ConditionSpec, float]]:
    baseline_curves = {c.alpha: normalize_log_curve(accumulators[c].shell_power_sum / max(accumulators[c].sample_count, 1))
                       for c in conditions if c.family == "baseline"}
    rows: list[dict[str, Any]] = []
    equivalent_alpha_map: dict[tuple[str, float, float], float] = {}
    coherence_map: dict[tuple[str, float, float], float] = {}
    psd_error_map: dict[ConditionSpec, float] = {}

    for condition in conditions:
        acc = accumulators[condition]
        summary = summarize(acc)
        # equivalent_alpha compares like-for-like: the actual model-input (final) PSD shape against the
        # final-based baseline sweep above.
        empirical_curve = normalize_log_curve(acc.shell_power_sum / max(acc.sample_count, 1))
        # psd_log_error validates the intervention *formula*, so both sides must stay pre-normalization: the
        # raw empirical PSD against the raw theoretical prediction (closed-form, or -- for independent-white --
        # derived from the real pink/eta components).
        raw_empirical_curve = normalize_log_curve(acc.raw_shell_power_sum / max(acc.sample_count, 1))
        if condition.family == "independent_white":
            theoretical_shell = independent_white_theoretical_shell(acc, condition.gamma_for_theory)
        else:
            theoretical_shell = theoretical_shell_curve(condition.family, condition.alpha, condition.gamma_for_theory, r,
                                                         bin_index_flat, counts, NUM_RADIAL_BINS)
        theoretical_curve = normalize_log_curve(theoretical_shell)
        valid = counts > 0
        psd_log_error = float((raw_empirical_curve[valid] - theoretical_curve[valid]).abs().mean())
        psd_error_map[condition] = psd_log_error

        equivalent_alpha = None
        if condition.family != "baseline":
            distances = {alpha: float(((empirical_curve[valid] - curve[valid]) ** 2).sum())
                         for alpha, curve in baseline_curves.items()}
            equivalent_alpha = min(distances, key=distances.get)
            equivalent_alpha_map[(condition.family, condition.alpha, condition.gamma_for_theory)] = equivalent_alpha
        coherence_map[(condition.family, condition.alpha, condition.gamma_for_theory)] = summary["coherence_to_base"]

        rows.append({
            "run_id": run_id, "block_id": "ALL_BLOCKS", "family": condition.family, "alpha": condition.alpha,
            "gamma": condition.gamma_for_theory if condition.family != "baseline" else "", "region": condition.region,
            "mean": summary["mean"], "std": summary["std"], "l2_norm": summary["l2_norm"],
            "low_frac": summary["low_frac"], "mid_frac": summary["mid_frac"], "high_frac": summary["high_frac"],
            "cosine_to_base": summary["cosine_to_base"], "coherence_to_base": summary["coherence_to_base"],
            "psd_log_error": psd_log_error, "psd_shape_match": psd_log_error < PSD_SHAPE_TOLERANCE,
            "equivalent_alpha": equivalent_alpha if equivalent_alpha is not None else "",
            "same_phase_exact_match": acc.same_phase_exact_ok if condition.family == "same_phase" else "",
            "same_phase_exact_max_error": acc.same_phase_max_error if condition.family == "same_phase" else "",
            "hash_available": acc.hash_checked > 0,
            "hash_matched": acc.hash_matched, "hash_checked": acc.hash_checked,
            "numeric_matched": acc.numeric_matched, "numeric_checked": acc.numeric_checked,
        })
    return rows, equivalent_alpha_map, coherence_map, psd_error_map


def _condition_shell_curves(conditions: list[ConditionSpec], accumulators: dict[ConditionSpec, Accumulator]
                             ) -> dict[ConditionSpec, torch.Tensor]:
    return {c: normalize_log_curve(accumulators[c].shell_power_sum / max(accumulators[c].sample_count, 1)) for c in conditions}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def _shell_centers(num_bins: int, r_max: float) -> np.ndarray:
    edges = np.linspace(0.0, r_max + 1e-6, num_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _save_figure(figure, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=220, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(figure)


def _plot_radial_psd_selected(target: Path, curves: dict[ConditionSpec, torch.Tensor], run_id: str, r_max: float,
                               alphas: tuple[float, ...], gammas: tuple[float, ...]) -> None:
    import matplotlib.pyplot as plt
    conditions = list(curves)
    num_bins = len(next(iter(curves.values())))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    cmap = plt.colormaps["viridis"]
    x = _shell_centers(num_bins, r_max)
    for axis, family in zip(axes, FAMILIES):
        baseline_conditions = sorted([c for c in conditions if c.family == "baseline"], key=lambda c: c.alpha)
        for index, bc in enumerate(baseline_conditions):
            axis.plot(x, curves[bc].numpy(), color="0.75", linewidth=0.8, zorder=1,
                      label="baseline (all α)" if index == 0 else None)
        # Only the two extreme baseline curves (alpha=0 and alpha=max) are labelled directly on the
        # plot -- the intermediate ones are visually ordered between them and don't need their own tag.
        for bc in (baseline_conditions[0], baseline_conditions[-1]):
            axis.annotate(f"α={_format_grid_value(bc.alpha)}", xy=(x[-1], curves[bc].numpy()[-1]),
                          xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.5, color="0.4",
                          annotation_clip=False)
        for alpha in alphas:
            for gamma_index, gamma in enumerate(gammas):
                match = next((c for c in conditions if c.family == family and c.alpha == alpha and c.gamma == gamma), None)
                if match is None:
                    continue
                color = cmap(0.1 + 0.8 * alphas.index(alpha) / max(len(alphas) - 1, 1))
                axis.plot(x, curves[match].numpy(), color=color, linestyle=_linestyle_for(gamma_index), linewidth=1.6,
                          label=f"α={_format_grid_value(alpha)}, γ={_format_grid_value(gamma)}", zorder=2)
        axis.set(xlabel="Radial frequency r (FFT-bin units)", ylabel="log normalized PSD" if family == FAMILIES[0] else "",
                 title="Same-phase" if family == "same_phase" else "Independent-white")
        axis.grid(alpha=0.2)
        axis.margins(x=0.14)
        axis.legend(fontsize=6, ncol=2, frameon=False)
    figure.suptitle(f"Representative radial PSD curves ({run_id}); grey lines are the baseline alpha sweep")
    _save_figure(figure, target)


def _plot_theoretical_vs_empirical(target: Path, conditions: list[ConditionSpec], accumulators: dict[ConditionSpec, Accumulator],
                                    r: torch.Tensor, bin_index_flat: torch.Tensor, counts: torch.Tensor, run_id: str,
                                    r_max: float, alphas: tuple[float, ...], gammas: tuple[float, ...]) -> None:
    # Overlays every requested alpha x gamma pair in one figure per family: color encodes alpha,
    # linestyle (LINESTYLE_CYCLE) encodes gamma, and empirical vs theoretical is distinguished by
    # weight/opacity only (empirical = full weight, theoretical = thinner and faded) so linestyle
    # stays free to encode gamma. Only the empirical curve carries a legend label -- the faded
    # theoretical curves reuse each empirical curve's own color, so a single generic swatch could
    # never represent them; that distinction is spelled out in the suptitle instead.
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    cmap = plt.colormaps["plasma"]
    x = _shell_centers(NUM_RADIAL_BINS, r_max)
    for axis, family in zip(axes, FAMILIES):
        for alpha_index, alpha in enumerate(alphas):
            color = cmap(0.1 + 0.8 * alpha_index / max(len(alphas) - 1, 1))
            for gamma_index, gamma in enumerate(gammas):
                match = next((c for c in conditions if c.family == family and c.alpha == alpha and c.gamma == gamma), None)
                if match is None:
                    continue
                acc = accumulators[match]
                empirical = normalize_log_curve(acc.raw_shell_power_sum / max(acc.sample_count, 1)).numpy()
                if family == "independent_white":
                    theoretical_shell = independent_white_theoretical_shell(acc, gamma)
                else:
                    theoretical_shell = theoretical_shell_curve(family, alpha, gamma, r, bin_index_flat,
                                                                 counts, NUM_RADIAL_BINS)
                theoretical = normalize_log_curve(theoretical_shell).numpy()
                linestyle = _linestyle_for(gamma_index)
                axis.plot(x, empirical, color=color, linestyle=linestyle, linewidth=1.6, alpha=1.0,
                          label=f"α={_format_grid_value(alpha)}, γ={_format_grid_value(gamma)}")
                axis.plot(x, theoretical, color=color, linestyle=linestyle, linewidth=0.9, alpha=0.45)
        axis.set(xlabel="Radial frequency r (FFT-bin units)", ylabel="log normalized PSD" if family == FAMILIES[0] else "",
                 title="Same-phase" if family == "same_phase" else "Independent-white")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6, ncol=2, frameon=False)
    figure.suptitle(f"Theoretical vs empirical PSD shape, pre-normalization ({run_id}); "
                     f"color=α, linestyle=γ, solid full-weight=empirical, faded thin=theory")
    _save_figure(figure, target)


def _grid_matrix(mapping: dict[tuple[str, float, float], float], family: str,
                  alphas: list[float], gammas: list[float]) -> np.ndarray:
    return np.array([[mapping.get((family, alpha, gamma), np.nan) for gamma in gammas] for alpha in alphas])


def _draw_primary_region_box(axis, alphas: list[float], gammas: list[float]) -> None:
    # The pre-registered primary region (alpha 0.3-0.7, gamma<=0.6) is only meaningful when this
    # run's grid actually spans it -- a restricted grid (e.g. a finer alpha 0.6-0.9 sweep) may not
    # contain these reference points at all, so skip the box rather than raise or mis-draw it.
    if PRIMARY_ALPHA_MIN not in alphas or PRIMARY_ALPHA_MAX not in alphas or PRIMARY_GAMMA_MAX not in gammas:
        return
    from matplotlib.patches import Rectangle
    alpha_lo = alphas.index(PRIMARY_ALPHA_MIN) - 0.5
    alpha_hi = alphas.index(PRIMARY_ALPHA_MAX) + 0.5
    gamma_hi = gammas.index(PRIMARY_GAMMA_MAX) + 0.5
    axis.add_patch(Rectangle((-0.5, alpha_lo), gamma_hi + 0.5, alpha_hi - alpha_lo,
                              fill=False, edgecolor="white", linewidth=1.8, zorder=5))


def _heatmap_panel(axis, matrix: np.ndarray, title: str, cmap: str, vmin: float | None, vmax: float | None,
                    alphas: list[float], gammas: list[float]):
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    axis.set_xticks(range(len(gammas))); axis.set_xticklabels([_format_grid_value(g) for g in gammas], fontsize=7)
    axis.set_yticks(range(len(alphas))); axis.set_yticklabels([_format_grid_value(a) for a in alphas], fontsize=7)
    axis.set(xlabel="γ", ylabel="α", title=title)
    _draw_primary_region_box(axis, alphas, gammas)
    return image


def _plot_equivalent_alpha_heatmap(target: Path, mapping: dict[tuple[str, float, float], float], run_id: str,
                                    alphas: list[float], gammas: list[float], baseline_alphas: list[float]) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, family in zip(axes, FAMILIES):
        matrix = _grid_matrix(mapping, family, alphas, gammas)
        image = _heatmap_panel(axis, matrix, "Same-phase" if family == "same_phase" else "Independent-white",
                                "viridis", min(baseline_alphas), max(baseline_alphas), alphas, gammas)
        figure.colorbar(image, ax=axis, label="equivalent baseline α", shrink=0.85)
    figure.suptitle(f"Equivalent baseline α by closest normalized log-PSD ({run_id}); white box = primary region")
    _save_figure(figure, target)


def _plot_coherence_heatmap(target: Path, mapping: dict[tuple[str, float, float], float], run_id: str,
                             alphas: list[float], gammas: list[float]) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, family in zip(axes, FAMILIES):
        matrix = _grid_matrix(mapping, family, alphas, gammas)
        image = _heatmap_panel(axis, matrix, "Same-phase" if family == "same_phase" else "Independent-white",
                                "viridis", 0.0, 1.0, alphas, gammas)
        figure.colorbar(image, ax=axis, label="Fourier coherence with base white", shrink=0.85)
    figure.suptitle(f"Coherence with base white noise ({run_id}); white box = primary region")
    _save_figure(figure, target)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--alphas", default=None,
                         help="Alphas to show in radial_psd_selected.png / theoretical_vs_empirical_psd.png: "
                              "a 'MIN:MAX' inclusive range, a single value, or a comma-separated list, resolved "
                              "against this run's actual alpha grid (read from its run_manifest.json -- may differ "
                              f"per run, e.g. a finer sweep). Default: {DEFAULT_SELECTED_ALPHAS} where present.")
    parser.add_argument("--gammas", default=None,
                         help="Gammas to show in the same two plots: a 'MIN:MAX' inclusive range, a single value, "
                              "or a comma-separated list, resolved against this run's actual gamma grid. "
                              f"Default: {DEFAULT_SELECTED_GAMMAS} where present.")
    args = parser.parse_args()

    result = run_validation(args.run_id, alpha_spec=args.alphas, gamma_spec=args.gammas)
    totals = result["hash_totals"]
    if totals["conditions_with_hashes"] == 0:
        hash_status = "no cached final-noise hashes found"
    else:
        hash_status = (f"{totals['matched']}/{totals['checked']} samples bit-exact across "
                        f"{totals['conditions_with_hashes']} conditions; "
                        f"{totals['numeric_matched']}/{totals['numeric_checked']} conditions numerically consistent "
                        f"(std/l2/PSD within {NUMERIC_HASH_TOLERANCE:g}) -- SHA-256 hashes are not bit-reproducible "
                        f"across FFT library builds, so a numeric mismatch (not a hash mismatch) is the real signal")
    print(f"run: {result['run_id']}")
    print(f"cache blocks: {result['num_blocks']} (integrity {'OK' if result['cache_integrity_ok'] else 'MISMATCH'})")
    print(f"reconstructed conditions: {result['num_conditions']}")
    print(f"final-noise hash check: {hash_status}")
    print(f"output directory: {result['out_dir']}")
    for name in ("noise_frequency_summary.csv", "radial_psd_selected.png", "theoretical_vs_empirical_psd.png",
                 "equivalent_alpha_heatmap.png", "coherence_heatmap.png"):
        print(f"  - {result['out_dir'] / name}")


if __name__ == "__main__":
    main()
