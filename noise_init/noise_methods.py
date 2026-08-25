"""Deterministic initial-noise construction.

``alpha`` is an amplitude-spectrum exponent: expected PSD of pink noise is
proportional to ``(1 + r)**(-2 * alpha)``. ``r`` uses integer FFT-bin index
units, matching DivGen: ``fftfreq(H) * H`` and ``rfftfreq(W) * W``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from io_utils import derived_seed, read_json, tensor_hash, write_json

EPSILON = 1e-8


@dataclass(frozen=True)
class NoiseBatch:
    base_white: torch.Tensor
    independent_eta: torch.Tensor
    sample_seeds: list[int]
    eta_sample_seeds: list[int]
    base_hashes: list[str]
    eta_hashes: list[str]


def radial_frequency_grid(height: int, width: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Radial frequency in DivGen's integer FFT-bin index units."""
    fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32).view(height, 1) * height
    fx = torch.fft.rfftfreq(width, device=device, dtype=torch.float32).view(1, width // 2 + 1) * width
    return torch.sqrt(fy.square() + fx.square())


def pink_filter(height: int, width: int, alpha: float, *, device: torch.device | str = "cpu") -> torch.Tensor:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return (1.0 + radial_frequency_grid(height, width, device=device)).pow(-float(alpha))


def _sample_seeded(shape: tuple[int, int, int, int], seeds: list[int]) -> torch.Tensor:
    _, channels, height, width = shape
    return torch.cat(
        [torch.randn((1, channels, height, width), generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32)
         for seed in seeds], dim=0
    )


def sample_noise_batch(master_seed: int, block_id: str, shape: tuple[int, int, int, int], *, batch_seed: int | None = None) -> NoiseBatch:
    batch_size = shape[0]
    # The experimental contract fixes the four base draws to s_b + i.  The
    # manifest stores s_b explicitly; the deterministic fallback is useful only
    # for isolated library callers/tests that do not have a manifest record.
    root_seed = derived_seed(master_seed, block_id, "base_white") if batch_seed is None else int(batch_seed)
    sample_seeds = [root_seed + index for index in range(batch_size)]
    eta_seeds = [derived_seed(master_seed, block_id, "independent_eta", index) for index in range(batch_size)]
    base, eta = _sample_seeded(shape, sample_seeds), _sample_seeded(shape, eta_seeds)
    return NoiseBatch(base, eta, sample_seeds, eta_seeds,
                      [tensor_hash(sample) for sample in base], [tensor_hash(sample) for sample in eta])


def normalize(latents: torch.Tensor, profile: str, eps: float = EPSILON) -> torch.Tensor:
    if profile == "none":
        return latents
    if profile == "per_sample_per_channel_zero_mean_unit_std":
        unbiased = False
    elif profile == "divgen_compat":
        # Matches DivGen and the white-vs-pink notebook's ``flat.std()``.
        unbiased = True
    else:
        raise ValueError(f"Unknown normalization profile: {profile}")
    flat = latents.reshape(latents.shape[0], latents.shape[1], -1)
    return ((flat - flat.mean(dim=-1, keepdim=True)) / (flat.std(dim=-1, keepdim=True, unbiased=unbiased) + eps)).reshape_as(latents)


def pink(base_white: torch.Tensor, alpha: float) -> torch.Tensor:
    if base_white.ndim != 4:
        raise ValueError("Noise must have shape (batch, channels, height, width)")
    # The white endpoint must be an exact reference to the saved base tensor,
    # not an approximately identity FFT/IFFT round trip.
    if alpha == 0.0:
        return base_white.clone()
    multiplier = pink_filter(base_white.shape[-2], base_white.shape[-1], alpha, device=base_white.device)
    return torch.fft.irfft2(torch.fft.rfft2(base_white, dim=(-2, -1)) * multiplier, s=base_white.shape[-2:], dim=(-2, -1))


def same_phase_floor(base_white: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    _validate_gamma(gamma)
    # Both identities are exact at the raw-noise level for every alpha/gamma
    # value listed here; preserve them before any FFT numerical round trip.
    if alpha == 0.0 or gamma == 1.0:
        return base_white.clone()
    h = pink_filter(base_white.shape[-2], base_white.shape[-1], alpha, device=base_white.device)
    multiplier = torch.sqrt((1.0 - gamma) * h.square() + gamma)
    return torch.fft.irfft2(torch.fft.rfft2(base_white, dim=(-2, -1)) * multiplier, s=base_white.shape[-2:], dim=(-2, -1))


def independent_white(base_white: torch.Tensor, eta: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    _validate_gamma(gamma)
    if base_white.shape != eta.shape:
        raise ValueError("epsilon and eta must have the same shape")
    if gamma == 0.0:
        return pink(base_white, alpha)
    if gamma == 1.0:
        return eta.clone()
    return (1.0 - gamma) ** 0.5 * pink(base_white, alpha) + gamma ** 0.5 * eta


def construct_noise(batch: NoiseBatch, method: str, alpha: float, gamma: float | None, normalization_profile: str) -> torch.Tensor:
    if method == "baseline":
        raw = pink(batch.base_white, alpha)
    elif method == "same_phase":
        raw = same_phase_floor(batch.base_white, alpha, _require_gamma(gamma))
    elif method == "independent_white":
        raw = independent_white(batch.base_white, batch.independent_eta, alpha, _require_gamma(gamma))
    else:
        raise ValueError(f"Unknown noise method: {method}")
    return normalize(raw, normalization_profile)


def noise_statistics(tensor: torch.Tensor) -> list[dict[str, float]]:
    """Per sample/channel provenance statistics, including a compact radial PSD proxy."""
    fft = torch.fft.rfft2(tensor.to(torch.float32), dim=(-2, -1))
    power = fft.abs().square()
    rows: list[dict[str, float]] = []
    for i in range(tensor.shape[0]):
        for c in range(tensor.shape[1]):
            field = tensor[i, c].to(torch.float32)
            rows.append({"base_index": i, "channel": c, "mean": float(field.mean()), "std": float(field.std(unbiased=False)),
                         "l2_norm": float(torch.linalg.vector_norm(field)), "psd_mean": float(power[i, c].mean()),
                         "psd_dc": float(power[i, c, 0, 0]), "psd_high_frequency_mean": float(power[i, c].reshape(-1)[1:].mean())})
    return rows


def cache_noise_batch(cache_dir: str | Path, batch: NoiseBatch, *, metadata: dict[str, Any]) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(batch.base_white, cache_dir / "base_white.pt")
    torch.save(batch.independent_eta, cache_dir / "independent_eta.pt")
    write_json(cache_dir / "noise_metadata.json", {**metadata, "sample_seeds": batch.sample_seeds,
               "eta_sample_seeds": batch.eta_sample_seeds, "base_hashes": batch.base_hashes, "eta_hashes": batch.eta_hashes})


def load_or_create_noise_batch(cache_dir: str | Path, master_seed: int, block_id: str, shape: tuple[int, int, int, int], *, batch_seed: int | None = None) -> NoiseBatch:
    cache_dir = Path(cache_dir)
    base_path, eta_path, metadata_path = cache_dir / "base_white.pt", cache_dir / "independent_eta.pt", cache_dir / "noise_metadata.json"
    if base_path.exists() and eta_path.exists() and metadata_path.exists():
        base, eta, metadata = torch.load(base_path, map_location="cpu", weights_only=True), torch.load(eta_path, map_location="cpu", weights_only=True), read_json(metadata_path)
        if tuple(base.shape) != shape or tuple(eta.shape) != shape:
            raise RuntimeError(f"Cached noise shape differs for {block_id}; create a new run id")
        batch = NoiseBatch(base, eta, metadata["sample_seeds"], metadata["eta_sample_seeds"], metadata["base_hashes"], metadata["eta_hashes"])
        if [tensor_hash(x) for x in base] != batch.base_hashes or [tensor_hash(x) for x in eta] != batch.eta_hashes:
            raise RuntimeError(f"Noise cache hash mismatch for {block_id}")
        return batch
    batch = sample_noise_batch(master_seed, block_id, shape, batch_seed=batch_seed)
    cache_noise_batch(cache_dir, batch, metadata={"block_id": block_id, "batch_seed": batch_seed, "shape": list(shape), "base_pre_normalization": noise_statistics(batch.base_white), "eta_pre_normalization": noise_statistics(batch.independent_eta)})
    return batch


def _validate_gamma(gamma: float) -> None:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")


def _require_gamma(gamma: float | None) -> float:
    if gamma is None:
        raise ValueError("gamma is required for this method")
    _validate_gamma(gamma)
    return gamma
