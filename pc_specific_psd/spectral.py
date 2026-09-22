"""Radial PSD estimation, transfer-matrix diagnostics, and Parseval checks.

Per plan §14.6: ``noise_init.noise_statistics``'s ``psd_mean``/
``psd_high_frequency_mean`` are NOT true annular/Hermitian-weighted PSD
measures and must not be reused to validate this method's radial-PSD
matching. This module builds its own full-FFT (or correctly conjugate-
weighted rFFT) annular binning instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def radial_bin_index(height: int, width: int, num_bins: int, *, rfft: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer-bin-index radial frequency grid (matches noise_init's convention)
    and the annulus index each frequency bin falls into, spanning
    ``[0, num_bins)`` linearly from 0 to the maximum radius present.
    """
    fy = torch.fft.fftfreq(height) * height
    fx = (torch.fft.rfftfreq(width) * width) if rfft else (torch.fft.fftfreq(width) * width)
    radius = torch.sqrt(fy.view(height, 1).square() + fx.view(1, -1).square())
    max_radius = radius.max()
    bin_width = max_radius / num_bins
    bin_index = torch.clamp((radius / bin_width).long(), max=num_bins - 1)
    return radius, bin_index


def _rfft_conjugate_weights(width: int) -> torch.Tensor:
    """Weight 2 for every rFFT column except DC and (if present) Nyquist,
    to correctly account for the omitted conjugate-symmetric half when
    building a full-plane annular sum from an rFFT-domain tensor.
    """
    n_cols = width // 2 + 1
    weights = torch.full((n_cols,), 2.0)
    weights[0] = 1.0
    if width % 2 == 0:
        weights[-1] = 1.0
    return weights


@dataclass(frozen=True)
class RadialPSD:
    bin_centers: torch.Tensor          # (num_bins,)
    power: torch.Tensor                # (num_bins,) cross-channel-averaged annular mean power
    total_energy: float                # Parseval-consistent total energy (sum over the full plane)
    counts: torch.Tensor               # (num_bins,) number of (weighted) full-plane frequency bins per annulus


def radial_psd(tensor: torch.Tensor, num_bins: int) -> RadialPSD:
    """Full-plane, cross-channel-averaged radial PSD of a (..., C, H, W) real tensor.

    Uses ``rfft2`` for efficiency but conjugate-weights every non-DC/non-
    Nyquist column by 2 before binning, so the annular mean and the total
    energy both agree with a full ``fft2`` computation (checked in
    tests/test_spectral.py's Parseval test).
    """
    if tensor.ndim < 3:
        raise ValueError("tensor must have shape (..., C, H, W)")
    height, width = tensor.shape[-2], tensor.shape[-1]
    fft = torch.fft.rfft2(tensor.to(torch.float64), dim=(-2, -1))
    power = fft.abs().square()  # (..., C, H, W//2+1)
    weights = _rfft_conjugate_weights(width).to(power.dtype)
    weighted_power = power * weights.view(*([1] * (power.ndim - 1)), -1)
    # Average over every leading dim except the trailing (H, W//2+1) grid, so a
    # (batch, C, H, W//2+1) or a bare (C, H, W//2+1) tensor both reduce to one
    # (H, W//2+1) cross-sample/cross-channel-averaged power grid.
    mean_power = weighted_power.reshape(-1, height, weighted_power.shape[-1]).mean(dim=0)
    _, bin_index = radial_bin_index(height, width, num_bins, rfft=True)
    flat_power, flat_bins = mean_power.reshape(-1), bin_index.reshape(-1)
    ones = torch.ones_like(flat_power)
    counts = torch.zeros(num_bins, dtype=flat_power.dtype).scatter_add_(0, flat_bins, ones)
    sums = torch.zeros(num_bins, dtype=flat_power.dtype).scatter_add_(0, flat_bins, flat_power)
    safe_counts = counts.clamp(min=1.0)
    annular_mean = sums / safe_counts
    radius, _ = radial_bin_index(height, width, num_bins, rfft=True)
    bin_width = radius.max() / num_bins
    bin_centers = (torch.arange(num_bins, dtype=torch.float64) + 0.5) * bin_width
    # Parseval (2D DFT, unnormalized): mean_x[x^2] = (1/(H*W)^2) * sum_{u,v} |F(u,v)|^2.
    total_energy = float(mean_power.sum()) / (height * width) ** 2
    return RadialPSD(bin_centers, annular_mean, total_energy, counts)


def parseval_energy(tensor: torch.Tensor) -> float:
    """Mean squared value in the spatial domain, for cross-checking radial_psd's total_energy."""
    return float(tensor.to(torch.float64).square().mean())


def angular_psd(tensor: torch.Tensor, num_angular_bins: int) -> torch.Tensor:
    """Cross-channel-averaged power as a function of angle, collapsed over radius.

    Descriptive diagnostic only (never used to gate acceptance).
    """
    height, width = tensor.shape[-2], tensor.shape[-1]
    fft = torch.fft.rfft2(tensor.to(torch.float64), dim=(-2, -1))
    power = fft.abs().square()
    weights = _rfft_conjugate_weights(width).to(power.dtype)
    weighted = (power * weights.view(*([1] * (power.ndim - 1)), -1)).reshape(-1, height, power.shape[-1]).mean(dim=0)
    fy = torch.fft.fftfreq(height) * height
    fx = torch.fft.rfftfreq(width) * width
    angle = torch.atan2(fy.view(height, 1).expand(height, fx.numel()), fx.view(1, -1).expand(height, fx.numel()))
    bin_index = torch.clamp(((angle + torch.pi) / (2 * torch.pi) * num_angular_bins).long(), 0, num_angular_bins - 1)
    flat_weighted, flat_bins = weighted.reshape(-1), bin_index.reshape(-1)
    counts = torch.zeros(num_angular_bins, dtype=flat_weighted.dtype).scatter_add_(0, flat_bins, torch.ones_like(flat_weighted))
    sums = torch.zeros(num_angular_bins, dtype=flat_weighted.dtype).scatter_add_(0, flat_bins, flat_weighted)
    return sums / counts.clamp(min=1.0)


def spatial_variance_map(tensor: torch.Tensor) -> torch.Tensor:
    """Per-pixel variance across the leading (batch) dimension; a coarse stationarity diagnostic."""
    if tensor.ndim < 3:
        raise ValueError("tensor must have a leading batch dimension")
    return tensor.to(torch.float64).var(dim=0, unbiased=False)


@dataclass(frozen=True)
class TransferMatrixDiagnostics:
    frequencies: torch.Tensor          # (n_freq, 2) integer (fy, fx) bin coordinates sampled
    singular_values: torch.Tensor      # (n_freq, C) singular values of M(omega) at each sampled frequency
    condition_numbers: torch.Tensor    # (n_freq,) max/min singular value ratio
    worst_condition_number: float


def linear_operator_frequency_response(
    apply_operator, channels: int, height: int, width: int
) -> torch.Tensor:
    """Return the complete rFFT transfer ``M(omega)`` of a fixed linear,
    translation-equivariant operator.

    The result has shape ``(H, W//2+1, C_out, C_in)``.  It is intentionally
    built from only ``channels`` impulses rather than a dense ``CHW x CHW``
    matrix.  Callers are responsible for ensuring their boundary handling
    makes the operator translation equivariant (the formal PSD editor uses
    circular patch extraction and fixed Fourier multipliers).
    """
    impulses = torch.zeros((channels, channels, height, width))
    for channel in range(channels):
        impulses[channel, channel, 0, 0] = 1.0
    responses = torch.stack(
        [apply_operator(impulses[channel:channel + 1])[0] for channel in range(channels)],
        dim=0,
    )  # (C_in, C_out, H, W)
    fft = torch.fft.rfft2(responses.to(torch.float64), dim=(-2, -1))
    return fft.permute(2, 3, 1, 0).contiguous()  # (H, Wf, C_out, C_in)


def covariance_distance_from_frequency_responses(
    reference_response: torch.Tensor,
    candidate_response: torch.Tensor,
    *,
    width: int,
) -> float:
    """Relative Frobenius distance between output covariance spectra.

    For unit white Gaussian input, ``Sigma(omega) = M(omega) M(omega)^*``.
    The aggregation uses the same real-FFT column multiplicities as radial
    PSD calculations.  This detects distribution changes that paired L2
    alone cannot: an orthogonal channel rotation has nonzero paired distance
    but exactly zero covariance distance.
    """
    if reference_response.shape != candidate_response.shape:
        raise ValueError("reference and candidate frequency responses must have the same shape")
    if reference_response.ndim != 4:
        raise ValueError("frequency responses must have shape (H, Wf, C_out, C_in)")
    ref = reference_response.to(torch.complex128)
    candidate = candidate_response.to(torch.complex128)
    sigma_ref = ref @ ref.conj().transpose(-2, -1)
    sigma_candidate = candidate @ candidate.conj().transpose(-2, -1)
    weights = _rfft_conjugate_weights(width).to(torch.float64).view(1, -1, 1, 1)
    numerator = ((sigma_candidate - sigma_ref).abs().square() * weights).sum()
    denominator = (sigma_ref.abs().square() * weights).sum()
    if denominator <= 0:
        return 0.0 if numerator <= 0 else float("inf")
    return float((numerator / denominator).sqrt())


def linear_operator_covariance_distance(
    reference_operator,
    candidate_operator,
    channels: int,
    height: int,
    width: int,
) -> float:
    """Convenience wrapper computing the analytic covariance distance."""
    reference = linear_operator_frequency_response(reference_operator, channels, height, width)
    candidate = linear_operator_frequency_response(candidate_operator, channels, height, width)
    return covariance_distance_from_frequency_responses(reference, candidate, width=width)


def impulse_response_transfer_matrix(apply_operator, channels: int, height: int, width: int, *, num_freq_samples: int = 32, seed: int = 0) -> TransferMatrixDiagnostics:
    """Per-frequency C x C transfer matrix of a linear per-channel-mixing operator.

    ``apply_operator`` must be a linear map ``(1, C, H, W) -> (1, C, H, W)``.
    Its per-frequency transfer matrix ``M(omega)`` is built from ``C`` impulse
    responses (one unit impulse per channel), each transformed with an FFT;
    ``M(omega)[:, c]`` is the FFT of the operator's response to a unit impulse
    in channel ``c``. Deliberately avoids constructing the full ``CHW x CHW``
    covariance/operator matrix.
    """
    response = linear_operator_frequency_response(apply_operator, channels, height, width)
    n_cols = response.shape[1]
    generator = torch.Generator().manual_seed(seed)
    total = height * n_cols
    num_samples = min(num_freq_samples, total)
    flat_indices = torch.randperm(total, generator=generator)[:num_samples]
    rows, cols = torch.div(flat_indices, n_cols, rounding_mode="floor"), flat_indices % n_cols
    singular_values, condition_numbers, coords = [], [], []
    for row, col in zip(rows.tolist(), cols.tolist()):
        matrix = response[row, col]  # (C_out, C_in) complex
        values = torch.linalg.svdvals(matrix)
        singular_values.append(values)
        condition_numbers.append(float(values.max() / values.min().clamp(min=1e-30)))
        coords.append((row, col))
    singular_values_tensor = torch.stack(singular_values)
    condition_numbers_tensor = torch.tensor(condition_numbers)
    return TransferMatrixDiagnostics(torch.tensor(coords), singular_values_tensor, condition_numbers_tensor, float(condition_numbers_tensor.max()))
