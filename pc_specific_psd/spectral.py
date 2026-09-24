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


RADIAL_PSD_DEFINITION_VERSION = "integer_radius_rfft_conjugate_weighted_v2"
CONDITIONING_DEFINITION_VERSION = "complete_rfft_grid_v2"


def diagnostic_definition_versions() -> dict[str, str]:
    return {
        "radial_psd": RADIAL_PSD_DEFINITION_VERSION,
        "conditioning": CONDITIONING_DEFINITION_VERSION,
    }


def low_mid_high_bin_ranges(num_bins: int) -> tuple[tuple[int, int], ...]:
    if num_bins < 3:
        raise ValueError("at least three radial bins are required for low/mid/high summaries")
    first = max(1, num_bins // 3)
    second = max(first + 1, 2 * num_bins // 3)
    second = min(second, num_bins - 1)
    return ((0, first), (first, second), (second, num_bins))


def radial_binning_metadata(height: int, width: int, num_bins: int) -> dict[str, object]:
    """Serializable definition of the exact annular coordinate system."""
    radius, _ = radial_bin_index(height, width, num_bins, rfft=True)
    radius_max = float(radius.max())
    edges = torch.linspace(0.0, radius_max, num_bins + 1, dtype=torch.float64)
    fy = (torch.fft.fftfreq(height) * height).to(torch.float64)
    fx = (torch.fft.rfftfreq(width) * width).to(torch.float64)
    ranges = low_mid_high_bin_ranges(num_bins)
    return {
        "version": RADIAL_PSD_DEFINITION_VERSION,
        "num_bins": num_bins,
        "bin_edges": edges.tolist(),
        "frequency_coordinates": {
            "units": "integer_fft_bins",
            "radius": "sqrt(fy**2 + fx**2)",
            "fy": fy.tolist(),
            "fx_rfft": fx.tolist(),
        },
        "boundary_rule": (
            "bins are [edge_i, edge_i+1), except the final bin includes radius_max; "
            "a frequency exactly on an interior edge belongs to the higher bin"
        ),
        "rfft_conjugate_weighting": (
            "weight 1 for DC and even-width Nyquist columns; weight 2 for other columns"
        ),
        "energy_bands": [
            {
                "name": name,
                "start_bin_inclusive": start,
                "end_bin_exclusive": end,
                "includes_dc": start == 0,
            }
            for name, (start, end) in zip(("low", "mid", "high"), ranges)
        ],
    }


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


def rfft_conjugate_weights(width: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Weight 2 for every rFFT column except DC and (if present) Nyquist,
    to correctly account for the omitted conjugate-symmetric half when
    building a full-plane annular sum from an rFFT-domain tensor.
    """
    n_cols = width // 2 + 1
    weights = torch.full((n_cols,), 2.0, dtype=dtype)
    weights[0] = 1.0
    if width % 2 == 0:
        weights[-1] = 1.0
    return weights


# Backward-compatible private spelling used by the existing diagnostics.
_rfft_conjugate_weights = rfft_conjugate_weights


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
    weights = rfft_conjugate_weights(width, dtype=power.dtype)
    weighted_power = power * weights.view(*([1] * (power.ndim - 1)), -1)
    # Average over every leading dim except the trailing (H, W//2+1) grid, so a
    # (batch, C, H, W//2+1) or a bare (C, H, W//2+1) tensor both reduce to one
    # (H, W//2+1) cross-sample/cross-channel-averaged power grid.
    mean_power = weighted_power.reshape(-1, height, weighted_power.shape[-1]).mean(dim=0)
    _, bin_index = radial_bin_index(height, width, num_bins, rfft=True)
    flat_power, flat_bins = mean_power.reshape(-1), bin_index.reshape(-1)
    flat_weights = weights.view(1, -1).expand(height, -1).reshape(-1)
    counts = torch.zeros(num_bins, dtype=flat_power.dtype).scatter_add_(0, flat_bins, flat_weights)
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
    frequencies: torch.Tensor          # (H*Wf, 2) integer (fy, fx) bin coordinates
    singular_values: torch.Tensor      # (H*Wf, C) singular values at every rFFT frequency
    condition_numbers: torch.Tensor    # (H*Wf,) per-frequency max/min ratio
    worst_condition_number: float
    minimum_singular_value: float | None = None
    maximum_singular_value: float | None = None
    worst_frequency: tuple[int, int] | None = None


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

def expected_mean_square_from_frequency_response(response: torch.Tensor, *, width: int) -> float:
    """Expected per-element output energy for unit white Gaussian input.

    For the unnormalised forward FFT used by torch, an LTI operator with
    transfer matrix ``M(k)`` has

    ``E[||L(epsilon)||^2] / (C_out*H*W) = sum_k w(k)||M(k)||_F^2 / (C_out*H*W)``.

    Accumulation is complex128/float64. Responses produced by
    :func:`linear_operator_frequency_response` still inherit the numerical
    precision of the operator used to create the impulse responses.
    """
    if response.ndim != 4:
        raise ValueError("frequency response must have shape (H, Wf, C_out, C_in)")
    height, n_cols, channels_out, _ = response.shape
    if n_cols != width // 2 + 1:
        raise ValueError("frequency response width is inconsistent with the declared spatial width")
    weights = rfft_conjugate_weights(width).view(1, n_cols, 1, 1)
    energy = (response.to(torch.complex128).abs().square() * weights).sum()
    return float(energy / float(channels_out * height * width))


@dataclass(frozen=True)
class ExpectedRadialPSD:
    """Cross-channel expected annular power of a fixed linear operator."""

    power: torch.Tensor
    counts: torch.Tensor
    bin_index: torch.Tensor
    total_expected_mean_square: float


def expected_radial_psd_from_frequency_response(
    response: torch.Tensor, *, width: int, num_bins: int
) -> ExpectedRadialPSD:
    """Return deterministic annular power for unit white input.

    ``power[b]`` follows the transfer-response convention from the research
    protocol. A raw unnormalised-FFT periodogram has expectation ``H*W``
    times this value; ratios and Fourier-control multipliers are unaffected.
    """
    if response.ndim != 4:
        raise ValueError("frequency response must have shape (H, Wf, C_out, C_in)")
    height, n_cols, channels_out, _ = response.shape
    if n_cols != width // 2 + 1:
        raise ValueError("frequency response width is inconsistent with the declared spatial width")
    _, bin_index = radial_bin_index(height, width, num_bins, rfft=True)
    weights = rfft_conjugate_weights(width).view(1, n_cols).expand(height, -1)
    per_frequency = response.to(torch.complex128).abs().square().sum(dim=(-2, -1)).real
    flat_bins = bin_index.reshape(-1)
    counts = torch.zeros(num_bins, dtype=torch.float64).scatter_add_(0, flat_bins, weights.reshape(-1))
    sums = torch.zeros(num_bins, dtype=torch.float64).scatter_add_(
        0, flat_bins, (per_frequency * weights).reshape(-1)
    )
    power = torch.zeros_like(sums)
    nonempty = counts > 0
    power[nonempty] = sums[nonempty] / (float(channels_out) * counts[nonempty])
    return ExpectedRadialPSD(
        power=power,
        counts=counts,
        bin_index=bin_index,
        total_expected_mean_square=expected_mean_square_from_frequency_response(response, width=width),
    )


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


def transfer_matrix_diagnostics_from_frequency_response(response: torch.Tensor) -> TransferMatrixDiagnostics:
    """Exhaustive singular-value diagnostics for an already-computed response."""
    if response.ndim != 4:
        raise ValueError("frequency response must have shape (H, Wf, C_out, C_in)")
    height, n_cols = response.shape[:2]
    matrices = response.to(torch.complex128).reshape(-1, response.shape[-2], response.shape[-1])
    singular_values = torch.linalg.svdvals(matrices)
    minima = singular_values.min(dim=1).values
    maxima = singular_values.max(dim=1).values
    condition_numbers = maxima / minima.clamp(min=1e-30)
    flat_worst = int(condition_numbers.argmax())
    rows = torch.arange(height, dtype=torch.int64).view(-1, 1).expand(height, n_cols).reshape(-1)
    cols = torch.arange(n_cols, dtype=torch.int64).view(1, -1).expand(height, n_cols).reshape(-1)
    frequencies = torch.stack((rows, cols), dim=1)
    worst_frequency = (flat_worst // n_cols, flat_worst % n_cols)
    return TransferMatrixDiagnostics(
        frequencies=frequencies,
        singular_values=singular_values,
        condition_numbers=condition_numbers,
        worst_condition_number=float(condition_numbers[flat_worst]),
        minimum_singular_value=float(minima.min()),
        maximum_singular_value=float(maxima.max()),
        worst_frequency=worst_frequency,
    )


def impulse_response_transfer_matrix(apply_operator, channels: int, height: int, width: int, *, num_freq_samples: int | None = None, seed: int = 0) -> TransferMatrixDiagnostics:
    """Complete-grid transfer diagnostics for a linear LTI operator.

    The legacy sampling keywords remain accepted for compatibility but are
    ignored: every ``H*(W//2+1)`` transfer matrix is now evaluated.
    """
    del num_freq_samples, seed
    response = linear_operator_frequency_response(apply_operator, channels, height, width)
    return transfer_matrix_diagnostics_from_frequency_response(response)
