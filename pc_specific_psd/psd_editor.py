"""Stride-1 overlapping PSD editor (plan section 5.2-5.4).

Given a candidate PC group ``B`` (a set of coefficient-map indices into the
``OverlapCodec`` basis) and its complement, this module applies

    A~_i(w) = h_ref(r) * t_B(r; tau) * A_i(w),   i in B
    A~_i(w) = h_ref(r) * A_i(w),                 i not in B

in the frequency domain of each per-PC coefficient map, inverse-FFTs, and
reconstructs via ``OverlapCodec.decode_center`` (Center() synthesis, not
overlap-add). Output-PSD calibration (the c_{B,tau}(b) correction factor) is
a separate, later step -- see calibration.py -- and is not applied here.

Frozen same-phase reference (plan section 5.1/section 14): the raw reference
amplitude response

    h_ref(r) = sqrt((1 - gamma) * H_alpha(r)**2 + gamma),  H_alpha(r) = (1+r)**(-alpha)

reuses noise_init's own same-phase white-floor construction at
SAME_PHASE_ALPHA=0.9 / SAME_PHASE_GAMMA=0.05. The public response multiplies
that raw response by one deterministic analytic scalar: the reciprocal of its
expected RMS under unit white input. The expectation uses Parseval's theorem
with the rfft2 half-spectrum's conjugate-column weights (1 for DC and the
even-width Nyquist column, 2 otherwise) and denominator H*W. This restores
expected unit latent RMS without any realization-dependent normalization, so
the operator remains linear, Gaussian-preserving, and phase-preserving. The
parameters and scale profile are module-level constants, not config fields --
config.py must reject override attempts and echo them into provenance.

Frequency axis: h_ref(r) and the gate w(r) are both evaluated on the same
integer-FFT-bin radius grid used by noise_init.radial_frequency_grid (fy =
fftfreq(H)*H, fx = rfftfreq(W)*W), over the coefficient maps' own (H, W)
spatial domain. This keeps the two functions on one consistent axis (plan
section 5.2); a separate cycles-per-latent-pixel axis is only needed for
cross-resolution comparison, which is out of scope within a single run where
H and W are fixed throughout.

tau=0 special case (plan section 6.1): t_B(r; 0) = exp(0) = 1 for every PC,
in-group or not, so the whole edit collapses to applying h_ref(r) uniformly
to every coefficient map. Because OverlapCodec's encode/decode_center round
trip is the identity for an all-ones filter (project_all_ones_is_identity_check),
a uniform frequency-domain filter commutes through that round trip, and
applying h_ref(r) to every coefficient map is therefore exactly equivalent to
applying h_ref(r) directly to the raw latent -- i.e.
noise_init.same_phase_floor(base_white, 0.9, 0.05) times the same analytic
expected-unit-RMS scale. ``apply_psd_edit_tau_zero`` is that scalar shortcut,
used for production tau=0 runs so calibration
sampling error can never manufacture a spurious nonzero intensity difference
at the identity point. tests/test_psd_editor.py separately forces the full
general codec path (``apply_psd_edit(..., reference=True)``) at tau=0 and
checks it agrees with both this shortcut and same_phase_floor via
torch.allclose, not bitwise equality.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import torch

from pc_specific_psd import manifests
from pc_specific_psd.compat_generation import pink_filter, radial_frequency_grid, same_phase_floor, sample_noise_batch
from pc_specific_psd.patch_codec import OverlapCodec

SAME_PHASE_ALPHA = 0.9
SAME_PHASE_GAMMA = 0.05
REFERENCE_SCALE_PROFILE = "expected_unit_rms_rfft_v1"


def base_white_for_draw(block: manifests.SeedBlock, base_index: int, *, channels: int, height: int, width: int) -> torch.Tensor:
    """The one base-draw rule shared by every condition in the study: a
    single-image draw at ``sample_seed = block.batch_seed + base_index``, via
    noise_init's own ``sample_noise_batch`` -- zero PCA-specific machinery.

    This is ``probing.draw_base_latent``'s rule generalized to any
    ``base_index`` (probing entries only ever use ``base_index=0``, so that
    function bakes the block's ``batch_seed`` in directly; this one takes
    ``base_index`` explicitly so preview/full-pilot draws at ``base_index``
    1-3 get the correct offset seed too). For a fixed ``(prompt_id, block_id,
    base_index)`` triple, every condition being compared (reference,
    candidate+/-, ...) must call this with the same ``block``/``base_index``
    and will get back a byte-identical tensor -- that pairing, not any one
    specific triple, is the contract; a single fixed triple is a valid test
    fixture but must never be treated as shared production noise.
    """
    sample_seed = block.sample_seed(base_index)
    batch = sample_noise_batch(sample_seed, block.block_id, (1, channels, height, width), batch_seed=sample_seed)
    return batch.base_white


def _raw_reference_amplitude_response(
    height: int,
    width: int,
    *,
    alpha: float = SAME_PHASE_ALPHA,
    gamma: float = SAME_PHASE_GAMMA,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Unscaled same-phase amplitude response, shape ``(H, W//2+1)``."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    h_alpha = pink_filter(height, width, alpha, device=device)
    return torch.sqrt((1.0 - gamma) * h_alpha.square() + gamma)


def _rfft_conjugate_column_weights(width: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Multiplicity of each rfft2 column in the omitted full spectrum."""
    weights = torch.full((width // 2 + 1,), 2.0, dtype=torch.float32, device=device)
    weights[0] = 1.0
    if width % 2 == 0:
        weights[-1] = 1.0
    return weights


@lru_cache(maxsize=None)
def reference_expected_rms_multiplier(
    height: int,
    width: int,
    alpha: float = SAME_PHASE_ALPHA,
    gamma: float = SAME_PHASE_GAMMA,
) -> float:
    """Expected output RMS for unit white input under the unscaled response.

    For torch's default FFT normalization, Parseval gives
    ``E[RMS^2] = sum_full_spectrum(|h|^2) / (height * width)``.  Since only
    the rfft2 half-spectrum is stored, the omitted conjugate columns are
    restored with weights 1/2/1 for DC/interior/Nyquist respectively.  All
    rows are already present, so no row weighting is needed.

    This size-only scalar is intentionally computed on CPU and cached.  It is
    applied as a Python scalar so callers' tensor dtype and device are
    unchanged and no latent is moved between devices.
    """
    raw = _raw_reference_amplitude_response(height, width, alpha=alpha, gamma=gamma)
    weights = _rfft_conjugate_column_weights(width)
    mean_square = (raw.square() * weights.view(1, -1)).sum() / float(height * width)
    return float(mean_square.sqrt())


def reference_expected_unit_rms_scale(
    height: int,
    width: int,
    alpha: float = SAME_PHASE_ALPHA,
    gamma: float = SAME_PHASE_GAMMA,
) -> float:
    """Fixed analytic scale making the reference's expected RMS equal one."""
    multiplier = reference_expected_rms_multiplier(height, width, alpha, gamma)
    if multiplier <= 0.0:
        raise ValueError("reference expected RMS multiplier must be positive")
    return 1.0 / multiplier


def reference_amplitude_response(height: int, width: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Expected-unit-RMS frozen reference response, shape ``(H, W//2+1)``.

    The response's relative spectrum is identical to ``same_phase_floor``;
    only one deterministic size-dependent scalar is added.
    """
    raw = _raw_reference_amplitude_response(height, width, device=device)
    return raw * reference_expected_unit_rms_scale(height, width)


def low_frequency_gate(height: int, width: int, r_s: float, beta: float, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """w(r) = [1 + (r/r_s)^beta]^-1, plan section 5.3. Shape (H, W//2+1)."""
    if r_s <= 0:
        raise ValueError("r_s must be positive")
    if beta <= 0:
        raise ValueError("beta must be positive")
    radius = radial_frequency_grid(height, width, device=device)
    return (1.0 + (radius / r_s).pow(beta)).reciprocal()


def group_transfer_multiplier(height: int, width: int, r_s: float, beta: float, tau: float, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """t_B(r; tau) = exp(0.5 * tau * w(r)), plan section 5.3. Shape (H, W//2+1)."""
    gate = low_frequency_gate(height, width, r_s, beta, device=device)
    return torch.exp(0.5 * float(tau) * gate)


def apply_psd_edit(
    codec: OverlapCodec,
    latents: torch.Tensor,
    group_indices: Sequence[int],
    tau: float,
    r_s: float,
    beta: float,
    *,
    reference: bool = False,
) -> torch.Tensor:
    """Applies the group/complement split filter and reconstructs via Center().

    ``group_indices`` is the candidate PC group ``B`` (typically
    ``list(pc_group.indices)`` from manifests.PCGroup) -- every other
    coefficient-map index is treated as the complement. ``reference=True``
    forces OverlapCodec's explicit-loop encode/decode_center_reference
    methods instead of the vectorized ones; this is the "forced full general
    codec path" required by tests/test_psd_editor.py's tau=0 equivalence
    check, not a separate/duplicate implementation of the frequency-domain
    filtering itself (an FFT has no meaningful loop-based analog to test
    against).
    """
    if latents.ndim != 4:
        raise ValueError("latents must have shape (batch, channels, height, width)")
    group_idx = list(group_indices)
    if len(group_idx) != len(set(group_idx)):
        raise ValueError("group_indices must not contain duplicates")
    if any(index < 0 or index >= codec.d for index in group_idx):
        raise ValueError(f"group_indices must be within [0, {codec.d})")
    height, width = latents.shape[-2], latents.shape[-1]

    coefficient_maps = codec.encode(latents, reference=reference)  # (batch, d, H, W)
    spectrum = torch.fft.rfft2(coefficient_maps, dim=(-2, -1))
    h_ref = reference_amplitude_response(height, width, device=latents.device).to(coefficient_maps.dtype)
    filtered = spectrum * h_ref
    if float(tau) != 0.0 and group_idx:
        t_b = group_transfer_multiplier(height, width, r_s, beta, tau, device=latents.device).to(coefficient_maps.dtype)
        filtered[:, group_idx, :, :] = filtered[:, group_idx, :, :] * t_b
    edited_maps = torch.fft.irfft2(filtered, s=(height, width), dim=(-2, -1)).to(coefficient_maps.dtype)
    return codec.decode_center_reference(edited_maps) if reference else codec.decode_center(edited_maps)


def apply_psd_edit_tau_zero(base_white: torch.Tensor) -> torch.Tensor:
    """Exact scalar-reference shortcut for tau=0 production runs.

    Exactly ``same_phase_floor(base_white, 0.9, 0.05)`` times the analytic
    expected-unit-RMS scale for the input's spatial size.  The fixed scalar
    preserves the input tensor's dtype/device and keeps the shortcut equal to
    the general codec path at tau=0.
    """
    height, width = base_white.shape[-2:]
    scale = reference_expected_unit_rms_scale(height, width)
    return same_phase_floor(base_white, SAME_PHASE_ALPHA, SAME_PHASE_GAMMA) * scale
