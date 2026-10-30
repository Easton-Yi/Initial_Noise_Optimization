"""Gaussian coupling across the batch dimension of image latents."""

import math
import numbers

import torch


def validate_strength(strength):
    if isinstance(strength, bool) or not isinstance(strength, numbers.Real):
        raise ValueError("Coupled Initialization: strength must be a number in [0, 1].")
    strength = float(strength)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("Coupled Initialization: strength must be finite and in [0, 1].")
    return strength



def validate_image_batch(samples):
    if not isinstance(samples, torch.Tensor) or samples.is_nested:
        raise ValueError("Coupled Initialization expects a regular image latent tensor.")
    if samples.layout != torch.strided or samples.ndim != 4:
        raise ValueError(
            "Coupled Initialization v0.1 expects image latents [batch, channels, height, width]. "
            "Video and nested latents are not supported in this version."
        )
    if not samples.is_floating_point() or any(size < 1 for size in samples.shape):
        raise ValueError("Coupled Initialization expects a nonempty floating-point image batch.")


def couple_noise(eps, strength=1.0):
    """Transform IID Gaussian rows; rho = -strength / (K - 1).

    The first row is unchanged, for a shared reference image with native IID.
    Each row retains its standard Gaussian marginal in exact arithmetic.
    Only the gallery dimension (axis 0) is coupled. Half-precision reductions
    are computed in float32; float64 inputs retain double precision.
    The identity endpoints return the input exactly, including its dtype.
    """
    strength = validate_strength(strength)
    validate_image_batch(eps)
    k = eps.shape[0]
    if strength == 0.0 or k == 1:
        return eps
    work = eps.float() if eps.dtype in (torch.float16, torch.bfloat16) else eps
    rho = -strength / (k - 1)
    tail = work[1:]
    mean = tail.mean(dim=0, keepdim=True)
    # Conditional covariance of the remaining rows given the first:
    # (1-rho) I + (rho-rho**2) 11^T. Its mean-mode eigenvalue is
    # (1-rho)(1-strength); its other eigenvalues are 1-rho.
    result = rho * work[:1] + math.sqrt(1.0 - rho) * (tail - mean)
    if strength < 1.0:
        result = result + math.sqrt((1.0 - rho) * (1.0 - strength)) * mean
    return torch.cat((eps[:1], result.to(dtype=eps.dtype)), dim=0)

# Mathematical transform above is vendored from Jing Jia's author implementation.
# MIT copyright (c) 2026 Coupled Initialization contributors.
# Full notice: docs/coupled_initialization_MIT_LICENSE.txt.
UPSTREAM_REVISION = "2cf9531d4becb591c37907c1f9c54d1e4383c7f9"
IMPLEMENTATION_VERSION = "first_row_gaussian_coupling_v1"
COUPLED_FAMILIES = ("antithetic", "repulsive")


def antithetic(base):
    validate_image_batch(base)
    if base.shape[0] != 4:
        raise ValueError("Antithetic formal galleries require four basis samples")
    return torch.stack((base[0], -base[0], base[2], -base[2]))


def formal_noise(base, family, normalization_profile):
    validate_image_batch(base)
    if base.shape[0] != 4 or base.device.type != "cpu" or base.dtype != torch.float32:
        raise ValueError("Coupled formal input must be CPU FP32 [4,C,H,W]")
    if normalization_profile != "none":
        raise ValueError("Coupled sampling requires normalization_profile=none")
    if not torch.isfinite(base).all():
        raise ValueError("Coupled basis must be finite")
    if family == "antithetic":
        return antithetic(base)
    if family == "repulsive":
        return couple_noise(base, 1.0)
    raise ValueError(f"Unknown coupled family: {family}")


def enabled(config):
    return any(config.get("coupled_sampling", {}).get(k, {}).get("enabled", False)
               for k in COUPLED_FAMILIES)


def validate_config(config):
    section = config.get("coupled_sampling", {})
    if set(section) - set(COUPLED_FAMILIES):
        raise ValueError("Unknown coupled_sampling method")
    for family, options in section.items():
        allowed = {"enabled", "pairing"} if family == "antithetic" else {"enabled", "strength"}
        if set(options) - allowed:
            raise ValueError(f"Unknown {family} configuration fields")
        if family == "antithetic" and options.get("pairing", "adjacent") != "adjacent":
            raise ValueError("Formal antithetic pairing must be adjacent")
        if family == "repulsive" and validate_strength(options.get("strength", 1.0)) != 1.0:
            raise ValueError("Formal repulsive strength must be 1.0")
    if not enabled(config):
        return
    exp = config["experiment"]
    if exp["gallery_size"] != 4 or exp["normalization_profile"] != "none":
        raise ValueError("Coupled experiments require gallery_size=4 and normalization_profile=none")
    if config["generation"]["generation_batch_size"] != 1:
        raise ValueError("Coupled experiments require generation_batch_size=1")
    for key in ("same_phase_floor", "independent_white"):
        if config.get(key, {}).get("enabled", False):
            raise ValueError("Coupled runs cannot include spectral floor conditions")
    baseline = config.get("baseline", {})
    if baseline.get("enabled", True) and baseline.get("alpha_values") != [0.0]:
        raise ValueError("Only explicitly enabled raw white can accompany coupled methods")


def source_identity():
    from pathlib import Path
    from io_utils import file_hash
    root = Path(__file__).parent
    return {"upstream_revision": UPSTREAM_REVISION, "version": IMPLEMENTATION_VERSION,
            "files": {name: file_hash(root / name) for name in
                      ("coupled_sampling.py", "coupled_run.py", "noise_methods.py", "io_utils.py",
                       "model_adapters.py", "run_experiment.py", "run_support.py")}}


def basis_provenance(batch, family):
    matrix = (torch.tensor([[1., 0., 0., 0.], [-1., 0., 0., 0.],
                            [0., 0., 1., 0.], [0., 0., -1., 0.]])
              if family == "antithetic" else
              couple_noise(torch.eye(4).reshape(4, 4, 1, 1)).reshape(4, 4))
    return {"basis_seeds": batch.sample_seeds, "basis_hashes": batch.base_hashes,
            "source_basis_indices": [0, 2] if family == "antithetic" else [0, 1, 2, 3],
            "transform_coefficients": matrix.tolist(),
            "pairing": [[0, 1], [2, 3]] if family == "antithetic" else None,
            "strength": 1.0 if family == "repulsive" else None,
            "implementation_version": IMPLEMENTATION_VERSION}
