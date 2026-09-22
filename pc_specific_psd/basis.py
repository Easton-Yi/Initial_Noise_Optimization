"""Phase A: shared PCA basis over clean VAE latent patches.

Per plan §3/§14.3 and prompt §6: patches are sampled from *posterior-mean*
VAE-encoded natural images (never from white/pink/benchmark-generated noise),
using only dataset-level centering -- the saved ``mean`` is a statistic for
provenance/inspection only and must never be added back into noise
coefficients (``a = V^T q``, ``q = V a``, no mean, no eigenvalue scaling, no
whitening). This module has zero hard dependency on any VAE/diffusers
loading code: real encoding is injected by the caller as a plain
``Callable[[Sequence[str]], Tensor]`` (a future ``adapters.py`` wraps the
actual SDXL VAE for this), so CPU/synthetic tests exercise the exact same
``build_pca_basis`` code path as the eventual real run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from pc_specific_psd.compat_generation import canonical_json, derived_seed, sha256_text

BASIS_FORMAT_VERSION = "pc_specific_psd_basis_v1"
PATCH_SAMPLING_SEED_NAMESPACE = "pc_basis_patch_sampling_v1"


@dataclass(frozen=True)
class ImageManifestEntry:
    image_id: str
    path: str
    sha256: str


@dataclass(frozen=True)
class DatasetManifest:
    """Locked-before-construction dataset source. ``source`` records which
    real dataset this came from (e.g. ``"imagenet_2000_v1"``) so it is never
    silently swapped after the fact; ``"synthetic_test_fixture"`` (or any
    source string a caller chooses to mark synthetic) is only ever used with
    ``build_pca_basis(..., synthetic=True)``.
    """
    source: str
    entries: tuple[ImageManifestEntry, ...]

    def __len__(self) -> int:
        return len(self.entries)

    def split_disjoint_halves(self, seed: int) -> tuple["DatasetManifest", "DatasetManifest"]:
        """Deterministic image-disjoint split for the split-half stability check (plan §3.2)."""
        if len(self.entries) < 2:
            raise ValueError("need at least 2 images to form a disjoint split")
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(self.entries), generator=generator).tolist()
        midpoint = len(order) // 2
        first = tuple(self.entries[i] for i in sorted(order[:midpoint]))
        second = tuple(self.entries[i] for i in sorted(order[midpoint:]))
        return DatasetManifest(self.source, first), DatasetManifest(self.source, second)


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    payload = json.loads(Path(path).read_text())
    entries = tuple(ImageManifestEntry(**entry) for entry in payload["entries"])
    return DatasetManifest(source=payload["source"], entries=entries)


def write_dataset_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    payload = {"source": manifest.source, "entries": [entry.__dict__ for entry in manifest.entries]}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def manifest_hash(manifest: DatasetManifest) -> str:
    payload = {"source": manifest.source, "entries": [entry.__dict__ for entry in manifest.entries]}
    return sha256_text(canonical_json(payload))


def sample_random_patches(latents: torch.Tensor, patch_size: int, patches_per_image: int, seed: int) -> torch.Tensor:
    """Random overlapping patches for basis fitting (plan §14.3: fitting may use
    overlapping patches; only the probing grid must be non-overlapping).

    Flatten order matches ``patch_codec.extract_patches``: channel-major, then
    row-in-patch, then col-in-patch, so a fitted basis can be dropped directly
    into either codec without a re-flatten step.

    Returns ``(batch * patches_per_image, C*patch_size*patch_size)``.
    """
    if latents.ndim != 4:
        raise ValueError("latents must have shape (batch, channels, height, width)")
    batch, channels, height, width = latents.shape
    if height < patch_size or width < patch_size:
        raise ValueError(f"patch_size {patch_size} does not fit in a {height}x{width} plane")
    generator = torch.Generator().manual_seed(seed)
    max_row, max_col = height - patch_size + 1, width - patch_size + 1
    rows = torch.randint(0, max_row, (batch, patches_per_image), generator=generator)
    cols = torch.randint(0, max_col, (batch, patches_per_image), generator=generator)
    d = channels * patch_size * patch_size
    out = torch.empty((batch * patches_per_image, d), dtype=latents.dtype)
    index = 0
    for b in range(batch):
        for k in range(patches_per_image):
            r0, c0 = int(rows[b, k]), int(cols[b, k])
            out[index] = latents[b, :, r0:r0 + patch_size, c0:c0 + patch_size].reshape(-1)
            index += 1
    return out


@dataclass
class StreamingMeanCovariance:
    """Float64-accumulated streaming mean/covariance (plan §3.2), so the full
    ~64,000-patch fit never needs every patch resident in memory at once.
    """
    dimension: int
    count: int = 0
    _sum: torch.Tensor = field(default=None, repr=False)
    _sum_outer: torch.Tensor = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._sum is None:
            self._sum = torch.zeros(self.dimension, dtype=torch.float64)
        if self._sum_outer is None:
            self._sum_outer = torch.zeros((self.dimension, self.dimension), dtype=torch.float64)

    def update(self, patches: torch.Tensor) -> None:
        if patches.ndim != 2 or patches.shape[1] != self.dimension:
            raise ValueError(f"expected patches of shape (*, {self.dimension}), got {tuple(patches.shape)}")
        patches64 = patches.to(torch.float64)
        self.count += patches64.shape[0]
        self._sum += patches64.sum(dim=0)
        self._sum_outer += patches64.T @ patches64

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, unbiased covariance) exactly matching plan §3.2's
        mu = (1/N) sum q_n, Sigma = (1/(N-1)) sum (q_n-mu)(q_n-mu)^T.
        """
        if self.count < 2:
            raise ValueError("need at least 2 samples to compute an unbiased covariance")
        mean = self._sum / self.count
        covariance = (self._sum_outer - self.count * torch.outer(mean, mean)) / (self.count - 1)
        return mean, covariance


def direct_mean_covariance(patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-streaming reference computation, kept permanently as a test fixture."""
    patches64 = patches.to(torch.float64)
    mean = patches64.mean(dim=0)
    centered = patches64 - mean
    covariance = (centered.T @ centered) / (patches64.shape[0] - 1)
    return mean, covariance


def eigendecompose_covariance(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Descending-eigenvalue-order orthonormal eigenbasis of a symmetric covariance."""
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)  # ascending
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvalues[order], eigenvectors[:, order]


def orthogonality_error(components: torch.Tensor) -> float:
    """max|V^T V - I|, for the required orthonormality check (acceptance ID A02/N04)."""
    d = components.shape[0]
    gram = components.T.to(torch.float64) @ components.to(torch.float64)
    return float((gram - torch.eye(d, dtype=torch.float64)).abs().max())


def principal_angles(basis_a: torch.Tensor, basis_b: torch.Tensor, num_components: int | None = None) -> torch.Tensor:
    """Principal angles (radians, ascending) between the subspaces spanned by the
    leading columns of two orthonormal bases sharing the same ambient dimension d.
    Singular values of V_a^T V_b are the cosines of the principal angles.
    """
    va = basis_a if num_components is None else basis_a[:, :num_components]
    vb = basis_b if num_components is None else basis_b[:, :num_components]
    cross = va.T.to(torch.float64) @ vb.to(torch.float64)
    cosines = torch.linalg.svdvals(cross).clamp(-1.0, 1.0)
    return torch.arccos(cosines)


@dataclass(frozen=True)
class PCABasis:
    components: torch.Tensor   # (d, d) float32, columns are components v_1..v_d, descending eigenvalue order
    eigenvalues: torch.Tensor  # (d,) float64, descending
    mean: torch.Tensor         # (d,) float64 dataset mean -- statistics/provenance only, never applied to noise
    patch_size: int
    channels: int
    num_samples: int
    metadata: dict


def build_pca_basis(
    manifest: DatasetManifest,
    encoder: Callable[[Sequence[str]], torch.Tensor],
    *,
    patch_size: int,
    channels: int,
    patches_per_image: int,
    sampling_seed: int,
    synthetic: bool,
    chunk_size: int = 64,
) -> PCABasis:
    """Streams the manifest through ``encoder`` in chunks, accumulating a
    float64 streaming covariance over randomly sampled patches, then
    eigendecomposes. ``synthetic`` must be passed explicitly (never inferred)
    and is stamped into the saved metadata so a synthetic basis can never be
    silently mistaken for a real one on load (see ``load_basis``).
    """
    if len(manifest) == 0:
        raise ValueError("dataset manifest is empty")
    d = channels * patch_size * patch_size
    streaming = StreamingMeanCovariance(d)
    image_paths = [entry.path for entry in manifest.entries]
    for chunk_start in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[chunk_start:chunk_start + chunk_size]
        latents = encoder(chunk_paths)
        if latents.ndim != 4 or latents.shape[0] != len(chunk_paths) or latents.shape[1] != channels:
            raise ValueError(
                f"encoder returned shape {tuple(latents.shape)}, expected "
                f"({len(chunk_paths)}, {channels}, H, W)"
            )
        chunk_seed = derived_seed(sampling_seed, PATCH_SAMPLING_SEED_NAMESPACE, chunk_start)
        patches = sample_random_patches(latents, patch_size, patches_per_image, chunk_seed)
        streaming.update(patches)
    mean, covariance = streaming.finalize()
    eigenvalues, components = eigendecompose_covariance(covariance)
    metadata = {
        "format_version": BASIS_FORMAT_VERSION,
        "synthetic": bool(synthetic),
        "dataset_source": manifest.source,
        "manifest_hash": manifest_hash(manifest),
        "num_images": len(manifest),
        "patches_per_image": patches_per_image,
        "num_samples": int(streaming.count),
        "sampling_seed": sampling_seed,
        "patch_size": patch_size,
        "channels": channels,
        "flatten_order": "channel_major_row_col",
    }
    return PCABasis(
        components=components.to(torch.float32),
        eigenvalues=eigenvalues,
        mean=mean,
        patch_size=patch_size,
        channels=channels,
        num_samples=int(streaming.count),
        metadata=metadata,
    )


@dataclass(frozen=True)
class EigenvalueGap:
    boundary_after_pc_1based: int
    absolute_a: float | None
    absolute_b: float | None
    relative_a: float | None
    relative_b: float | None


@dataclass(frozen=True)
class BandSubspaceStability:
    indices_0based: tuple[int, ...]
    angles_radians: torch.Tensor
    mean_angle_radians: float
    max_angle_radians: float
    boundary_eigenvalue_gap: EigenvalueGap


@dataclass(frozen=True)
class SplitHalfStability:
    # Kept for provenance only. Full square bases span the same ambient space
    # and therefore cannot diagnose leading-PC stability.
    angles_full: torch.Tensor
    angles_leading: torch.Tensor
    mean_angle_leading: float
    max_angle_leading: float
    leading_eigenvalue_gap: EigenvalueGap
    bands: dict[str, BandSubspaceStability]
    angle_unit: str
    eigenvalue_gap_definition: str
    split_seed: int
    num_images_a: int
    num_images_b: int
    num_samples_a: int
    num_samples_b: int


def _boundary_eigenvalue_gap(
    eigenvalues_a: torch.Tensor, eigenvalues_b: torch.Tensor, boundary_1based: int
) -> EigenvalueGap:
    """Gap after PC ``boundary_1based``: lambda_k - lambda_(k+1).

    Relative gaps divide by ``abs(lambda_k)``.  The final ambient component
    has no following eigenvalue and is reported as ``None`` rather than a
    fabricated zero.
    """
    index = boundary_1based - 1
    if index < 0 or index >= eigenvalues_a.numel() or index >= eigenvalues_b.numel():
        raise ValueError("eigenvalue-gap boundary is outside the basis")
    if index + 1 >= eigenvalues_a.numel() or index + 1 >= eigenvalues_b.numel():
        return EigenvalueGap(boundary_1based, None, None, None, None)
    gap_a = float(eigenvalues_a[index] - eigenvalues_a[index + 1])
    gap_b = float(eigenvalues_b[index] - eigenvalues_b[index + 1])
    denom_a = max(abs(float(eigenvalues_a[index])), 1e-30)
    denom_b = max(abs(float(eigenvalues_b[index])), 1e-30)
    return EigenvalueGap(boundary_1based, gap_a, gap_b, gap_a / denom_a, gap_b / denom_b)


def compare_basis_subspaces(
    basis_a: PCABasis,
    basis_b: PCABasis,
    *,
    num_leading_components: int,
    bands: Mapping[str, Sequence[int]] = (),
    split_seed: int = 0,
    num_images_a: int = 0,
    num_images_b: int = 0,
) -> SplitHalfStability:
    """Compare leading and existing-band subspaces between two fitted bases.

    Components are sliced *before* principal-angle calculation.  This is the
    critical distinction from comparing two complete square bases and then
    slicing their (necessarily near-zero) full-space angles.
    """
    d = basis_a.components.shape[1]
    if basis_a.components.shape != basis_b.components.shape:
        raise ValueError("split-half bases must have the same component shape")
    if not 1 <= num_leading_components <= d:
        raise ValueError("num_leading_components must be within the basis dimension")
    angles_full = principal_angles(basis_a.components, basis_b.components)
    angles_leading = principal_angles(
        basis_a.components[:, :num_leading_components],
        basis_b.components[:, :num_leading_components],
    )
    band_results: dict[str, BandSubspaceStability] = {}
    for name, raw_indices in dict(bands).items():
        indices = tuple(int(index) for index in raw_indices)
        if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= d:
            raise ValueError(f"band {name!r} has invalid component indices")
        band_angles = principal_angles(
            basis_a.components[:, list(indices)], basis_b.components[:, list(indices)]
        )
        gap = _boundary_eigenvalue_gap(basis_a.eigenvalues, basis_b.eigenvalues, max(indices) + 1)
        band_results[name] = BandSubspaceStability(
            indices_0based=indices,
            angles_radians=band_angles,
            mean_angle_radians=float(band_angles.mean()),
            max_angle_radians=float(band_angles.max()),
            boundary_eigenvalue_gap=gap,
        )
    return SplitHalfStability(
        angles_full=angles_full,
        angles_leading=angles_leading,
        mean_angle_leading=float(angles_leading.mean()),
        max_angle_leading=float(angles_leading.max()),
        leading_eigenvalue_gap=_boundary_eigenvalue_gap(
            basis_a.eigenvalues, basis_b.eigenvalues, num_leading_components
        ),
        bands=band_results,
        angle_unit="radians",
        eigenvalue_gap_definition="lambda_k - lambda_(k+1), relative gap divided by abs(lambda_k)",
        split_seed=int(split_seed),
        num_images_a=int(num_images_a),
        num_images_b=int(num_images_b),
        num_samples_a=basis_a.num_samples,
        num_samples_b=basis_b.num_samples,
    )


def split_half_stability(
    manifest: DatasetManifest,
    encoder: Callable[[Sequence[str]], torch.Tensor],
    *,
    patch_size: int,
    channels: int,
    patches_per_image: int,
    sampling_seed: int,
    split_seed: int,
    synthetic: bool,
    chunk_size: int = 64,
    num_leading_components: int | None = None,
    bands: Mapping[str, Sequence[int]] = (),
) -> SplitHalfStability:
    """Image-disjoint split-half subspace stability check (plan §3.2/§14.3): builds
    two independent bases from disjoint image halves and reports how far apart
    their leading subspaces are, so unstable/near-degenerate directions are
    visible rather than assumed away.
    """
    first_half, second_half = manifest.split_disjoint_halves(split_seed)
    basis_a = build_pca_basis(
        first_half, encoder, patch_size=patch_size, channels=channels,
        patches_per_image=patches_per_image, sampling_seed=sampling_seed,
        chunk_size=chunk_size, synthetic=synthetic,
    )
    basis_b = build_pca_basis(
        second_half, encoder, patch_size=patch_size, channels=channels,
        patches_per_image=patches_per_image, sampling_seed=sampling_seed,
        chunk_size=chunk_size, synthetic=synthetic,
    )
    leading_count = num_leading_components or basis_a.components.shape[1]
    return compare_basis_subspaces(
        basis_a,
        basis_b,
        num_leading_components=leading_count,
        bands=bands,
        split_seed=split_seed,
        num_images_a=len(first_half),
        num_images_b=len(second_half),
    )


def save_basis(basis: PCABasis, path: str | Path) -> None:
    payload = {
        "components": basis.components,
        "eigenvalues": basis.eigenvalues,
        "mean": basis.mean,
        "patch_size": basis.patch_size,
        "channels": basis.channels,
        "num_samples": basis.num_samples,
        "metadata": basis.metadata,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, Path(path))


def load_basis(
    path: str | Path,
    *,
    allow_synthetic: bool = False,
    expected_patch_size: int | None = None,
    expected_channels: int | None = None,
) -> PCABasis:
    """Rejects (rather than silently accepting) a format mismatch, an
    unrequested synthetic basis, or a patch/channel mismatch against the
    caller's expected codec shape.
    """
    payload = torch.load(Path(path), weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("format_version") != BASIS_FORMAT_VERSION:
        raise ValueError(
            f"basis file format_version mismatch: {metadata.get('format_version')!r} != {BASIS_FORMAT_VERSION!r}"
        )
    if metadata.get("synthetic") and not allow_synthetic:
        raise ValueError("refusing to load a synthetic basis without allow_synthetic=True")
    if expected_patch_size is not None and payload["patch_size"] != expected_patch_size:
        raise ValueError(f"basis patch_size mismatch: expected {expected_patch_size}, got {payload['patch_size']}")
    if expected_channels is not None and payload["channels"] != expected_channels:
        raise ValueError(f"basis channels mismatch: expected {expected_channels}, got {payload['channels']}")
    return PCABasis(
        components=payload["components"],
        eigenvalues=payload["eigenvalues"],
        mean=payload["mean"],
        patch_size=payload["patch_size"],
        channels=payload["channels"],
        num_samples=payload["num_samples"],
        metadata=metadata,
    )


def basis_vector_patch_frequency_centroid(component: torch.Tensor, patch_size: int, channels: int) -> float:
    """Weighted-radius centroid of one basis vector's patch-internal power
    spectrum (plan §3.3): r_bar_i = sum_nu |nu| R_i(nu) / sum_nu R_i(nu),
    R_i(nu) = sum_c |FFT_{pxp} v_i(c,:,:)(nu)|^2. Summed directly per frequency
    point rather than per annular bin, so no bin-count weighting is needed
    (the plan explicitly calls out that mistake in an earlier draft).

    Descriptive only -- never used to assign a structural/frequency role.
    """
    patch = component.reshape(channels, patch_size, patch_size)
    fft = torch.fft.fft2(patch.to(torch.float64), dim=(-2, -1))
    power = fft.abs().square().sum(dim=0)  # sum over channels -> (p, p)
    fy = torch.fft.fftfreq(patch_size) * patch_size
    fx = torch.fft.fftfreq(patch_size) * patch_size
    radius = torch.sqrt(fy.view(-1, 1).square() + fx.view(1, -1).square())
    total_power = power.sum()
    if total_power <= 0:
        return 0.0
    return float((radius * power).sum() / total_power)


def basis_frequency_centroids(components: torch.Tensor, patch_size: int, channels: int) -> torch.Tensor:
    d = components.shape[1]
    return torch.tensor(
        [basis_vector_patch_frequency_centroid(components[:, i], patch_size, channels) for i in range(d)]
    )
