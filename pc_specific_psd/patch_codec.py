"""Two structurally distinct patch/PCA codecs over VAE latent tensors.

``NonOverlapCodec`` is used only for probing (Phase B): a centered grid of
disjoint ``p x p`` patches, one PCA coefficient vector per patch, block
rotation applied in coefficient space. ``OverlapCodec`` is used only for the
final PSD editor (Phase C): a stride-1, circularly-padded coefficient map per
PC, edited in the frequency domain. The two are never interchangeable and a
test in tests/test_patch_codec.py asserts they give different results on the
same input.

Every public function has a reference (explicit Python-loop) implementation
suffixed ``_reference`` alongside a vectorized implementation; both are kept
permanently as test fixtures, not thrown away after validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GridSpec:
    """A centered, non-overlapping patch grid over an ``H x W`` plane."""
    height: int
    width: int
    patch_size: int
    n_rows: int
    n_cols: int
    origin_row: int
    origin_col: int

    @property
    def n_patches(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def coverage_fraction(self) -> float:
        return (self.n_rows * self.patch_size * self.n_cols * self.patch_size) / (self.height * self.width)


def centered_grid(height: int, width: int, patch_size: int) -> GridSpec:
    """Centered patch grid per plan §14.3: floor division, centered remainder.

    ``n_H = floor(H / p)``, ``n_W = floor(W / p)``, origin
    ``(floor((H - n_H p) / 2), floor((W - n_W p) / 2))``. No zero-padding, no
    edge duplication: boundary remainder rows/columns are simply outside the
    grid and are left untouched by any codec built on this grid.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    n_rows, n_cols = height // patch_size, width // patch_size
    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"Patch size {patch_size} does not fit in a {height}x{width} plane")
    origin_row = (height - n_rows * patch_size) // 2
    origin_col = (width - n_cols * patch_size) // 2
    return GridSpec(height, width, patch_size, n_rows, n_cols, origin_row, origin_col)


def extract_patches_reference(latents: torch.Tensor, grid: GridSpec) -> torch.Tensor:
    """Explicit-loop reference: returns (batch, n_patches, C*p*p)."""
    batch, channels = latents.shape[0], latents.shape[1]
    d = channels * grid.patch_size * grid.patch_size
    out = torch.zeros((batch, grid.n_patches, d), dtype=latents.dtype)
    p = grid.patch_size
    for b in range(batch):
        index = 0
        for row in range(grid.n_rows):
            for col in range(grid.n_cols):
                r0 = grid.origin_row + row * p
                c0 = grid.origin_col + col * p
                patch = latents[b, :, r0:r0 + p, c0:c0 + p].reshape(-1)
                out[b, index] = patch
                index += 1
    return out


def extract_patches(latents: torch.Tensor, grid: GridSpec) -> torch.Tensor:
    """Vectorized patch extraction; (batch, n_patches, C*p*p), joint-channel flatten."""
    if latents.ndim != 4:
        raise ValueError("latents must have shape (batch, channels, height, width)")
    batch, channels, height, width = latents.shape
    if (height, width) != (grid.height, grid.width):
        raise ValueError("latents shape does not match the grid it was built for")
    p = grid.patch_size
    cropped = latents[:, :, grid.origin_row:grid.origin_row + grid.n_rows * p, grid.origin_col:grid.origin_col + grid.n_cols * p]
    # (batch, C, n_rows, p, n_cols, p) -> (batch, n_rows, n_cols, C, p, p) -> (batch, n_patches, C*p*p)
    unfolded = cropped.reshape(batch, channels, grid.n_rows, p, grid.n_cols, p)
    unfolded = unfolded.permute(0, 2, 4, 1, 3, 5).contiguous()
    return unfolded.reshape(batch, grid.n_patches, channels * p * p)


def scatter_patches(patches: torch.Tensor, grid: GridSpec, channels: int, template: torch.Tensor) -> torch.Tensor:
    """Inverse of extract_patches: writes edited patches back into a copy of template.

    ``template`` supplies the untouched boundary region (patches never zero
    it out); only the covered grid cells are overwritten.
    """
    out = template.clone()
    batch = patches.shape[0]
    p = grid.patch_size
    reshaped = patches.reshape(batch, grid.n_rows, grid.n_cols, channels, p, p).permute(0, 3, 1, 4, 2, 5).contiguous()
    reshaped = reshaped.reshape(batch, channels, grid.n_rows * p, grid.n_cols * p)
    out[:, :, grid.origin_row:grid.origin_row + grid.n_rows * p, grid.origin_col:grid.origin_col + grid.n_cols * p] = reshaped
    return out


class NonOverlapCodec:
    """Probing-only codec: disjoint patches, PCA coefficients, block rotation.

    ``a'_B = cos(theta) * a_B + sin(theta) * eta_B`` where ``B`` is a PC-index
    group (an index range into the ``d``-dim coefficient vector) and ``eta_B``
    is an independent donor draw restricted to the same coefficient indices.
    Everything outside the grid (boundary remainder) and every coefficient
    index outside ``B`` is provably unchanged.
    """

    def __init__(self, grid: GridSpec, basis: torch.Tensor, channels: int):
        if basis.ndim != 2:
            raise ValueError("basis must be a (d, d) orthonormal matrix, columns are components")
        self.grid, self.basis, self.channels = grid, basis, channels
        self.d = basis.shape[0]

    def encode(self, latents: torch.Tensor) -> torch.Tensor:
        patches = extract_patches(latents, self.grid)
        return patches @ self.basis  # (batch, n_patches, d)

    def decode(self, coefficients: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        patches = coefficients @ self.basis.T
        return scatter_patches(patches, self.grid, self.channels, template)

    def rotate_block_reference(self, coefficients: torch.Tensor, donor_coefficients: torch.Tensor, group: range, theta: float) -> torch.Tensor:
        """Explicit-loop reference implementation of the block rotation."""
        out = coefficients.clone()
        cos_t, sin_t = float(torch.cos(torch.tensor(theta))), float(torch.sin(torch.tensor(theta)))
        for b in range(coefficients.shape[0]):
            for patch in range(coefficients.shape[1]):
                for index in group:
                    out[b, patch, index] = cos_t * coefficients[b, patch, index] + sin_t * donor_coefficients[b, patch, index]
        return out

    def rotate_block(self, coefficients: torch.Tensor, donor_coefficients: torch.Tensor, group: range, theta: float) -> torch.Tensor:
        """Vectorized block rotation; only ``group`` coefficient indices change."""
        if coefficients.shape != donor_coefficients.shape:
            raise ValueError("coefficients and donor_coefficients must have the same shape")
        out = coefficients.clone()
        cos_t, sin_t = float(torch.cos(torch.tensor(float(theta)))), float(torch.sin(torch.tensor(float(theta))))
        idx = list(group)
        out[..., idx] = cos_t * coefficients[..., idx] + sin_t * donor_coefficients[..., idx]
        return out


class OverlapCodec:
    """Final-editor-only codec: stride-1, circularly-padded per-PC coefficient maps.

    For every principal component ``i`` and every spatial position ``(x, y)``
    (stride 1, circular padding), ``a_i(x, y) = v_i^T q_{x,y}`` where
    ``q_{x,y}`` is the ``C*p*p`` patch centered at ``(x, y)``. Editing happens
    per-PC in the frequency domain of ``a_i``; reconstruction is
    ``Center(sum_i a_i(x, y) v_i)`` -- NOT overlap-add averaging, and each of
    the ``d`` coefficient maps is NOT an independent Gaussian source (see
    spectral.py's transfer-matrix diagnostics).
    """

    def __init__(self, basis: torch.Tensor, patch_size: int, channels: int):
        if basis.ndim != 2:
            raise ValueError("basis must be a (d, d) orthonormal matrix")
        self.basis, self.patch_size, self.channels = basis, patch_size, channels
        self.d = basis.shape[0]

    def _circular_patches_reference(self, latents: torch.Tensor) -> torch.Tensor:
        """Explicit-loop reference: (batch, H, W, C*p*p) circularly-padded patches.

        Flattens channel-major then row-in-patch then col-in-patch, matching
        both ``extract_patches`` (the non-overlap codec) and the vectorized
        ``_circular_patches`` below -- required so a patch vector reshaped
        back to ``(channels, p, p)`` in ``decode_center`` addresses the
        correct center pixel.
        """
        batch, channels, height, width = latents.shape
        p = self.patch_size
        left = p // 2
        out = torch.zeros((batch, height, width, channels * p * p), dtype=latents.dtype)
        for b in range(batch):
            for y in range(height):
                for x in range(width):
                    values = []
                    for c in range(channels):
                        for dy in range(p):
                            for dx in range(p):
                                ry = (y - left + dy) % height
                                rx = (x - left + dx) % width
                                values.append(latents[b, c, ry, rx])
                    out[b, y, x] = torch.stack(values)
        return out

    def _circular_patches(self, latents: torch.Tensor) -> torch.Tensor:
        """Vectorized circular-padded stride-1 patch extraction via unfold."""
        batch, channels, height, width = latents.shape
        p = self.patch_size
        left = p // 2
        right = p - 1 - left
        padded = torch.cat([latents[:, :, -left:, :], latents, latents[:, :, :right, :]], dim=2) if left or right else latents
        padded = torch.cat([padded[:, :, :, -left:], padded, padded[:, :, :, :right]], dim=3) if left or right else padded
        unfolded = padded.unfold(2, p, 1).unfold(3, p, 1)  # (batch, C, H, W, p, p)
        return unfolded.permute(0, 2, 3, 1, 4, 5).reshape(batch, height, width, channels * p * p)

    def encode(self, latents: torch.Tensor, *, reference: bool = False) -> torch.Tensor:
        """Returns (batch, d, H, W) coefficient maps ``a_i(x, y)``."""
        patches = self._circular_patches_reference(latents) if reference else self._circular_patches(latents)
        coeffs = patches @ self.basis  # (batch, H, W, d)
        return coeffs.permute(0, 3, 1, 2)

    def decode_center_reference(self, coefficient_maps: torch.Tensor) -> torch.Tensor:
        """Explicit-loop reference for Center() synthesis: pick the patch's own center pixel."""
        batch, d, height, width = coefficient_maps.shape
        channels = self.channels
        p = self.patch_size
        left = p // 2
        out = torch.zeros((batch, channels, height, width))
        for b in range(batch):
            for y in range(height):
                for x in range(width):
                    coeff = coefficient_maps[b, :, y, x]
                    patch = self.basis @ coeff  # (C*p*p,)
                    patch = patch.reshape(channels, p, p)
                    out[b, :, y, x] = patch[:, left, left]
        return out

    def decode_center(self, coefficient_maps: torch.Tensor) -> torch.Tensor:
        """Vectorized Center() synthesis: reconstruct each patch, keep only its center pixel.

        For a stride-1 circular codec every spatial position owns exactly one
        patch (the one centered there), so ``Center()`` synthesis is just
        "reconstruct that one patch and read off its center channel vector" --
        it is not an overlap-add average across neighboring patches.
        """
        batch, d, height, width = coefficient_maps.shape
        channels = self.channels
        p = self.patch_size
        left = p // 2
        flat = coefficient_maps.permute(0, 2, 3, 1).reshape(batch * height * width, d)  # (N, d)
        reconstructed = flat @ self.basis.T  # (N, C*p*p)
        reconstructed = reconstructed.reshape(batch, height, width, channels, p, p)
        center = reconstructed[:, :, :, :, left, left]  # (batch, H, W, C)
        return center.permute(0, 3, 1, 2)

    def project_all_ones_is_identity_check(self, latents: torch.Tensor) -> torch.Tensor:
        """Round-trip with an untouched coefficient map; used only by tests."""
        coeffs = self.encode(latents)
        return self.decode_center(coeffs)
