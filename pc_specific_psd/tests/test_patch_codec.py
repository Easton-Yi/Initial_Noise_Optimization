import unittest

import torch

from pc_specific_psd.patch_codec import (
    GridSpec,
    NonOverlapCodec,
    OverlapCodec,
    centered_grid,
    extract_patches,
    extract_patches_reference,
    scatter_patches,
)


def _random_orthonormal(d: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn((d, d), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix)
    return q.to(torch.float32)


class CenteredGridTests(unittest.TestCase):
    def test_exact_multiple(self):
        grid = centered_grid(16, 16, 4)
        self.assertEqual((grid.n_rows, grid.n_cols), (4, 4))
        self.assertEqual((grid.origin_row, grid.origin_col), (0, 0))
        self.assertEqual(grid.n_patches, 16)
        self.assertEqual(grid.coverage_fraction, 1.0)

    def test_boundary_remainder_centered(self):
        # 18 / 4 = 4 remainder 2 -> origin offset floor(2/2) = 1 on each axis.
        grid = centered_grid(18, 18, 4)
        self.assertEqual((grid.n_rows, grid.n_cols), (4, 4))
        self.assertEqual((grid.origin_row, grid.origin_col), (1, 1))
        self.assertLess(grid.coverage_fraction, 1.0)

    def test_asymmetric_remainder(self):
        # 19 / 4 = 4 remainder 3 -> origin offset floor(3/2) = 1, one row/col left over on the far edge.
        grid = centered_grid(19, 17, 4)
        self.assertEqual((grid.n_rows, grid.n_cols), (4, 4))
        self.assertEqual(grid.origin_row, (19 - 16) // 2)
        self.assertEqual(grid.origin_col, (17 - 16) // 2)

    def test_rejects_patch_too_large(self):
        with self.assertRaises(ValueError):
            centered_grid(3, 3, 4)

    def test_rejects_nonpositive_patch(self):
        with self.assertRaises(ValueError):
            centered_grid(8, 8, 0)


class ExtractPatchesTests(unittest.TestCase):
    def _check_ref_vs_vec(self, height, width, patch_size, channels=3, batch=2):
        grid = centered_grid(height, width, patch_size)
        latents = torch.randn(batch, channels, height, width)
        ref = extract_patches_reference(latents, grid)
        vec = extract_patches(latents, grid)
        torch.testing.assert_close(ref, vec)

    def test_even_exact(self):
        self._check_ref_vs_vec(16, 16, 4)

    def test_odd_patch_size(self):
        self._check_ref_vs_vec(17, 17, 3)

    def test_boundary_remainder(self):
        self._check_ref_vs_vec(18, 22, 4)

    def test_rectangular(self):
        self._check_ref_vs_vec(20, 12, 4, channels=4, batch=3)

    def test_shape_mismatch_raises(self):
        grid = centered_grid(16, 16, 4)
        latents = torch.randn(1, 3, 8, 8)
        with self.assertRaises(ValueError):
            extract_patches(latents, grid)

    def test_scatter_roundtrip_and_boundary_untouched(self):
        height, width, patch_size, channels = 18, 18, 4, 3
        grid = centered_grid(height, width, patch_size)
        latents = torch.randn(2, channels, height, width)
        patches = extract_patches(latents, grid)
        template = torch.full((2, channels, height, width), fill_value=-999.0)
        reconstructed = scatter_patches(patches, grid, channels, template)
        # Covered region reproduces the original exactly.
        covered = reconstructed[:, :, grid.origin_row:grid.origin_row + grid.n_rows * patch_size,
                                 grid.origin_col:grid.origin_col + grid.n_cols * patch_size]
        original_covered = latents[:, :, grid.origin_row:grid.origin_row + grid.n_rows * patch_size,
                                    grid.origin_col:grid.origin_col + grid.n_cols * patch_size]
        torch.testing.assert_close(covered, original_covered)
        # Boundary (outside the grid) is left as the template's untouched value.
        self.assertTrue(torch.all(reconstructed[:, :, 0, :] == -999.0))


class NonOverlapCodecTests(unittest.TestCase):
    def setUp(self):
        self.channels, self.patch_size = 4, 4
        self.d = self.channels * self.patch_size * self.patch_size
        self.grid = centered_grid(16, 16, self.patch_size)
        self.basis = _random_orthonormal(self.d, seed=42)
        self.codec = NonOverlapCodec(self.grid, self.basis, self.channels)

    def test_encode_decode_identity_on_covered_region(self):
        latents = torch.randn(2, self.channels, 16, 16)
        coeffs = self.codec.encode(latents)
        template = torch.zeros_like(latents)
        reconstructed = self.codec.decode(coeffs, template)
        torch.testing.assert_close(reconstructed, latents)

    def test_decode_leaves_boundary_from_template(self):
        grid = centered_grid(18, 18, self.patch_size)
        codec = NonOverlapCodec(grid, self.basis, self.channels)
        latents = torch.randn(1, self.channels, 18, 18)
        coeffs = codec.encode(latents)
        template = torch.full_like(latents, fill_value=-5.0)
        reconstructed = codec.decode(coeffs, template)
        self.assertTrue(torch.all(reconstructed[:, :, 0, :] == -5.0))

    def test_rotate_block_theta_zero_is_identity(self):
        latents = torch.randn(2, self.channels, 16, 16)
        donor = torch.randn(2, self.channels, 16, 16)
        coeffs = self.codec.encode(latents)
        donor_coeffs = self.codec.encode(donor)
        group = range(0, 5)
        rotated = self.codec.rotate_block(coeffs, donor_coeffs, group, theta=0.0)
        torch.testing.assert_close(rotated, coeffs)

    def test_rotate_block_ref_vs_vec(self):
        latents = torch.randn(2, self.channels, 16, 16)
        donor = torch.randn(2, self.channels, 16, 16)
        coeffs = self.codec.encode(latents)
        donor_coeffs = self.codec.encode(donor)
        group = range(2, 7)
        ref = self.codec.rotate_block_reference(coeffs, donor_coeffs, group, theta=0.37)
        vec = self.codec.rotate_block(coeffs, donor_coeffs, group, theta=0.37)
        torch.testing.assert_close(ref, vec)

    def test_rotate_block_only_touches_group_indices(self):
        latents = torch.randn(2, self.channels, 16, 16)
        donor = torch.randn(2, self.channels, 16, 16)
        coeffs = self.codec.encode(latents)
        donor_coeffs = self.codec.encode(donor)
        group = range(3, 6)
        rotated = self.codec.rotate_block(coeffs, donor_coeffs, group, theta=0.5)
        outside = [i for i in range(self.d) if i not in group]
        torch.testing.assert_close(rotated[..., outside], coeffs[..., outside])
        # And at least the in-group values actually changed (rotation isn't a no-op here).
        self.assertFalse(torch.allclose(rotated[..., list(group)], coeffs[..., list(group)]))

    def test_rejects_non_square_basis(self):
        with self.assertRaises(ValueError):
            NonOverlapCodec(self.grid, torch.randn(self.d, self.d, 1), self.channels)


class OverlapCodecTests(unittest.TestCase):
    def setUp(self):
        self.channels = 3

    def _make_codec(self, patch_size):
        d = self.channels * patch_size * patch_size
        basis = _random_orthonormal(d, seed=7 + patch_size)
        return OverlapCodec(basis, patch_size, self.channels), d

    def test_circular_patches_ref_vs_vec_odd_patch(self):
        codec, _ = self._make_codec(patch_size=3)
        latents = torch.randn(1, self.channels, 8, 8)
        ref = codec._circular_patches_reference(latents)
        vec = codec._circular_patches(latents)
        torch.testing.assert_close(ref, vec)

    def test_circular_patches_ref_vs_vec_even_patch(self):
        codec, _ = self._make_codec(patch_size=4)
        latents = torch.randn(1, self.channels, 8, 8)
        ref = codec._circular_patches_reference(latents)
        vec = codec._circular_patches(latents)
        torch.testing.assert_close(ref, vec)

    def test_encode_ref_vs_vec(self):
        codec, _ = self._make_codec(patch_size=3)
        latents = torch.randn(1, self.channels, 8, 8)
        ref = codec.encode(latents, reference=True)
        vec = codec.encode(latents, reference=False)
        torch.testing.assert_close(ref, vec, atol=1e-5, rtol=1e-5)

    def test_decode_center_ref_vs_vec(self):
        codec, d = self._make_codec(patch_size=3)
        coeffs = torch.randn(1, d, 6, 6)
        ref = codec.decode_center_reference(coeffs)
        vec = codec.decode_center(coeffs)
        torch.testing.assert_close(ref, vec, atol=1e-5, rtol=1e-5)

    def test_full_roundtrip_identity(self):
        codec, _ = self._make_codec(patch_size=3)
        latents = torch.randn(1, self.channels, 8, 8)
        reconstructed = codec.project_all_ones_is_identity_check(latents)
        torch.testing.assert_close(reconstructed, latents, atol=1e-5, rtol=1e-5)

    def test_full_roundtrip_identity_even_patch(self):
        codec, _ = self._make_codec(patch_size=4)
        latents = torch.randn(1, self.channels, 8, 8)
        reconstructed = codec.project_all_ones_is_identity_check(latents)
        torch.testing.assert_close(reconstructed, latents, atol=1e-5, rtol=1e-5)

    def test_circular_shift_equivariance(self):
        codec, _ = self._make_codec(patch_size=3)
        latents = torch.randn(1, self.channels, 8, 8)
        shifted = torch.roll(latents, shifts=(2, 3), dims=(-2, -1))
        coeffs_direct = codec.encode(latents)
        coeffs_shifted = codec.encode(shifted)
        coeffs_direct_then_shifted = torch.roll(coeffs_direct, shifts=(2, 3), dims=(-2, -1))
        torch.testing.assert_close(coeffs_shifted, coeffs_direct_then_shifted, atol=1e-5, rtol=1e-5)

    def test_rejects_non_2d_basis(self):
        with self.assertRaises(ValueError):
            OverlapCodec(torch.randn(4, 4, 1), patch_size=2, channels=1)


class CodecsAreDistinctTests(unittest.TestCase):
    def test_non_overlap_and_overlap_give_different_results(self):
        channels, patch_size = 2, 3
        d = channels * patch_size * patch_size
        basis = _random_orthonormal(d, seed=99)
        grid = centered_grid(9, 9, patch_size)
        latents = torch.randn(1, channels, 9, 9)

        non_overlap = NonOverlapCodec(grid, basis, channels)
        overlap = OverlapCodec(basis, patch_size, channels)

        non_overlap_coeffs = non_overlap.encode(latents)  # (batch, n_patches, d)
        overlap_coeffs = overlap.encode(latents)  # (batch, d, H, W)

        # Different shapes entirely: non-overlap yields one coefficient vector per
        # disjoint patch (n_patches = 9 for a 9x9 plane with p=3), overlap yields a
        # full-resolution per-pixel coefficient map (H*W = 81 positions).
        self.assertEqual(non_overlap_coeffs.shape[1], grid.n_patches)
        self.assertEqual(overlap_coeffs.shape[-2:], (9, 9))
        self.assertNotEqual(non_overlap_coeffs.numel(), overlap_coeffs.numel())


if __name__ == "__main__":
    unittest.main()
