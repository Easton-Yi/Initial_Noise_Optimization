import unittest

import torch

from pc_specific_psd import spectral


class RadialPSDParsevalTests(unittest.TestCase):
    def _check_parseval(self, shape, num_bins):
        torch.manual_seed(0)
        tensor = torch.randn(*shape)
        result = spectral.radial_psd(tensor, num_bins)
        direct = spectral.parseval_energy(tensor)
        self.assertAlmostEqual(result.total_energy, direct, places=9)
        self.assertFalse(torch.isnan(result.power).any())
        self.assertFalse(torch.isnan(torch.tensor(result.total_energy)))

    def test_even_square(self):
        self._check_parseval((4, 3, 16, 16), num_bins=8)

    def test_odd_width(self):
        self._check_parseval((3, 2, 16, 15), num_bins=6)

    def test_odd_height(self):
        self._check_parseval((3, 2, 15, 16), num_bins=6)

    def test_odd_both(self):
        self._check_parseval((2, 2, 15, 17), num_bins=5)

    def test_bare_channel_plane_no_batch(self):
        self._check_parseval((4, 12, 12), num_bins=5)

    def test_full_fft2_agrees_with_rfft_based_computation(self):
        torch.manual_seed(1)
        tensor = torch.randn(5, 4, 14, 14)
        result = spectral.radial_psd(tensor, num_bins=7)
        fft_full = torch.fft.fft2(tensor.to(torch.float64), dim=(-2, -1))
        power_full = fft_full.abs().square()
        mean_power_full = power_full.reshape(-1, 14, 14).mean(dim=0)
        total_full = float(mean_power_full.sum()) / (14 * 14) ** 2
        self.assertAlmostEqual(result.total_energy, total_full, places=9)

    def test_counts_cover_every_frequency_bin(self):
        height, width, num_bins = 10, 9, 6
        result = spectral.radial_psd(torch.randn(2, 3, height, width), num_bins)
        n_rfft_cols = width // 2 + 1
        self.assertEqual(int(result.counts.sum().item()), height * n_rfft_cols)

    def test_shift_invariance(self):
        torch.manual_seed(2)
        tensor = torch.randn(3, 4, 12, 12)
        shifted = torch.roll(tensor, shifts=(3, 5), dims=(-2, -1))
        original = spectral.radial_psd(tensor, num_bins=6)
        rolled = spectral.radial_psd(shifted, num_bins=6)
        torch.testing.assert_close(original.power, rolled.power)
        self.assertAlmostEqual(original.total_energy, rolled.total_energy, places=9)


class AngularPSDTests(unittest.TestCase):
    def test_no_nan_and_correct_shape(self):
        tensor = torch.randn(2, 3, 10, 10)
        result = spectral.angular_psd(tensor, num_angular_bins=8)
        self.assertEqual(result.shape, (8,))
        self.assertFalse(torch.isnan(result).any())


class SpatialVarianceMapTests(unittest.TestCase):
    def test_zero_variance_for_constant_batch(self):
        tensor = torch.ones(5, 3, 4, 4) * 2.0
        variance = spectral.spatial_variance_map(tensor)
        self.assertTrue(torch.allclose(variance, torch.zeros_like(variance)))

    def test_requires_leading_batch_dim(self):
        with self.assertRaises(ValueError):
            spectral.spatial_variance_map(torch.randn(4, 4))


class TransferMatrixDiagnosticsTests(unittest.TestCase):
    def test_identity_operator_has_unit_singular_values(self):
        def identity_op(t):
            return t

        diag = spectral.impulse_response_transfer_matrix(identity_op, channels=4, height=8, width=8, num_freq_samples=12, seed=0)
        torch.testing.assert_close(diag.singular_values, torch.ones_like(diag.singular_values), atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(diag.worst_condition_number, 1.0, places=5)

    def test_near_singular_operator_is_flagged(self):
        def zero_one_channel(t):
            out = t.clone()
            out[:, 1, :, :] = 0.0
            return out

        diag = spectral.impulse_response_transfer_matrix(zero_one_channel, channels=4, height=8, width=8, num_freq_samples=12, seed=0)
        self.assertGreater(diag.worst_condition_number, 1e6)

    def test_scaling_operator_singular_values_match_scale(self):
        scale = 3.0

        def scaling_op(t):
            return t * scale

        diag = spectral.impulse_response_transfer_matrix(scaling_op, channels=2, height=6, width=6, num_freq_samples=8, seed=3)
        torch.testing.assert_close(diag.singular_values, torch.full_like(diag.singular_values, scale), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
