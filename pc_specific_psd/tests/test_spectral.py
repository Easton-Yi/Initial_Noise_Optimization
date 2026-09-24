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
        self.assertEqual(int(result.counts.sum().item()), height * width)


    def test_annular_power_and_counts_match_full_fft_even_and_odd(self):
        for height, width, num_bins in ((8, 10, 6), (7, 9, 5), (7, 10, 5)):
            with self.subTest(height=height, width=width):
                torch.manual_seed(height * 100 + width)
                tensor = torch.randn(3, 2, height, width)
                actual = spectral.radial_psd(tensor, num_bins)
                fft = torch.fft.fft2(tensor.to(torch.float64), dim=(-2, -1))
                grid_power = fft.abs().square().reshape(-1, height, width).mean(dim=0)
                _, bins = spectral.radial_bin_index(height, width, num_bins, rfft=False)
                counts = torch.zeros(num_bins, dtype=torch.float64).scatter_add_(
                    0, bins.reshape(-1), torch.ones(height * width, dtype=torch.float64)
                )
                sums = torch.zeros(num_bins, dtype=torch.float64).scatter_add_(
                    0, bins.reshape(-1), grid_power.reshape(-1)
                )
                expected = sums / counts.clamp(min=1)
                torch.testing.assert_close(actual.counts, counts)
                torch.testing.assert_close(actual.power, expected, atol=1e-9, rtol=1e-9)

    def test_binning_metadata_records_edges_coordinates_and_band_ownership(self):
        metadata = spectral.radial_binning_metadata(8, 10, 6)
        self.assertEqual(metadata["version"], spectral.RADIAL_PSD_DEFINITION_VERSION)
        self.assertEqual(len(metadata["bin_edges"]), 7)
        self.assertEqual(len(metadata["frequency_coordinates"]["fy"]), 8)
        self.assertEqual(len(metadata["frequency_coordinates"]["fx_rfft"]), 6)
        ranges = [
            (band["start_bin_inclusive"], band["end_bin_exclusive"])
            for band in metadata["energy_bands"]
        ]
        self.assertEqual(tuple(ranges), spectral.low_mid_high_bin_ranges(6))
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 6)
        self.assertIn("higher bin", metadata["boundary_rule"])
        self.assertEqual(
            spectral.diagnostic_definition_versions(),
            {
                "radial_psd": spectral.RADIAL_PSD_DEFINITION_VERSION,
                "conditioning": spectral.CONDITIONING_DEFINITION_VERSION,
            },
        )

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
    def test_identical_operators_have_zero_covariance_distance(self):
        distance = spectral.linear_operator_covariance_distance(
            lambda tensor: tensor, lambda tensor: tensor, channels=2, height=6, width=6
        )
        self.assertAlmostEqual(distance, 0.0, places=12)

    def test_orthogonal_channel_rotation_moves_pairs_but_not_covariance(self):
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])

        def rotate(tensor):
            return torch.einsum("oc,bchw->bohw", rotation, tensor)

        bank = torch.randn(8, 2, 6, 6)
        from pc_specific_psd.calibration_v2 import paired_relative_l2
        self.assertGreater(paired_relative_l2(rotate(bank), bank), 0.1)
        distance = spectral.linear_operator_covariance_distance(
            lambda tensor: tensor, rotate, channels=2, height=6, width=6
        )
        self.assertAlmostEqual(distance, 0.0, places=12)

    def test_identity_operator_has_unit_singular_values(self):
        def identity_op(t):
            return t

        diag = spectral.impulse_response_transfer_matrix(identity_op, channels=4, height=8, width=8, num_freq_samples=12, seed=0)
        self.assertEqual(diag.frequencies.shape[0], 8 * (8 // 2 + 1))
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



    def test_complete_grid_finds_a_single_bad_frequency(self):
        height = width = 8
        bad_row, bad_col = 7, 3

        def sparse_bad_frequency(tensor):
            spectrum = torch.fft.rfft2(tensor, dim=(-2, -1))
            spectrum = spectrum.clone()
            spectrum[:, 1, bad_row, bad_col] *= 1e-8
            return torch.fft.irfft2(spectrum, s=(height, width), dim=(-2, -1))

        diag = spectral.impulse_response_transfer_matrix(
            sparse_bad_frequency, channels=2, height=height, width=width,
            num_freq_samples=1, seed=0,
        )
        self.assertEqual(diag.frequencies.shape[0], height * (width // 2 + 1))
        self.assertEqual(diag.worst_frequency, (bad_row, bad_col))
        self.assertGreater(diag.worst_condition_number, 1e6)

if __name__ == "__main__":    unittest.main()
