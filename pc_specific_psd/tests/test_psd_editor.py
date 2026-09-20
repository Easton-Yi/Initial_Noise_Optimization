import unittest

import torch

from pc_specific_psd import psd_editor
from pc_specific_psd.compat_generation import same_phase_floor
from pc_specific_psd.patch_codec import OverlapCodec


def _random_orthonormal(d: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn((d, d), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix)
    return q.to(torch.float32)


class FrozenConstantsTests(unittest.TestCase):
    def test_exact_values(self):
        self.assertEqual(psd_editor.SAME_PHASE_ALPHA, 0.9)
        self.assertEqual(psd_editor.SAME_PHASE_GAMMA, 0.05)


class ReferenceAmplitudeResponseTests(unittest.TestCase):
    def test_dc_is_exactly_one(self):
        h_ref = psd_editor.reference_amplitude_response(8, 8)
        self.assertAlmostEqual(float(h_ref[0, 0]), 1.0, places=6)

    def test_matches_same_phase_floor_internal_multiplier(self):
        # Applying h_ref(r) directly to a random tensor's rfft must reproduce
        # noise_init.same_phase_floor bit-for-bit (up to fp tolerance): this
        # is the exact multiplier same_phase_floor computes internally.
        torch.manual_seed(0)
        base_white = torch.randn(2, 4, 8, 10)
        h_ref = psd_editor.reference_amplitude_response(8, 10)
        direct = torch.fft.irfft2(torch.fft.rfft2(base_white, dim=(-2, -1)) * h_ref, s=(8, 10), dim=(-2, -1))
        expected = same_phase_floor(base_white, psd_editor.SAME_PHASE_ALPHA, psd_editor.SAME_PHASE_GAMMA)
        torch.testing.assert_close(direct, expected, atol=1e-5, rtol=1e-5)

    def test_monotonically_non_increasing_along_first_row(self):
        h_ref = psd_editor.reference_amplitude_response(16, 16)
        row = h_ref[0]
        self.assertTrue(torch.all(row[1:] <= row[:-1] + 1e-6))


class LowFrequencyGateTests(unittest.TestCase):
    def test_dc_is_exactly_one(self):
        gate = psd_editor.low_frequency_gate(8, 8, r_s=2.0, beta=2.0)
        self.assertAlmostEqual(float(gate[0, 0]), 1.0, places=6)

    def test_bounded_in_zero_one(self):
        gate = psd_editor.low_frequency_gate(16, 16, r_s=3.0, beta=1.5)
        self.assertTrue(torch.all(gate > 0.0))
        self.assertTrue(torch.all(gate <= 1.0 + 1e-6))

    def test_rejects_nonpositive_r_s(self):
        with self.assertRaises(ValueError):
            psd_editor.low_frequency_gate(8, 8, r_s=0.0, beta=1.0)

    def test_rejects_nonpositive_beta(self):
        with self.assertRaises(ValueError):
            psd_editor.low_frequency_gate(8, 8, r_s=1.0, beta=-1.0)


class GroupTransferMultiplierTests(unittest.TestCase):
    def test_tau_zero_is_exactly_one_everywhere(self):
        t_b = psd_editor.group_transfer_multiplier(16, 16, r_s=2.0, beta=2.0, tau=0.0)
        torch.testing.assert_close(t_b, torch.ones_like(t_b))

    def test_matches_closed_form(self):
        r_s, beta, tau = 2.5, 1.7, 0.6
        gate = psd_editor.low_frequency_gate(10, 12, r_s, beta)
        expected = torch.exp(0.5 * tau * gate)
        actual = psd_editor.group_transfer_multiplier(10, 12, r_s, beta, tau)
        torch.testing.assert_close(actual, expected)

    def test_positive_and_negative_tau_are_reciprocal_in_log(self):
        r_s, beta = 2.0, 2.0
        plus = psd_editor.group_transfer_multiplier(8, 8, r_s, beta, tau=0.8)
        minus = psd_editor.group_transfer_multiplier(8, 8, r_s, beta, tau=-0.8)
        torch.testing.assert_close(plus.log(), -minus.log(), atol=1e-6, rtol=1e-6)


class ApplyPsdEditTauZeroTests(unittest.TestCase):
    def setUp(self):
        self.channels, self.patch_size = 3, 3
        self.d = self.channels * self.patch_size * self.patch_size
        self.basis = _random_orthonormal(self.d, seed=11)
        self.codec = OverlapCodec(self.basis, self.patch_size, self.channels)
        torch.manual_seed(1)
        self.latents = torch.randn(1, self.channels, 8, 8)

    def test_vectorized_path_matches_same_phase_floor(self):
        edited = psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[0, 1], tau=0.0, r_s=2.0, beta=2.0)
        expected = same_phase_floor(self.latents, psd_editor.SAME_PHASE_ALPHA, psd_editor.SAME_PHASE_GAMMA)
        torch.testing.assert_close(edited, expected, atol=1e-4, rtol=1e-4)

    def test_forced_full_codec_path_matches_shortcut_and_reference(self):
        # The plan's required check: force the full general codec path (loop
        # implementations, fast path disabled) at tau=0 and confirm it agrees
        # with both the exact scalar shortcut and the unmodified reference.
        forced = psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[2], tau=0.0, r_s=3.0, beta=1.5, reference=True)
        vectorized = psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[2], tau=0.0, r_s=3.0, beta=1.5, reference=False)
        shortcut = psd_editor.apply_psd_edit_tau_zero(self.latents)
        unmodified_reference = same_phase_floor(self.latents, psd_editor.SAME_PHASE_ALPHA, psd_editor.SAME_PHASE_GAMMA)
        torch.testing.assert_close(forced, vectorized, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(forced, shortcut, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(forced, unmodified_reference, atol=1e-4, rtol=1e-4)

    def test_tau_zero_result_independent_of_group_choice(self):
        # t_B(r; 0) == 1 regardless of which group is named, so the edit must
        # not depend on group_indices at all when tau == 0.
        edited_a = psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[0], tau=0.0, r_s=2.0, beta=2.0)
        edited_b = psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=list(range(self.d)), tau=0.0, r_s=5.0, beta=0.5)
        torch.testing.assert_close(edited_a, edited_b, atol=1e-5, rtol=1e-5)

    def test_apply_psd_edit_tau_zero_delegates_to_same_phase_floor(self):
        expected = same_phase_floor(self.latents, 0.9, 0.05)
        torch.testing.assert_close(psd_editor.apply_psd_edit_tau_zero(self.latents), expected)


class ApplyPsdEditGeneralPathTests(unittest.TestCase):
    def setUp(self):
        self.channels, self.patch_size = 3, 3
        self.d = self.channels * self.patch_size * self.patch_size
        self.basis = _random_orthonormal(self.d, seed=23)
        self.codec = OverlapCodec(self.basis, self.patch_size, self.channels)
        torch.manual_seed(2)
        self.latents = torch.randn(1, self.channels, 8, 8)

    def test_reference_and_vectorized_agree_at_nonzero_tau(self):
        ref = psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[0, 3, 5], tau=0.7, r_s=2.0, beta=1.8, reference=True)
        vec = psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[0, 3, 5], tau=0.7, r_s=2.0, beta=1.8, reference=False)
        torch.testing.assert_close(ref, vec, atol=1e-4, rtol=1e-4)

    def test_linearity(self):
        torch.manual_seed(3)
        other = torch.randn_like(self.latents)
        a, b = 1.3, -0.7
        combined = psd_editor.apply_psd_edit(self.codec, a * self.latents + b * other, group_indices=[1, 4], tau=0.5, r_s=2.5, beta=2.0)
        separate = a * psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[1, 4], tau=0.5, r_s=2.5, beta=2.0) \
            + b * psd_editor.apply_psd_edit(self.codec, other, group_indices=[1, 4], tau=0.5, r_s=2.5, beta=2.0)
        torch.testing.assert_close(combined, separate, atol=1e-4, rtol=1e-4)

    def test_circular_shift_equivariance(self):
        shifted = torch.roll(self.latents, shifts=(2, 3), dims=(-2, -1))
        direct = psd_editor.apply_psd_edit(self.codec, shifted, group_indices=[2, 6], tau=-0.4, r_s=2.0, beta=2.0)
        shift_then = torch.roll(
            psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[2, 6], tau=-0.4, r_s=2.0, beta=2.0),
            shifts=(2, 3), dims=(-2, -1),
        )
        torch.testing.assert_close(direct, shift_then, atol=1e-4, rtol=1e-4)

    def test_nonzero_tau_actually_changes_output_vs_tau_zero(self):
        tau_zero = psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[0, 1], tau=0.0, r_s=2.0, beta=2.0)
        tau_nonzero = psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[0, 1], tau=1.5, r_s=2.0, beta=2.0)
        self.assertFalse(torch.allclose(tau_zero, tau_nonzero, atol=1e-4, rtol=1e-4))

    def test_rejects_duplicate_group_indices(self):
        with self.assertRaises(ValueError):
            psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[0, 0], tau=0.5, r_s=2.0, beta=2.0)

    def test_rejects_out_of_range_group_indices(self):
        with self.assertRaises(ValueError):
            psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[self.d], tau=0.5, r_s=2.0, beta=2.0)

    def test_rejects_bad_gate_params_only_when_tau_nonzero(self):
        # r_s/beta are only consulted when tau != 0 (group_transfer_multiplier
        # is skipped entirely at tau == 0), so a degenerate gate must not
        # break a tau == 0 call.
        edited = psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[0], tau=0.0, r_s=-1.0, beta=-1.0)
        expected = psd_editor.apply_psd_edit_tau_zero(self.latents)
        torch.testing.assert_close(edited, expected, atol=1e-4, rtol=1e-4)
        with self.assertRaises(ValueError):
            psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[0], tau=0.5, r_s=-1.0, beta=-1.0)

    def test_rejects_wrong_latent_ndim(self):
        with self.assertRaises(ValueError):
            psd_editor.apply_psd_edit(self.codec, torch.randn(3, 8, 8), group_indices=[0], tau=0.5, r_s=2.0, beta=2.0)


class ExplicitTwoChannelRotationTests(unittest.TestCase):
    """Independent, manually-computed check of the group/complement split.

    Uses patch_size=1 so OverlapCodec.encode/decode_center reduce to a plain
    per-pixel channel rotation (no circular padding neighbourhood involved),
    letting the expected frequency-domain result be computed by hand with
    plain einsum + torch.fft calls rather than by re-using the module under
    test's own machinery.
    """

    def setUp(self):
        self.channels = 2
        self.basis = _random_orthonormal(self.channels, seed=5)  # (C, C): columns are the two PCs.
        self.codec = OverlapCodec(self.basis, patch_size=1, channels=self.channels)
        torch.manual_seed(4)
        self.latents = torch.randn(1, self.channels, 8, 10)

    def _manual_expected(self, group_indices, tau, r_s, beta):
        height, width = self.latents.shape[-2:]
        coeffs = torch.einsum("bchw,cd->bdhw", self.latents, self.basis)  # a_d(x,y)
        spectrum = torch.fft.rfft2(coeffs, dim=(-2, -1))
        h_ref = psd_editor.reference_amplitude_response(height, width)
        t_b = psd_editor.group_transfer_multiplier(height, width, r_s, beta, tau)
        filtered = spectrum.clone()
        for index in range(self.basis.shape[1]):
            filtered[:, index] = filtered[:, index] * h_ref * (t_b if index in group_indices else 1.0)
        edited_coeffs = torch.fft.irfft2(filtered, s=(height, width), dim=(-2, -1))
        return torch.einsum("bdhw,cd->bchw", edited_coeffs, self.basis)

    def test_matches_manual_computation_group_only(self):
        expected = self._manual_expected(group_indices={0}, tau=0.9, r_s=2.0, beta=1.5)
        actual = psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[0], tau=0.9, r_s=2.0, beta=1.5)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_matches_manual_computation_complement_only(self):
        expected = self._manual_expected(group_indices={1}, tau=-0.6, r_s=3.0, beta=2.2)
        actual = psd_editor.apply_psd_edit(self.codec, self.latents, group_indices=[1], tau=-0.6, r_s=3.0, beta=2.2)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
