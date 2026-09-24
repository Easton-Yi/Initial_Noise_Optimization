import dataclasses
import math
import unittest
from unittest import mock

import torch

from pc_specific_psd import calibration_v2, expected_rms, psd_editor, spectral
from pc_specific_psd.patch_codec import OverlapCodec


class ExpectedEnergyFormulaTests(unittest.TestCase):
    def test_scalar_and_channel_mixing_operators(self):
        height, width, channels = 5, 6, 2
        scalar = 2.5
        scalar_response = spectral.linear_operator_frequency_response(
            lambda tensor: tensor * scalar, channels, height, width
        )
        self.assertAlmostEqual(
            spectral.expected_mean_square_from_frequency_response(scalar_response, width=width),
            scalar ** 2,
            places=10,
        )
        matrix = torch.tensor([[2.0, -0.5], [0.25, 1.5]])

        def mix(tensor):
            return torch.einsum("oc,bchw->bohw", matrix, tensor)

        response = spectral.linear_operator_frequency_response(mix, channels, height, width)
        expected = float(matrix.to(torch.float64).square().sum() / channels)
        actual = spectral.expected_mean_square_from_frequency_response(response, width=width)
        self.assertAlmostEqual(actual, expected, places=10)
        torch.manual_seed(11)
        bank = torch.randn(4096, channels, height, width)
        measured = float(mix(bank).to(torch.float64).square().mean())
        self.assertAlmostEqual(measured, expected, delta=0.04)


class FrozenExpectedRMSOperatorTests(unittest.TestCase):
    def setUp(self):
        self.codec = OverlapCodec(torch.eye(2), patch_size=1, channels=2)
        self.height, self.width, self.num_bins = 6, 7, 4
        self.reference_response = spectral.linear_operator_frequency_response(
            psd_editor.apply_psd_edit_tau_zero, 2, self.height, self.width
        )

    def _candidate(self, tau=0.5):
        return expected_rms.construct_candidate(
            self.codec, "G", (0,), tau, 4.0, 2.0,
            height=self.height, width=self.width, num_bins=self.num_bins,
            reference_response=self.reference_response,
        )

    def test_tau_zero_uses_reference_shortcut_and_scale_one(self):
        zero = self._candidate(0.0)
        self.assertEqual(zero.operator.scale, 1.0)
        base = torch.randn(3, 2, self.height, self.width)
        actual = expected_rms.apply_frozen_operator(self.codec, base, zero.operator)
        expected = psd_editor.apply_psd_edit_tau_zero(base)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_candidate_is_linear_translation_equivariant_and_response_reconstructs(self):
        candidate = self._candidate()
        torch.manual_seed(4)
        x = torch.randn(2, 2, self.height, self.width)
        y = torch.randn_like(x)
        apply = lambda value: expected_rms.apply_frozen_operator(self.codec, value, candidate.operator)
        torch.testing.assert_close(apply(x + y), apply(x) + apply(y), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            apply(torch.roll(x, shifts=(2, 3), dims=(-2, -1))),
            torch.roll(apply(x), shifts=(2, 3), dims=(-2, -1)),
            atol=2e-5, rtol=2e-5,
        )
        spectrum = torch.fft.rfft2(x.to(torch.float64), dim=(-2, -1))
        predicted = torch.einsum(
            "hwoc,bchw->bohw", candidate.response.to(torch.complex128), spectrum.to(torch.complex128)
        )
        reconstructed = torch.fft.irfft2(predicted, s=(self.height, self.width), dim=(-2, -1))
        torch.testing.assert_close(reconstructed, apply(x).to(torch.float64), atol=2e-5, rtol=2e-5)

    def test_fixed_scale_matches_reference_expected_energy_without_per_sample_normalization(self):
        candidate = self._candidate()
        reference_energy = spectral.expected_mean_square_from_frequency_response(
            self.reference_response, width=self.width
        )
        self.assertAlmostEqual(candidate.operator.expected_mean_square, reference_energy, places=6)
        torch.manual_seed(8)
        base = torch.randn(2, 2, self.height, self.width)
        base[1] *= 3.0
        output = expected_rms.apply_frozen_operator(self.codec, base, candidate.operator)
        sample_rms = output.to(torch.float64).reshape(2, -1).square().mean(dim=1).sqrt()
        self.assertGreater(float(sample_rms[1] / sample_rms[0]), 2.0)
        self.assertFalse(torch.allclose(sample_rms, torch.ones_like(sample_rms), atol=0.1))

    def test_fourier_control_matches_candidate_annuli_and_expected_energy(self):
        candidate = self._candidate()
        control = expected_rms.construct_fourier_control(candidate, self.reference_response)
        nonempty = candidate.radial_psd.counts > 0
        torch.testing.assert_close(
            control.radial_psd.power[nonempty], candidate.radial_psd.power[nonempty],
            atol=1e-10, rtol=1e-10,
        )
        self.assertAlmostEqual(
            control.radial_psd.total_expected_mean_square,
            candidate.radial_psd.total_expected_mean_square,
            places=10,
        )
        base = torch.randn(1, 2, self.height, self.width)
        candidate_output = expected_rms.apply_frozen_operator(self.codec, base, candidate.operator)
        control_output = expected_rms.apply_frozen_operator(self.codec, base, control.operator)
        self.assertEqual(candidate_output.shape, control_output.shape)
        self.assertFalse(torch.equal(candidate_output, control_output))

    def test_diagnostics_are_batch_size_invariant_and_report_injection_dtype(self):
        candidate = self._candidate(0.25)
        torch.manual_seed(12)
        bank = torch.randn(11, 2, self.height, self.width)
        one = expected_rms.diagnose_operator(
            self.codec, bank, candidate, self.reference_response,
            batch_size=1, injection_dtype="float16",
            condition_number_threshold=50.0, minimum_covariance_distance=0.0,
        )
        many = expected_rms.diagnose_operator(
            self.codec, bank, candidate, self.reference_response,
            batch_size=5, injection_dtype="float16",
            condition_number_threshold=50.0, minimum_covariance_distance=0.0,
        )
        self.assertAlmostEqual(one.paired_relative_l2_final_fp32, many.paired_relative_l2_final_fp32, places=10)
        self.assertAlmostEqual(one.measured_rms, many.measured_rms, places=10)
        torch.testing.assert_close(one.measured_radial_power, many.measured_radial_power, atol=1e-9, rtol=1e-9)
        self.assertTrue(math.isfinite(one.paired_relative_l2_injection_dtype))
        for field in dataclasses.fields(calibration_v2.CoefficientEnergyDiagnostics):
            if field.name == "measurement_domain":
                continue
            self.assertAlmostEqual(
                getattr(one.coefficient_projection, field.name),
                getattr(many.coefficient_projection, field.name),
                places=10,
            )

        reference = psd_editor.apply_psd_edit_tau_zero(bank)
        final = expected_rms.apply_frozen_operator(self.codec, bank, candidate.operator)
        reference_cast = reference.to(torch.float16).to(torch.float64)
        final_cast = final.to(torch.float16).to(torch.float64)
        direct = float(
            ((final_cast - reference_cast).square().sum() / reference_cast.square().sum()).sqrt()
        )
        self.assertAlmostEqual(one.paired_relative_l2_injection_dtype, direct, places=12)

    def test_coefficient_projection_is_batched_and_matches_direct_reference(self):
        candidate = self._candidate(0.25)
        torch.manual_seed(21)
        bank = torch.randn(11, 2, self.height, self.width)
        reference = psd_editor.apply_psd_edit_tau_zero(bank)
        final = expected_rms.apply_frozen_operator(self.codec, bank, candidate.operator)
        direct = calibration_v2.coefficient_energy_diagnostics(
            self.codec, reference, final, candidate.operator.group_indices
        )
        with mock.patch.object(self.codec, "encode", wraps=self.codec.encode) as encode:
            batched = expected_rms.diagnose_operator(
                self.codec, bank, candidate, self.reference_response,
                batch_size=3, injection_dtype="float16",
                condition_number_threshold=50.0, minimum_covariance_distance=0.0,
            )
        self.assertTrue(encode.call_args_list)
        self.assertLessEqual(
            max(call.args[0].shape[0] for call in encode.call_args_list),
            3,
        )
        for field in dataclasses.fields(calibration_v2.CoefficientEnergyDiagnostics):
            if field.name == "measurement_domain":
                continue
            self.assertAlmostEqual(
                getattr(batched.coefficient_projection, field.name),
                getattr(direct, field.name),
                places=10,
            )

    def test_frozen_payload_is_hash_bound_and_rejects_radial_correction(self):
        candidate = self._candidate()
        payload = expected_rms.operator_to_payload(candidate.operator)
        loaded = expected_rms.operator_from_payload(payload)
        self.assertEqual(expected_rms.operator_hash(loaded), payload["operator_hash"])
        stale = dict(payload)
        stale["scale"] *= 1.01
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            expected_rms.operator_from_payload(stale)
        corrected = dict(payload)
        corrected["correction"] = [1.0] * self.num_bins
        corrected["operator_hash"] = expected_rms.operator_hash(candidate.operator)
        with self.assertRaisesRegex(ValueError, "radial correction"):
            expected_rms.operator_from_payload(corrected)

    def test_pair_integrity_rejects_missing_duplicate_and_bad_control_links(self):
        candidate = self._candidate()
        control = expected_rms.construct_fourier_control(candidate, self.reference_response)
        pairs = expected_rms.validate_operator_pairs((candidate.operator, control.operator))
        self.assertEqual(len(pairs), 1)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            expected_rms.validate_operator_pairs((candidate.operator,))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            expected_rms.validate_operator_pairs((candidate.operator, candidate.operator, control.operator))
        broken = dataclasses.replace(control.operator, candidate_id="missing")
        with self.assertRaisesRegex(ValueError, "missing candidate"):
            expected_rms.validate_operator_pairs((candidate.operator, broken))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.codec = OverlapCodec(torch.eye(2), 1, 2)
        torch.manual_seed(13)
        self.bank = torch.randn(12, 2, 6, 7)

    def test_refines_between_zero_and_first_declared_tau(self):
        result = expected_rms.calibrate(
            self.codec, self.bank, group_id="G", group_indices=(0,),
            tau_plus_candidates=(0.25,), tau_minus_candidates=(-0.25,), targets=(0.01,),
            gate_r_s=4.0, gate_beta=2.0, num_bins=4, batch_size=4,
            injection_dtype="float16", condition_number_threshold=50.0,
            minimum_covariance_distance=0.0,
        )
        self.assertTrue(any(0 < abs(tau) < 0.25 for tau in result.attempted_taus["plus"]))
        self.assertTrue(any(selection.status == "SELECTED" for selection in result.selections))

    def test_unreachable_target_reports_nearest_without_generating_condition(self):
        result = expected_rms.calibrate(
            self.codec, self.bank, group_id="G", group_indices=(0,),
            tau_plus_candidates=(0.25,), tau_minus_candidates=(-0.25,), targets=(5.0,),
            gate_r_s=4.0, gate_beta=2.0, num_bins=4, batch_size=4,
            injection_dtype="float16", condition_number_threshold=50.0,
            minimum_covariance_distance=0.0,
        )
        self.assertTrue(all(selection.status == "TARGET_UNREACHABLE" for selection in result.selections))
        self.assertTrue(all(selection.nearest_feasible_tau is not None for selection in result.selections))
        self.assertEqual(result.operators, ())

    def test_one_failed_control_excludes_only_its_pair(self):
        original = expected_rms.diagnose_operator

        def reject_positive_control(*args, **kwargs):
            diagnostic = original(*args, **kwargs)
            construction = args[2]
            if construction.operator.operator_type == "fourier_control" and construction.operator.tau > 0:
                return dataclasses.replace(
                    diagnostic,
                    status="rejected",
                    failure_reason="synthetic_control_failure",
                )
            return diagnostic

        with mock.patch("pc_specific_psd.expected_rms.diagnose_operator", side_effect=reject_positive_control):
            result = expected_rms.calibrate(
                self.codec, self.bank, group_id="G", group_indices=(0,),
                tau_plus_candidates=(0.25,), tau_minus_candidates=(-0.25,), targets=(0.05,),
                gate_r_s=4.0, gate_beta=2.0, num_bins=4, batch_size=4,
                injection_dtype="float16", condition_number_threshold=50.0,
                minimum_covariance_distance=0.0, configured_tolerance=10.0,
            )
        self.assertEqual(result.status, "SELECTED")
        pairs = expected_rms.validate_operator_pairs(result.operators)
        self.assertEqual(len(pairs), 1)
        self.assertLess(pairs[0].candidate.tau, 0)
        excluded_ids = {item.condition_id for item in result.exclusions}
        self.assertTrue(any("_plus_" in condition_id for condition_id in excluded_ids))
        self.assertTrue(any("fourier_control" in condition_id for condition_id in excluded_ids))


if __name__ == "__main__":
    unittest.main()
