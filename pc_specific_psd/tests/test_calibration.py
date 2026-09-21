import unittest
from unittest.mock import patch

import torch

from pc_specific_psd import calibration, psd_editor
from pc_specific_psd.compat_generation import normalize
from pc_specific_psd.patch_codec import OverlapCodec

CHANNELS, PATCH_SIZE = 2, 3
D = CHANNELS * PATCH_SIZE * PATCH_SIZE
NUM_BINS = 8


def _random_orthonormal(d: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn((d, d), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix)
    return q.to(torch.float32)


def _codec() -> OverlapCodec:
    return OverlapCodec(_random_orthonormal(D, seed=7), PATCH_SIZE, CHANNELS)


def _bank(seed: int = 42, batch: int = 10, height: int = 16, width: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, CHANNELS, height, width, generator=generator)


class ComputeReferencePowerTests(unittest.TestCase):
    def test_operator_clean_matches_manual_unnormalized_measurement(self):
        bank = _bank()
        manual = calibration.radial_power(psd_editor.apply_psd_edit_tau_zero(bank), NUM_BINS)
        actual = calibration.compute_reference_power(bank, NUM_BINS, protocol="operator_clean")
        self.assertTrue(torch.allclose(manual, actual))

    def test_legacy_matched_applies_normalize_and_differs_from_operator_clean(self):
        bank = _bank()
        manual = calibration.radial_power(
            normalize(psd_editor.apply_psd_edit_tau_zero(bank), calibration.LEGACY_NORMALIZATION_PROFILE), NUM_BINS
        )
        legacy = calibration.compute_reference_power(bank, NUM_BINS, protocol="legacy_matched")
        operator_clean = calibration.compute_reference_power(bank, NUM_BINS, protocol="operator_clean")
        self.assertTrue(torch.allclose(manual, legacy))
        self.assertFalse(torch.allclose(legacy, operator_clean))

    def test_operator_clean_never_calls_per_sample_normalize(self):
        bank = _bank()
        with patch.object(calibration, "normalize", side_effect=AssertionError("normalize called")):
            calibration.compute_reference_power(bank, NUM_BINS, protocol="operator_clean")


class ComputeRadialCorrectionTests(unittest.TestCase):
    def test_matching_powers_give_all_ones_correction(self):
        power = torch.tensor([4.0, 9.0, 16.0], dtype=torch.float64)
        result = calibration.compute_radial_correction(power, power, max_gain_bound=10.0)
        self.assertTrue(torch.allclose(result.correction, torch.ones(3, dtype=torch.float64)))
        self.assertFalse(result.exceeds_gain_bound)
        self.assertEqual(result.max_gain, 1.0)

    def test_correction_formula_is_sqrt_ratio(self):
        reference = torch.tensor([4.0], dtype=torch.float64)
        measured = torch.tensor([1.0], dtype=torch.float64)
        result = calibration.compute_radial_correction(reference, measured, max_gain_bound=10.0)
        self.assertAlmostEqual(float(result.correction[0]), 2.0, places=10)

    def test_near_zero_measured_power_forces_noop_not_huge_gain(self):
        reference = torch.tensor([4.0, 4.0], dtype=torch.float64)
        measured = torch.tensor([1.0, 1e-15], dtype=torch.float64)
        result = calibration.compute_radial_correction(reference, measured, max_gain_bound=10.0, min_power=1e-12)
        self.assertTrue(bool(result.invalid_bins[1]))
        self.assertEqual(float(result.correction[1]), 1.0)
        self.assertFalse(bool(result.invalid_bins[0]))

    def test_near_zero_reference_power_also_forces_noop(self):
        reference = torch.tensor([1e-15, 4.0], dtype=torch.float64)
        measured = torch.tensor([1.0, 1.0], dtype=torch.float64)
        result = calibration.compute_radial_correction(reference, measured, max_gain_bound=10.0, min_power=1e-12)
        self.assertTrue(bool(result.invalid_bins[0]))
        self.assertEqual(float(result.correction[0]), 1.0)

    def test_exceeds_gain_bound_flag(self):
        reference = torch.tensor([100.0], dtype=torch.float64)
        measured = torch.tensor([1.0], dtype=torch.float64)
        result = calibration.compute_radial_correction(reference, measured, max_gain_bound=5.0)
        self.assertTrue(result.exceeds_gain_bound)
        self.assertEqual(result.max_gain, 10.0)

    def test_mismatched_shapes_raise(self):
        with self.assertRaises(ValueError):
            calibration.compute_radial_correction(torch.ones(3, dtype=torch.float64), torch.ones(2, dtype=torch.float64), max_gain_bound=1.0)


class ApplyRadialCorrectionTests(unittest.TestCase):
    def test_all_ones_correction_is_a_noop(self):
        tensor = _bank()
        correction = torch.ones(NUM_BINS, dtype=torch.float64)
        corrected = calibration.apply_radial_correction(tensor, correction, NUM_BINS)
        self.assertTrue(torch.allclose(corrected, tensor, atol=1e-5))

    def test_correction_derived_from_compute_radial_correction_is_exact_in_one_shot(self):
        """A pure per-radial-bin scalar multiplier's effect on mean bin power
        is an exact algebraic identity when nothing nonlinear (like
        normalize()) intervenes -- this is why operator_clean converges in
        one correction pass. Verified directly on a plain tensor here,
        independent of the editor/codec machinery.
        """
        tensor = _bank()
        measured = calibration.radial_power(tensor, NUM_BINS)
        target = measured * torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5, 0.8, 2.5, 1.2], dtype=torch.float64)
        step = calibration.compute_radial_correction(target, measured, max_gain_bound=1e9)
        corrected = calibration.apply_radial_correction(tensor, step.correction, NUM_BINS)
        remeasured = calibration.radial_power(corrected, NUM_BINS)
        self.assertTrue(torch.allclose(remeasured, target, rtol=1e-4, atol=1e-8))


class PsdRelativeErrorTests(unittest.TestCase):
    def test_zero_error_when_equal(self):
        power = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        error = calibration.psd_relative_error(power, power)
        self.assertTrue(torch.allclose(error, torch.zeros(3, dtype=torch.float64)))

    def test_formula(self):
        reference = torch.tensor([2.0], dtype=torch.float64)
        measured = torch.tensor([3.0], dtype=torch.float64)
        error = calibration.psd_relative_error(reference, measured)
        self.assertAlmostEqual(float(error[0]), 0.5, places=10)


class EvaluateCandidateTests(unittest.TestCase):
    def setUp(self):
        self.codec = _codec()
        self.bank = _bank()
        self.group_indices = (0, 1)
        self.gate = calibration.GateCandidate(r_s=0.5, beta=6.0)

    def test_unknown_protocol_raises(self):
        reference_power = calibration.compute_reference_power(self.bank, NUM_BINS, protocol="operator_clean")
        with self.assertRaises(ValueError):
            calibration.evaluate_candidate(
                self.codec, self.bank, self.group_indices, self.gate, 0.5, reference_power,
                num_bins=NUM_BINS, protocol="bogus", psd_tolerance=0.05, correction_gain_bound=10.0,
            )

    def test_operator_clean_candidate_never_calls_per_sample_normalize(self):
        reference_power = calibration.compute_reference_power(self.bank, NUM_BINS, protocol="operator_clean")
        with patch.object(calibration, "normalize", side_effect=AssertionError("normalize called")):
            evaluation = calibration.evaluate_candidate(
                self.codec, self.bank, self.group_indices, self.gate, 0.5, reference_power,
                num_bins=NUM_BINS, protocol="operator_clean", psd_tolerance=1.0,
                correction_gain_bound=1e9,
            )
        self.assertTrue(evaluation.accepted)

    def test_operator_clean_converges_in_two_iterations_even_at_tight_tolerance(self):
        reference_power = calibration.compute_reference_power(self.bank, NUM_BINS, protocol="operator_clean")
        for tau in (3.0, -3.0, 1.5):
            evaluation = calibration.evaluate_candidate(
                self.codec, self.bank, self.group_indices, self.gate, tau, reference_power,
                num_bins=NUM_BINS, protocol="operator_clean", psd_tolerance=1e-6, correction_gain_bound=1e9,
            )
            self.assertTrue(evaluation.accepted, msg=f"tau={tau} reason={evaluation.reason}")
            self.assertEqual(evaluation.iterations_used, 2)

    def test_legacy_matched_can_fail_to_converge_within_iteration_cap(self):
        reference_power = calibration.compute_reference_power(self.bank, NUM_BINS, protocol="legacy_matched")
        evaluation = calibration.evaluate_candidate(
            self.codec, self.bank, self.group_indices, self.gate, 3.0, reference_power,
            num_bins=NUM_BINS, protocol="legacy_matched", psd_tolerance=1e-6, correction_gain_bound=1e9,
        )
        self.assertFalse(evaluation.accepted)
        self.assertEqual(evaluation.reason, "psd_tolerance_exceeded")
        self.assertEqual(evaluation.iterations_used, calibration.MAX_CALIBRATION_ITERATIONS)

    def test_legacy_matched_passes_at_loose_tolerance(self):
        reference_power = calibration.compute_reference_power(self.bank, NUM_BINS, protocol="legacy_matched")
        evaluation = calibration.evaluate_candidate(
            self.codec, self.bank, self.group_indices, self.gate, 0.6, reference_power,
            num_bins=NUM_BINS, protocol="legacy_matched", psd_tolerance=0.05, correction_gain_bound=1e9,
        )
        self.assertTrue(evaluation.accepted)

    def test_tau_zero_needs_no_nontrivial_correction_under_either_protocol(self):
        for protocol in ("operator_clean", "legacy_matched"):
            reference_power = calibration.compute_reference_power(self.bank, NUM_BINS, protocol=protocol)
            evaluation = calibration.evaluate_candidate(
                self.codec, self.bank, self.group_indices, self.gate, 0.0, reference_power,
                num_bins=NUM_BINS, protocol=protocol, psd_tolerance=0.05, correction_gain_bound=1e9,
            )
            self.assertTrue(evaluation.accepted)
            self.assertEqual(evaluation.iterations_used, 1)
            self.assertTrue(torch.allclose(evaluation.correction, torch.ones(NUM_BINS, dtype=torch.float64), atol=1e-3))

    def test_condition_number_threshold_gates_acceptance(self):
        reference_power = calibration.compute_reference_power(self.bank, NUM_BINS, protocol="operator_clean")
        unconstrained = calibration.evaluate_candidate(
            self.codec, self.bank, self.group_indices, self.gate, 0.6, reference_power,
            num_bins=NUM_BINS, protocol="operator_clean", psd_tolerance=0.05, correction_gain_bound=1e9,
        )
        measured_condition_number = unconstrained.worst_condition_number
        self.assertIsNone(measured_condition_number)  # not requested -> None

        tight = calibration.evaluate_candidate(
            self.codec, self.bank, self.group_indices, self.gate, 0.6, reference_power,
            num_bins=NUM_BINS, protocol="operator_clean", psd_tolerance=0.05, correction_gain_bound=1e9,
            condition_number_threshold=1.0001,
        )
        self.assertFalse(tight.accepted)
        self.assertEqual(tight.reason, "condition_number_exceeded")

        loose = calibration.evaluate_candidate(
            self.codec, self.bank, self.group_indices, self.gate, 0.6, reference_power,
            num_bins=NUM_BINS, protocol="operator_clean", psd_tolerance=0.05, correction_gain_bound=1e9,
            condition_number_threshold=1e9,
        )
        self.assertTrue(loose.accepted)
        self.assertAlmostEqual(tight.worst_condition_number, loose.worst_condition_number, places=6)


class ValidateOnBankTests(unittest.TestCase):
    def setUp(self):
        self.codec = _codec()
        self.calibration_bank = _bank(seed=42)
        self.reference_power = calibration.compute_reference_power(self.calibration_bank, NUM_BINS, protocol="operator_clean")
        group_spec = calibration.GroupCandidateSpec("B1", (0, 1), (0.6,), (-0.6,), target_rms=1.0)
        self.result = calibration.select_gate_and_taus(
            self.codec, self.calibration_bank, self.reference_power,
            [calibration.GateCandidate(r_s=0.5, beta=6.0)], [group_spec],
            protocol="operator_clean", num_bins=NUM_BINS, psd_tolerance=0.05, correction_gain_bound=1e9,
        )
        self.assertEqual(self.result.status, "SELECTED")

    def test_requires_selected_status(self):
        rejected = calibration.CalibrationResult(status="REJECTED_ALL_GATES", gate=None, protocol="operator_clean", group_selections={}, rejected_gates=())
        with self.assertRaises(ValueError):
            calibration.validate_on_bank(
                self.codec, self.calibration_bank, rejected, "B1", "plus", self.reference_power,
                num_bins=NUM_BINS, psd_tolerance=0.05,
            )

    def test_invalid_sign_raises(self):
        with self.assertRaises(ValueError):
            calibration.validate_on_bank(
                self.codec, self.calibration_bank, self.result, "B1", "sideways", self.reference_power,
                num_bins=NUM_BINS, psd_tolerance=0.05,
            )

    def test_deterministic_repeat_call_gives_identical_outcome(self):
        first = calibration.validate_on_bank(
            self.codec, self.calibration_bank, self.result, "B1", "plus", self.reference_power,
            num_bins=NUM_BINS, psd_tolerance=0.05,
        )
        second = calibration.validate_on_bank(
            self.codec, self.calibration_bank, self.result, "B1", "plus", self.reference_power,
            num_bins=NUM_BINS, psd_tolerance=0.05,
        )
        self.assertEqual(first.passed, second.passed)
        self.assertEqual(first.reason, second.reason)
        self.assertTrue(torch.allclose(first.measured_rel_error, second.measured_rel_error))

    def test_passes_on_the_calibration_bank_itself_at_loose_tolerance(self):
        outcome = calibration.validate_on_bank(
            self.codec, self.calibration_bank, self.result, "B1", "plus", self.reference_power,
            num_bins=NUM_BINS, psd_tolerance=0.05,
        )
        self.assertTrue(outcome.passed)


if __name__ == "__main__":
    unittest.main()
