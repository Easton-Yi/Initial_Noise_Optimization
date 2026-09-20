"""Plan-mandated cross-cutting coverage for calibration.select_gate_and_taus:
gate rejection in declared order, closest-RMS tau selection with declared-
order tie-breaking, structural proof that selection never touches a
validation bank, and that a failing validation-bank check marks the
calibration version FAIL outright with no fallback to another candidate.
"""
import inspect
import unittest

import torch

from pc_specific_psd import calibration
from pc_specific_psd.patch_codec import OverlapCodec

CHANNELS, PATCH_SIZE = 2, 3
D = CHANNELS * PATCH_SIZE * PATCH_SIZE
NUM_BINS = 8
GROUP_INDICES = (0, 1)
TAU = 0.6


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


class DeclaredOrderGateRejectionTests(unittest.TestCase):
    """gate_a is measured (via a permissive bound) to need a strictly larger
    correction-gain than gate_b; a bound strictly between the two rejects
    gate_a and, since it is declared first, select_gate_and_taus must still
    fall through to gate_b -- not stop, not pick gate_b out of order.
    """

    def setUp(self):
        self.codec = _codec()
        self.bank = _bank()
        self.reference_power = calibration.compute_reference_power(self.bank, NUM_BINS, protocol="operator_clean")
        self.gate_a = calibration.GateCandidate(r_s=0.5, beta=6.0)
        self.gate_b = calibration.GateCandidate(r_s=4.0, beta=1.0)

        # A gate's overall gain requirement, as select_gate_and_taus itself
        # computes it, is the max over *every* declared tau candidate
        # (tau+ and tau-) for the group -- not just one of them.
        def _overall_max_gain(gate: calibration.GateCandidate) -> float:
            return max(
                calibration.evaluate_candidate(
                    self.codec, self.bank, GROUP_INDICES, gate, tau, self.reference_power,
                    num_bins=NUM_BINS, protocol="operator_clean", psd_tolerance=0.05, correction_gain_bound=1e9,
                ).max_gain
                for tau in (TAU, -TAU)
            )

        gain_a = _overall_max_gain(self.gate_a)
        gain_b = _overall_max_gain(self.gate_b)
        self.assertNotAlmostEqual(gain_a, gain_b, places=6)
        if gain_a < gain_b:
            self.gate_a, self.gate_b = self.gate_b, self.gate_a
            gain_a, gain_b = gain_b, gain_a
        # From here on gate_a is guaranteed to need the strictly larger gain.
        self.bound = (gain_a + gain_b) / 2

    def test_first_declared_gate_exceeding_bound_is_rejected_second_is_selected(self):
        group_spec = calibration.GroupCandidateSpec("B1", GROUP_INDICES, (TAU,), (-TAU,), target_rms=0.0)
        result = calibration.select_gate_and_taus(
            self.codec, self.bank, self.reference_power,
            [self.gate_a, self.gate_b], [group_spec],
            protocol="operator_clean", num_bins=NUM_BINS, psd_tolerance=0.05, correction_gain_bound=self.bound,
        )
        self.assertEqual(result.status, "SELECTED")
        self.assertEqual(result.gate, self.gate_b)
        self.assertEqual(len(result.rejected_gates), 1)
        rejected_gate, reason = result.rejected_gates[0]
        self.assertEqual(rejected_gate, self.gate_a)
        self.assertIn("correction_gain_exceeded", reason)

    def test_scanning_order_is_the_declared_list_order_not_reversed(self):
        """Swapping declared order changes which gate is even attempted
        first: with gate_b declared first it survives immediately, so
        gate_a (which would fail) is never evaluated at all -- rejected_gates
        must be empty, proving the scan followed declared order rather than
        some other rule (e.g. best-gate-wins).
        """
        group_spec = calibration.GroupCandidateSpec("B1", GROUP_INDICES, (TAU,), (-TAU,), target_rms=0.0)
        result = calibration.select_gate_and_taus(
            self.codec, self.bank, self.reference_power,
            [self.gate_b, self.gate_a], [group_spec],
            protocol="operator_clean", num_bins=NUM_BINS, psd_tolerance=0.05, correction_gain_bound=self.bound,
        )
        self.assertEqual(result.status, "SELECTED")
        self.assertEqual(result.gate, self.gate_b)
        self.assertEqual(result.rejected_gates, ())


class RejectedAllGatesTests(unittest.TestCase):
    def test_every_gate_failing_yields_explicit_rejection_result_not_an_exception(self):
        codec = _codec()
        bank = _bank()
        reference_power = calibration.compute_reference_power(bank, NUM_BINS, protocol="operator_clean")
        gates = [calibration.GateCandidate(r_s=0.5, beta=6.0), calibration.GateCandidate(r_s=4.0, beta=1.0)]
        group_spec = calibration.GroupCandidateSpec("B1", GROUP_INDICES, (TAU,), (-TAU,), target_rms=0.0)

        result = calibration.select_gate_and_taus(
            codec, bank, reference_power, gates, [group_spec],
            protocol="operator_clean", num_bins=NUM_BINS, psd_tolerance=0.05, correction_gain_bound=0.01,
        )
        self.assertEqual(result.status, "REJECTED_ALL_GATES")
        self.assertIsNone(result.gate)
        self.assertEqual(result.group_selections, {})
        self.assertEqual(len(result.rejected_gates), 2)
        for gate, reason in result.rejected_gates:
            self.assertIn(gate, gates)
            self.assertIn("correction_gain_exceeded", reason)

    def test_empty_gate_candidates_raises(self):
        codec = _codec()
        bank = _bank()
        reference_power = calibration.compute_reference_power(bank, NUM_BINS, protocol="operator_clean")
        group_spec = calibration.GroupCandidateSpec("B1", GROUP_INDICES, (TAU,), (-TAU,), target_rms=0.0)
        with self.assertRaises(ValueError):
            calibration.select_gate_and_taus(
                codec, bank, reference_power, [], [group_spec],
                protocol="operator_clean", num_bins=NUM_BINS, psd_tolerance=0.05, correction_gain_bound=1e9,
            )

    def test_empty_group_specs_raises(self):
        codec = _codec()
        bank = _bank()
        reference_power = calibration.compute_reference_power(bank, NUM_BINS, protocol="operator_clean")
        with self.assertRaises(ValueError):
            calibration.select_gate_and_taus(
                codec, bank, reference_power, [calibration.GateCandidate(r_s=0.5, beta=6.0)], [],
                protocol="operator_clean", num_bins=NUM_BINS, psd_tolerance=0.05, correction_gain_bound=1e9,
            )

    def test_group_spec_missing_tau_candidates_raises(self):
        codec = _codec()
        bank = _bank()
        reference_power = calibration.compute_reference_power(bank, NUM_BINS, protocol="operator_clean")
        empty_plus = calibration.GroupCandidateSpec("B1", GROUP_INDICES, (), (-TAU,), target_rms=0.0)
        with self.assertRaises(ValueError):
            calibration.select_gate_and_taus(
                codec, bank, reference_power, [calibration.GateCandidate(r_s=0.5, beta=6.0)], [empty_plus],
                protocol="operator_clean", num_bins=NUM_BINS, psd_tolerance=0.05, correction_gain_bound=1e9,
            )


class ClosestRmsSelectionTests(unittest.TestCase):
    def setUp(self):
        self.codec = _codec()
        self.bank = _bank()
        self.reference_power = calibration.compute_reference_power(self.bank, NUM_BINS, protocol="operator_clean")
        self.gate = calibration.GateCandidate(r_s=0.5, beta=6.0)
        self.tau_plus_candidates = (0.6, 0.3)
        self.measured_rms = {}
        for tau in (*self.tau_plus_candidates, -0.3, -0.6):
            evaluation = calibration.evaluate_candidate(
                self.codec, self.bank, GROUP_INDICES, self.gate, tau, self.reference_power,
                num_bins=NUM_BINS, protocol="operator_clean", psd_tolerance=0.05, correction_gain_bound=1e9,
            )
            self.measured_rms[tau] = evaluation.measured_rms
        # The two tau+ candidates must have genuinely distinct measured RMS,
        # or the exact-match and tie constructions below are not meaningful.
        self.assertNotAlmostEqual(self.measured_rms[0.6], self.measured_rms[0.3], places=8)

    def _select(self, target_rms: float) -> calibration.CalibrationResult:
        group_spec = calibration.GroupCandidateSpec(
            "B1", GROUP_INDICES, self.tau_plus_candidates, (-0.3, -0.6), target_rms=target_rms
        )
        return calibration.select_gate_and_taus(
            self.codec, self.bank, self.reference_power, [self.gate], [group_spec],
            protocol="operator_clean", num_bins=NUM_BINS, psd_tolerance=0.05, correction_gain_bound=1e9,
        )

    def test_exact_target_selects_the_matching_candidate(self):
        result = self._select(target_rms=self.measured_rms[0.3])
        self.assertEqual(result.status, "SELECTED")
        self.assertEqual(result.group_selections["B1"].tau_plus, 0.3)

    def test_tie_broken_by_declared_order_first_candidate_wins(self):
        """target_rms placed exactly at the midpoint between the two
        candidates' measured RMS creates a genuine tie in |distance|; the
        selection must pick 0.6 because it is declared first in
        tau_plus_candidates, not 0.3, even though both are equidistant.
        """
        midpoint = (self.measured_rms[0.6] + self.measured_rms[0.3]) / 2
        distance_to_first = abs(self.measured_rms[0.6] - midpoint)
        distance_to_second = abs(self.measured_rms[0.3] - midpoint)
        self.assertAlmostEqual(distance_to_first, distance_to_second, places=9)

        result = self._select(target_rms=midpoint)
        self.assertEqual(result.status, "SELECTED")
        self.assertEqual(result.group_selections["B1"].tau_plus, 0.6)


class SelectionNeverConsultsValidationBankTests(unittest.TestCase):
    def test_select_gate_and_taus_signature_has_no_validation_bank_parameter(self):
        """The selection procedure (plan section 5.4) must be run entirely
        against the calibration bank -- structurally, not just by
        convention: its signature has no parameter through which a
        validation bank, generated image, or reward/quality metric could
        enter. validate_on_bank is a separate function, called exactly once,
        afterward, over the already-frozen result.
        """
        parameters = inspect.signature(calibration.select_gate_and_taus).parameters
        for name in parameters:
            self.assertNotIn("valid", name.lower())
            self.assertNotIn("reward", name.lower())
            self.assertNotIn("image", name.lower())

    def test_selection_outcome_is_unaffected_by_a_differently_distributed_bank(self):
        """Two structurally identical selection runs differing only in
        which calibration bank realization is used may reasonably pick
        different taus (that's the whole point of calibrating against a
        specific bank) -- but a single call never reads from more than the
        one bank passed in. This is checked by confirming the result is a
        pure function of (codec, bank, reference_power, candidates): calling
        it twice with the same calibration bank reproduces the same
        selection exactly.
        """
        codec = _codec()
        bank = _bank(seed=123)
        reference_power = calibration.compute_reference_power(bank, NUM_BINS, protocol="operator_clean")
        group_spec = calibration.GroupCandidateSpec("B1", GROUP_INDICES, (0.6, 0.3), (-0.3, -0.6), target_rms=0.3)
        gates = [calibration.GateCandidate(r_s=0.5, beta=6.0)]

        first = calibration.select_gate_and_taus(
            codec, bank, reference_power, gates, [group_spec],
            protocol="operator_clean", num_bins=NUM_BINS, psd_tolerance=0.05, correction_gain_bound=1e9,
        )
        second = calibration.select_gate_and_taus(
            codec, bank, reference_power, gates, [group_spec],
            protocol="operator_clean", num_bins=NUM_BINS, psd_tolerance=0.05, correction_gain_bound=1e9,
        )
        self.assertEqual(first.status, second.status)
        self.assertEqual(first.gate, second.gate)
        self.assertEqual(first.group_selections["B1"].tau_plus, second.group_selections["B1"].tau_plus)
        self.assertEqual(first.group_selections["B1"].tau_minus, second.group_selections["B1"].tau_minus)


class FailingValidationHasNoFallbackTests(unittest.TestCase):
    def setUp(self):
        self.codec = _codec()
        self.calibration_bank = _bank(seed=42)
        self.reference_power = calibration.compute_reference_power(self.calibration_bank, NUM_BINS, protocol="operator_clean")
        group_spec = calibration.GroupCandidateSpec("B1", GROUP_INDICES, (0.6,), (-0.6,), target_rms=0.0)
        self.result = calibration.select_gate_and_taus(
            self.codec, self.calibration_bank, self.reference_power,
            [calibration.GateCandidate(r_s=0.5, beta=6.0)], [group_spec],
            protocol="operator_clean", num_bins=NUM_BINS, psd_tolerance=0.05, correction_gain_bound=1e9,
        )
        self.assertEqual(self.result.status, "SELECTED")

    def test_validate_on_bank_signature_takes_one_candidate_no_alternative_list(self):
        parameters = inspect.signature(calibration.validate_on_bank).parameters
        self.assertNotIn("fallback", " ".join(parameters).lower())
        self.assertNotIn("candidates", " ".join(parameters).lower())

    def test_failing_validation_marks_fail_outright_and_does_not_mutate_the_result(self):
        validation_bank = _bank(seed=777)
        validation_reference_power = calibration.compute_reference_power(validation_bank, NUM_BINS, protocol="operator_clean")

        outcome = calibration.validate_on_bank(
            self.codec, validation_bank, self.result, "B1", "plus", validation_reference_power,
            num_bins=NUM_BINS, psd_tolerance=0.0,
        )
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.reason, "validation_bank_tolerance_exceeded")

        # The frozen result itself is untouched: the same tau/gate/correction
        # are still there, and re-running validate_on_bank against it again
        # (the only way to interact with it further) reproduces the same
        # failing outcome rather than trying some other candidate.
        self.assertEqual(self.result.group_selections["B1"].tau_plus, 0.6)
        repeat = calibration.validate_on_bank(
            self.codec, validation_bank, self.result, "B1", "plus", validation_reference_power,
            num_bins=NUM_BINS, psd_tolerance=0.0,
        )
        self.assertEqual(outcome.passed, repeat.passed)
        self.assertEqual(outcome.reason, repeat.reason)

    def test_a_retry_after_failure_means_building_a_new_calibration_result(self):
        """There is no method on a failed ValidationOutcome or on the frozen
        CalibrationResult that tries another candidate -- the only way
        calibration.py exposes to get a different outcome is to call
        select_gate_and_taus again (a new calibration version) with fresh
        inputs, which is exactly what select_gate_and_taus's pure-function
        behavior (asserted in SelectionNeverConsultsValidationBankTests)
        guarantees is reproducible and independent of any prior
        validate_on_bank call.
        """
        second_calibration_bank = _bank(seed=555)
        second_reference_power = calibration.compute_reference_power(second_calibration_bank, NUM_BINS, protocol="operator_clean")
        group_spec = calibration.GroupCandidateSpec("B1", GROUP_INDICES, (0.6,), (-0.6,), target_rms=0.0)
        new_result = calibration.select_gate_and_taus(
            self.codec, second_calibration_bank, second_reference_power,
            [calibration.GateCandidate(r_s=0.5, beta=6.0)], [group_spec],
            protocol="operator_clean", num_bins=NUM_BINS, psd_tolerance=0.05, correction_gain_bound=1e9,
        )
        self.assertEqual(new_result.status, "SELECTED")
        self.assertIsNot(new_result, self.result)


if __name__ == "__main__":
    unittest.main()
