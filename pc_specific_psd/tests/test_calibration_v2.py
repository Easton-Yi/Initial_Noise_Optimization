import types
import unittest
from unittest.mock import patch

import torch

from pc_specific_psd import calibration, calibration_v2, cli, psd_editor
from pc_specific_psd.patch_codec import OverlapCodec


def _diagnostic(tau, delta, status="accepted", group_id="B1"):
    return types.SimpleNamespace(
        group_id=group_id,
        sign="plus" if tau > 0 else "minus" if tau < 0 else "zero",
        tau=float(tau), status=status, failure_reason="" if status == "accepted" else "unsafe",
        paired_relative_l2_final=float(delta), final_max_radial_psd_error=0.0,
        reference_rms=1.0, candidate_rms=1.0,
        covariance_distance=0.0,
        correction=torch.ones(2, dtype=torch.float64),
    )


class EffectSelectionTests(unittest.TestCase):
    def _run(self, values, *, targets=(0.05,), plus=(1.0, 2.0), minus=(-1.0, -2.0), tolerance=None):
        spec = calibration_v2.EffectGroupSpec("B1", (0,), plus, minus, targets, tolerance)

        def fake(*args, **kwargs):
            tau = float(args[5])
            return values[tau]

        with patch.object(calibration_v2, "fit_and_diagnose_candidate", side_effect=fake):
            return calibration_v2.select_effect_size_targets(
                object(), torch.zeros(1), torch.ones(1), calibration.GateCandidate(4.0, 2.0), [spec],
                protocol="operator_clean", num_bins=1, psd_tolerance=0.05,
                correction_gain_bound=4.0, condition_number_threshold=50.0,
            )

    def test_rms_independent_selection_ranks_by_final_delta(self):
        # Candidates may have indistinguishable RMS; only final effect enters this rule.
        values = {0.0: _diagnostic(0, 0), 1.0: _diagnostic(1, 0.049), 2.0: _diagnostic(2, 0.09),
                  -1.0: _diagnostic(-1, 0.049), -2.0: _diagnostic(-2, 0.09)}
        result = self._run(values)
        self.assertEqual([s.selected_tau for s in result.selections], [1.0, -1.0])

    def test_failed_tau_does_not_reject_other_candidates_or_other_sign(self):
        values = {0.0: _diagnostic(0, 0), 1.0: _diagnostic(1, 0.05, "rejected"),
                  2.0: _diagnostic(2, 0.052), -1.0: _diagnostic(-1, 0.049),
                  -2.0: _diagnostic(-2, 0.09)}
        result = self._run(values)
        self.assertEqual(result.status, "SELECTED")
        self.assertEqual(result.selections[0].selected_tau, 2.0)
        self.assertEqual(result.selections[1].selected_tau, -1.0)

    def test_unreachable_target_is_explicit(self):
        values = {0.0: _diagnostic(0, 0), 1.0: _diagnostic(1, 0.01), 2.0: _diagnostic(2, 0.02),
                  -1.0: _diagnostic(-1, 0.01), -2.0: _diagnostic(-2, 0.02)}
        result = self._run(values, targets=(0.10,))
        self.assertEqual(result.status, "FAIL")
        self.assertTrue(all(s.status == "TARGET_UNREACHABLE" for s in result.selections))
        self.assertTrue(all(s.feasible_delta_max == 0.02 for s in result.selections))

    def test_tie_break_prefers_smaller_absolute_tau(self):
        values = {0.0: _diagnostic(0, 0), 1.0: _diagnostic(1, 0.04), 2.0: _diagnostic(2, 0.06),
                  -1.0: _diagnostic(-1, 0.04), -2.0: _diagnostic(-2, 0.06)}
        result = self._run(values)
        self.assertEqual([s.selected_tau for s in result.selections], [1.0, -1.0])

    def test_partial_reachability_freezes_reachable_conditions(self):
        values = {0.0: _diagnostic(0, 0), 1.0: _diagnostic(1, 0.05), 2.0: _diagnostic(2, 0.03),
                  -1.0: _diagnostic(-1, 0.05), -2.0: _diagnostic(-2, 0.03)}
        result = self._run(values, targets=(0.05, 0.10))
        self.assertEqual(result.status, "SELECTED")
        self.assertFalse(result.all_targets_reached)
        self.assertEqual(result.unreachable_target_count, 2)
        self.assertEqual(sum(item.status == "SELECTED" for item in result.selections), 2)
        frozen = cli._effect_condition_selections(result)
        self.assertEqual(set(frozen), {"B1_plus_tau_1", "B1_minus_tau_1"})
        self.assertTrue(all(raw["target_relative_l2"] == [0.05] for raw in frozen.values()))

    def test_tau_zero_must_be_accepted_with_negligible_covariance_distance(self):
        values = {0.0: _diagnostic(0, 0), 1.0: _diagnostic(1, 0.05), 2.0: _diagnostic(2, 0.09),
                  -1.0: _diagnostic(-1, 0.05), -2.0: _diagnostic(-2, 0.09)}
        values[0.0].status = "rejected"
        result = self._run(values)
        self.assertEqual(result.status, "FAIL")
        values[0.0].status = "accepted"
        values[0.0].covariance_distance = 1e-3
        result = self._run(values)
        self.assertEqual(result.status, "FAIL")


class FinalOperatorTests(unittest.TestCase):
    def test_correction_statistics_use_reference_valid_mask_when_final_bin_collapses(self):
        codec = OverlapCodec(torch.eye(1), patch_size=1, channels=1)
        bank = torch.randn(2, 1, 4, 4)
        transfer = types.SimpleNamespace(worst_condition_number=1.0)
        with patch.object(calibration, "radial_power", side_effect=[
            torch.tensor([1.0, 1.0], dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
        ]), patch("pc_specific_psd.calibration_v2.spectral.linear_operator_covariance_distance", return_value=0.2), patch(
            "pc_specific_psd.calibration_v2.spectral.impulse_response_transfer_matrix", return_value=transfer
        ):
            diagnostic = calibration_v2.diagnose_fixed_operator(
                codec, bank, "B1", (0,), calibration.GateCandidate(2.0, 2.0), 0.5,
                torch.tensor([2.0, 0.1], dtype=torch.float64), num_bins=2,
                psd_tolerance=0.05, correction_gain_bound=20.0,
                condition_number_threshold=50.0,
            )
        self.assertEqual(diagnostic.failure_reason, "invalid_power_bin")
        self.assertEqual(diagnostic.valid_bin_mask.tolist(), [True, True])
        self.assertEqual(diagnostic.final_power_valid_mask.tolist(), [True, False])
        self.assertAlmostEqual(diagnostic.min_correction, 0.1)
        self.assertAlmostEqual(diagnostic.max_correction, 2.0)
        self.assertAlmostEqual(diagnostic.symmetric_correction_factor, 10.0)
        self.assertAlmostEqual(diagnostic.final_max_radial_psd_error, 1.0)

    def test_tau_zero_reproduces_reference(self):
        bank = torch.randn(3, 1, 8, 8)
        torch.testing.assert_close(
            psd_editor.apply_psd_edit_tau_zero(bank),
            psd_editor.apply_psd_edit_tau_zero(bank.clone()),
        )
        self.assertEqual(calibration_v2.paired_relative_l2(
            psd_editor.apply_psd_edit_tau_zero(bank), psd_editor.apply_psd_edit_tau_zero(bank)
        ), 0.0)

    def test_fixed_correction_formal_operator_is_linear(self):
        components = torch.eye(4)
        codec = OverlapCodec(components, patch_size=2, channels=1)
        correction = torch.linspace(0.9, 1.1, 4, dtype=torch.float64)

        def operator(value):
            edited = psd_editor.apply_psd_edit(codec, value, (0,), 0.5, 2.0, 2.0)
            return calibration.apply_radial_correction(edited, correction, 4)

        x, y = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
        torch.testing.assert_close(operator(x + y), operator(x) + operator(y), atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
