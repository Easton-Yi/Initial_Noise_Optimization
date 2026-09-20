"""Plan-mandated regression coverage for the tau=0 special case in
calibration.py: c_{B,0}(b) is exactly 1.0 for every bin by definition (never
a measured output of evaluate_candidate/select_gate_and_taus), the forced
full-codec path at tau=0 agrees with the scalar-reference shortcut via
numerical tolerance (not bitwise equality), and applying the all-ones
correction on top of that is confirmed to be a true no-op, independently of
the codec-agreement check.
"""
import unittest

import torch

from pc_specific_psd import calibration, psd_editor
from pc_specific_psd.patch_codec import OverlapCodec

CHANNELS, PATCH_SIZE = 2, 3
D = CHANNELS * PATCH_SIZE * PATCH_SIZE
NUM_BINS = 8
GROUP_INDICES = (0, 1)


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


class ZeroTauCorrectionIsExactlyOneTests(unittest.TestCase):
    def test_exactly_one_for_every_bin_regardless_of_num_bins(self):
        for num_bins in (1, 4, 8, 16):
            correction = calibration.zero_tau_correction(num_bins)
            self.assertTrue(torch.equal(correction, torch.ones(num_bins, dtype=torch.float64)))

    def test_is_a_definition_not_derived_from_any_measurement(self):
        """zero_tau_correction takes only num_bins -- structurally, it cannot
        read a calibration bank, codec, or gate, so it cannot possibly be a
        measured quantity subject to calibration-bank sampling noise.
        """
        import inspect
        parameters = list(inspect.signature(calibration.zero_tau_correction).parameters)
        self.assertEqual(parameters, ["num_bins"])


class ApplyingTheAllOnesCorrectionIsATrueNoopTests(unittest.TestCase):
    def test_apply_radial_correction_with_zero_tau_correction_reproduces_the_input(self):
        tensor = _bank()
        correction = calibration.zero_tau_correction(NUM_BINS)
        corrected = calibration.apply_radial_correction(tensor, correction, NUM_BINS)
        self.assertTrue(torch.allclose(corrected, tensor, atol=1e-5))

    def test_noop_holds_for_the_actual_tau_zero_editor_output_too(self):
        codec = _codec()
        bank = _bank()
        edited = psd_editor.apply_psd_edit(codec, bank, GROUP_INDICES, 0.0, r_s=0.5, beta=6.0)
        correction = calibration.zero_tau_correction(NUM_BINS)
        corrected = calibration.apply_radial_correction(edited, correction, NUM_BINS)
        self.assertTrue(torch.allclose(corrected, edited, atol=1e-5))


class ForcedFullCodecPathAgreesWithShortcutTests(unittest.TestCase):
    """Cross-references psd_editor's own tau=0 equivalence check (already
    covered directly in test_psd_editor.py) specifically in combination with
    calibration's correction step: applying the (all-ones) tau=0 correction
    must not disturb that agreement.
    """

    def test_general_codec_path_at_tau_zero_matches_shortcut_via_allclose_not_bitwise(self):
        codec = _codec()
        bank = _bank()
        general_path = psd_editor.apply_psd_edit(codec, bank, GROUP_INDICES, 0.0, r_s=0.5, beta=6.0)
        shortcut = psd_editor.apply_psd_edit_tau_zero(bank)
        self.assertTrue(torch.allclose(general_path, shortcut, atol=1e-4, rtol=1e-4))
        self.assertFalse(torch.equal(general_path, shortcut))  # numerically close, not bit-identical

    def test_agreement_survives_applying_the_frozen_zero_tau_correction(self):
        codec = _codec()
        bank = _bank()
        general_path = psd_editor.apply_psd_edit(codec, bank, GROUP_INDICES, 0.0, r_s=0.5, beta=6.0)
        shortcut = psd_editor.apply_psd_edit_tau_zero(bank)

        correction = calibration.zero_tau_correction(NUM_BINS)
        corrected_general_path = calibration.apply_radial_correction(general_path, correction, NUM_BINS)
        self.assertTrue(torch.allclose(corrected_general_path, shortcut, atol=1e-4, rtol=1e-4))


class EvaluateCandidateAtTauZeroYieldsTheDefinitionalCorrectionTests(unittest.TestCase):
    def test_evaluate_candidate_correction_matches_zero_tau_correction_under_both_protocols(self):
        codec = _codec()
        bank = _bank()
        gate = calibration.GateCandidate(r_s=0.5, beta=6.0)
        for protocol in ("operator_clean", "legacy_matched"):
            reference_power = calibration.compute_reference_power(bank, NUM_BINS, protocol=protocol)
            evaluation = calibration.evaluate_candidate(
                codec, bank, GROUP_INDICES, gate, 0.0, reference_power,
                num_bins=NUM_BINS, protocol=protocol, psd_tolerance=0.05, correction_gain_bound=1e9,
            )
            self.assertTrue(evaluation.accepted)
            self.assertTrue(torch.allclose(evaluation.correction, calibration.zero_tau_correction(NUM_BINS), atol=1e-3))


if __name__ == "__main__":
    unittest.main()
