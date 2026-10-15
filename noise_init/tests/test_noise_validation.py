import unittest
import sys
import types

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    # Grid-selection tests do not execute tensor code. Allow this focused test
    # module to run in lightweight environments without the generation stack.
    sys.modules["torch"] = types.ModuleType("torch")

from noise_validation import ConditionSpec, _selected_gamma_grids, full_condition_grid


class NoiseValidationGridTests(unittest.TestCase):
    def setUp(self):
        self.alphas = {"same_phase": [0.8, 0.9], "independent_white": [0.8, 0.9]}
        self.gammas = {
            "same_phase": [0.0125, 0.025, 0.05],
            "independent_white": [0.5, 0.75, 0.9],
        }

    def test_condition_grid_preserves_method_specific_gammas(self):
        conditions = full_condition_grid([0.0, 0.7], self.alphas, self.gammas)
        self.assertEqual(len(conditions), 2 + 2 * 3 + 2 * 3)
        self.assertIn(ConditionSpec("same_phase", 0.9, 0.05), conditions)
        self.assertIn(ConditionSpec("independent_white", 0.9, 0.9), conditions)
        self.assertNotIn(ConditionSpec("independent_white", 0.9, 0.05), conditions)
        self.assertNotIn(ConditionSpec("same_phase", 0.9, 0.9), conditions)

    def test_method_specific_gamma_selection(self):
        selected = _selected_gamma_grids(None, "0.05", "0.9", self.gammas)
        self.assertEqual(selected["same_phase"], (0.05,))
        self.assertEqual(selected["independent_white"], (0.9,))

    def test_shared_gamma_rejected_for_disjoint_grids(self):
        with self.assertRaisesRegex(ValueError, "gamma grids are disjoint"):
            _selected_gamma_grids("0.05", None, None, self.gammas)

    def test_shared_gamma_supported_for_common_grid(self):
        common_grids = {"same_phase": [0.1, 0.2], "independent_white": [0.1, 0.2]}
        selected = _selected_gamma_grids("0.2", None, None, common_grids)
        self.assertEqual(selected, {"same_phase": (0.2,), "independent_white": (0.2,)})

    def test_shared_and_method_specific_flags_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            _selected_gamma_grids("0.1", "0.05", None, self.gammas)


if __name__ == "__main__":
    unittest.main()
