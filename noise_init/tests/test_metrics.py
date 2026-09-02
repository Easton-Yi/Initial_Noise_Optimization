import unittest

import numpy as np

from analysis import _available_gammas, _bootstrap_matched_improvements, _curve_id, _two_panel_rows, interpolate_within, pareto_frontier
from metric_runner import _group_quality_rows


class MetricContractTests(unittest.TestCase):
    def test_four_images_group_to_one_quality_record(self):
        records = [{"block_id": "b", "condition_id": "c", "metric": "clip_cosine", "score": n, "metric_config_hash": "x", "method": "white"} for n in range(4)]
        grouped = _group_quality_rows(records)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["n"], 4)
        self.assertEqual(grouped[0]["score"], 1.5)

    def test_vendi_identity_kernel_is_one(self):
        # The effective-number definition equals exp(entropy(eigenvalues)).
        eigenvalues = np.linalg.eigvalsh(np.eye(4)) / 4
        self.assertAlmostEqual(float(np.exp(-(eigenvalues * np.log(eigenvalues)).sum())), 4.0)

    def test_frontier_and_interpolation_prohibit_extrapolation(self):
        points = [{"diversity": .1, "quality": .9}, {"diversity": .2, "quality": .8}, {"diversity": .3, "quality": .95}]
        front = pareto_frontier(points)
        self.assertEqual(len(front), 2)
        self.assertIsNone(interpolate_within(front, .05))
        self.assertIsNone(interpolate_within(front, .4))
        self.assertIsNotNone(interpolate_within(front, .2))

    def test_each_proposed_gamma_is_a_separate_curve(self):
        self.assertEqual(_curve_id({"family": "baseline", "gamma": ""}), "baseline")
        self.assertEqual(_curve_id({"family": "same_phase", "gamma": "0.1"}), "same_phase_gamma_0.1")
        self.assertEqual(_curve_id({"family": "independent_white", "gamma": .9}), "independent_white_gamma_0.9")
        self.assertEqual(_curve_id({"family": "same_phase", "gamma": .0125}), "same_phase_gamma_0.0125")
        self.assertEqual(_curve_id({"family": "same_phase", "gamma": .025}), "same_phase_gamma_0.025")
        self.assertNotEqual(_curve_id({"family": "same_phase", "gamma": .0125}), _curve_id({"family": "same_phase", "gamma": .025}))

    def test_two_panel_keeps_the_complete_baseline_curve(self):
        rows = [
            {"family": "baseline", "alpha": alpha, "gamma": ""}
            for alpha in (.0, .1, .2, .3, .4, .5, .6, .7)
        ] + [
            {"family": "same_phase", "alpha": alpha, "gamma": gamma}
            for alpha in (.5, .6, .7)
            for gamma in (.0125, .1)
        ]
        selected = _two_panel_rows(rows, {"alpha_values": (.6, .7), "gamma_values": (.0125, .1)})
        self.assertEqual([row["alpha"] for row in selected if row["family"] == "baseline"], [.0, .1, .2, .3, .4, .5, .6, .7])
        self.assertTrue(all(row["alpha"] in (.6, .7) for row in selected if row["family"] != "baseline"))

    def test_gamma_axis_uses_only_actual_available_grid_values(self):
        rows = [
            {"family": "same_phase", "gamma": ".0125", "available": True},
            {"family": "same_phase", "gamma": ".025", "available": True},
            {"family": "independent_white", "gamma": ".05", "available": True},
            {"family": "same_phase", "gamma": ".1", "available": False},
            {"family": "baseline", "gamma": "", "available": True},
        ]
        self.assertEqual(_available_gammas(rows), (.0125, .025, .05))

    def test_matched_diversity_bootstrap_includes_a_per_gamma_summary(self):
        values = {}
        for block, offset in (("b0", 0.0), ("b1", .01)):
            values[block, "baseline/alpha_0p0"] = {"q": .8 + offset, "d": .2}
            values[block, "baseline/alpha_0p7"] = {"q": .6 + offset, "d": .6}
            values[block, "same_phase/alpha_0p0_gamma_0p5"] = {"q": .85 + offset, "d": .2}
            values[block, "same_phase/alpha_0p7_gamma_0p5"] = {"q": .7 + offset, "d": .6}
        points, summary = _bootstrap_matched_improvements(values, ["baseline/alpha_0p0", "baseline/alpha_0p7"], ["same_phase/alpha_0p0_gamma_0p5", "same_phase/alpha_0p7_gamma_0p5"], "q", "d", [.2, .4, .6], {"bootstrap_replicates": 10, "bootstrap_seed": 1, "confidence_level": .95})
        self.assertEqual(len(points), 3)
        self.assertEqual(summary["bootstrap_replicates"], 10)
        self.assertGreater(summary["bootstrap_ci_high"], 0)


if __name__ == "__main__": unittest.main()
