import unittest

import numpy as np

from analysis import interpolate_within, pareto_frontier
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


if __name__ == "__main__": unittest.main()
