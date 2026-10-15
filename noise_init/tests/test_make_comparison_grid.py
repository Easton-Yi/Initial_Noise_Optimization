import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from io_utils import condition_id
from make_comparison_grid import build_grid, resolve_seed_gammas


class ComparisonGridTests(unittest.TestCase):
    def test_legacy_gamma_maps_disjoint_method_grids_by_position(self):
        self.assertEqual(
            resolve_seed_gammas(0.05, None, None, [0.0125, 0.025, 0.05], [0.5, 0.75, 0.9]),
            (0.05, 0.9),
        )

    def test_one_seed_run_uses_method_specific_gamma_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            block = {
                "batch_seed": 4100000,
                "block_id": "geneval_0000_s000",
                "prompt": "a photo of a bench",
                "prompt_id": "geneval_0000",
                "seed_batch_id": "s000",
            }
            (run_dir / "blocks.jsonl").write_text(json.dumps(block) + "\n", encoding="utf-8")

            model_id = "sdxl_turbo"
            base = run_dir / "generations" / model_id / block["block_id"]
            baseline_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            same_phase_gammas = [0.025, 0.038, 0.05]
            independent_gammas = [0.75, 0.825, 0.9]
            conditions = [
                ("baseline", alpha, None) for alpha in baseline_alphas
            ] + [
                ("same_phase", 0.9, gamma) for gamma in same_phase_gammas
            ] + [
                ("independent_white", 0.9, gamma) for gamma in independent_gammas
            ]
            for family, alpha, gamma in conditions:
                target = base / condition_id(family, alpha, gamma) / "grid_1x4.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (40, 10), "white").save(target)

            grid = build_grid(
                run_dir=run_dir,
                model_id=model_id,
                prompt_id="geneval_0000",
                alpha=0.9,
                gamma=None,
                same_phase_gamma=None,
                independent_gamma=None,
                same_phase_gammas=same_phase_gammas,
                independent_gammas=independent_gammas,
                baseline_alphas=baseline_alphas,
                baseline_seed_batch=None,
                label_width=80,
                row_gap=2,
                column_gap=10,
            )

            self.assertGreater(grid.width, 0)
            self.assertGreater(grid.height, 0)


if __name__ == "__main__":
    unittest.main()
