"""generate_preview/generate_full_pilot branch on select_candidates()'s actual
output length as real conditionals (plan §7.1): 0 candidates -> no manifest
at all, 1 -> 3 configs (36 preview / 144 full-pilot images), 2 -> 5 configs
(60 preview / 240 full-pilot images) -- never a hardcoded count.
"""
import tempfile
import unittest
from pathlib import Path

from pc_specific_psd import manifests, runner
from pc_specific_psd.tests._runner_test_support import FakeAdapterPCA, freeze_selected_calibration, load_test_config


def _count_images(run_dir: Path) -> int:
    return len(list(run_dir.rglob("image.png")))


class ZeroCandidateBranchTests(unittest.TestCase):
    def test_generate_preview_returns_none_for_zero_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            self.assertIsNone(runner.generate_preview(loaded, [], allow_synthetic_basis=True))

    def test_generate_full_pilot_returns_none_for_zero_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            self.assertIsNone(runner.generate_full_pilot(loaded, [], allow_synthetic_basis=True))


class OneCandidateBranchTests(unittest.TestCase):
    def test_preview_branch_produces_36_images_across_3_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])
            adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
            run_dir = runner.generate_manifest(
                loaded, manifests.build_preview_manifest_entries(["B5"]),
                run_id="preview1", force=False, adapter=adapter, allow_synthetic_basis=True,
            )
            self.assertEqual(_count_images(run_dir), 36)
            self.assertEqual(adapter.call_count, 36)
            conditions = {p.name for p in (run_dir / "generations").iterdir()}
            self.assertEqual(conditions, {"reference", "B5_plus", "B5_minus"})

    def test_full_pilot_branch_produces_144_images_across_3_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])
            adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
            run_dir = runner.generate_manifest(
                loaded, manifests.build_full_pilot_manifest_entries(["B5"]),
                run_id="full1", force=False, adapter=adapter, allow_synthetic_basis=True,
            )
            self.assertEqual(_count_images(run_dir), 144)
            self.assertEqual(adapter.call_count, 144)


class TwoCandidateBranchTests(unittest.TestCase):
    def test_preview_branch_produces_60_images_across_5_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5", "B6"])
            adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
            run_dir = runner.generate_manifest(
                loaded, manifests.build_preview_manifest_entries(["B5", "B6"]),
                run_id="preview2", force=False, adapter=adapter, allow_synthetic_basis=True,
            )
            self.assertEqual(_count_images(run_dir), 60)
            self.assertEqual(adapter.call_count, 60)
            conditions = {p.name for p in (run_dir / "generations").iterdir()}
            self.assertEqual(conditions, {"reference", "B5_plus", "B5_minus", "B6_plus", "B6_minus"})

    def test_full_pilot_branch_produces_240_images_across_5_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5", "B6"])
            adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
            run_dir = runner.generate_manifest(
                loaded, manifests.build_full_pilot_manifest_entries(["B5", "B6"]),
                run_id="full2", force=False, adapter=adapter, allow_synthetic_basis=True,
            )
            self.assertEqual(_count_images(run_dir), 240)
            self.assertEqual(adapter.call_count, 240)


if __name__ == "__main__":
    unittest.main()
