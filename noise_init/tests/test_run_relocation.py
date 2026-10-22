"""Run relocation preserves source records and strict image-hash validation."""
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from io_utils import file_hash
from result_integrity import load_generation_records, resolve_recorded_image, validate_analysis_inputs
from test_formal_analysis import fixture


class RelocationTests(unittest.TestCase):
    def test_moved_run_validates_without_rewriting_sources_and_rejects_changed_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, _ = fixture(root, single=True)
            originals = {str(p.relative_to(run)): file_hash(p) for p in run.rglob('*') if p.is_file()}
            moved = root / 'moved_run'
            run.rename(moved)
            validate_analysis_inputs(moved)
            self.assertEqual(originals, {str(p.relative_to(moved)): file_hash(p) for p in moved.rglob('*') if p.is_file()})
            rows = load_generation_records(moved)
            self.assertTrue(all(Path(r['image_path']).is_relative_to(moved) for r in rows))
            self.assertTrue(all('recorded_image_path' in r for r in rows))
            Image.new('RGB', (8, 8), 'red').save(rows[0]['image_path'])
            with self.assertRaisesRegex(RuntimeError, 'invalid per_image'):
                validate_analysis_inputs(moved)

    def test_selected_run_missing_image_does_not_borrow_old_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, _ = fixture(root, single=True, same=False, independent=False)
            copied = root / 'copied_run'
            shutil.copytree(run, copied)
            validate_analysis_inputs(copied)
            next(copied.glob('generations/**/image_00.png')).unlink()
            with self.assertRaisesRegex(RuntimeError, 'missing PNG files'):
                validate_analysis_inputs(copied)

    def test_alias_uses_source_identity_and_never_basename_search(self):
        root = Path('/tmp/relocation-test-root')
        suffix = 'generations/model/block/baseline/alpha_0/image_00.png'
        row = {'model_id': 'model', 'block_id': 'block', 'condition_id': 'same_phase/alpha_0_gamma_0p1',
               'alias_of_condition_id': 'baseline/alpha_0', 'base_index': 0,
               'image_path': '/old/run/' + suffix}
        self.assertEqual(resolve_recorded_image(root, row), root / suffix)
        row['image_path'] = '/old/run/generations/other/block/baseline/alpha_0/image_00.png'
        self.assertEqual(resolve_recorded_image(root, row), Path(row['image_path']))
        row['image_path'] = suffix
        self.assertEqual(resolve_recorded_image(root, row), root / suffix)


if __name__ == '__main__': unittest.main()
