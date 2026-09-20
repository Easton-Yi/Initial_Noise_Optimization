"""Confirms CLI path resolution is anchored to the config file's own
directory (``config.resolve_root``), never the process's current working
directory -- required since real usage invokes this CLI from arbitrary cwds.
"""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from pc_specific_psd import cli
from pc_specific_psd.tests._runner_test_support import load_test_config


class CwdIndependenceTests(unittest.TestCase):
    def test_inspect_basis_dry_run_resolves_relative_paths_against_config_dir(self):
        with tempfile.TemporaryDirectory() as config_tmp, tempfile.TemporaryDirectory() as unrelated_tmp:
            loaded = load_test_config(Path(config_tmp))
            config_path = str(loaded.config_path.resolve())

            original_cwd = os.getcwd()
            os.chdir(unrelated_tmp)
            try:
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = cli.main(["inspect-basis", "--config", config_path, "--dry-run"])
            finally:
                os.chdir(original_cwd)

            self.assertEqual(code, 0, err.getvalue())
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["status"], "dry_run")
            self.assertTrue(payload["ready"])
            self.assertEqual(Path(payload["basis_output_path"]), (Path(config_tmp) / "basis.pt").resolve())


if __name__ == "__main__":
    unittest.main()
