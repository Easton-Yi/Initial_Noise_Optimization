"""Static import-graph check: compat_generation.py and compat_metrics.py must
never import each other, so the generation venv's diffusers/newer-transformers
pin and the metrics venv's older-transformers/HPSv3 pin never need to coexist
in one importable graph. Checked via ``ast`` inspection of each module's own
source rather than by actually importing both (which would prove nothing
about venv separation on a machine that happens to have both stacks
installed) -- this is the "simulate the metrics venv's pin" check the plan
calls for.
"""
import ast
import importlib.util
import unittest


def _imported_module_names(module_name: str) -> set[str]:
    spec = importlib.util.find_spec(module_name)
    source = spec.loader.get_source(module_name)
    tree = ast.parse(source, filename=spec.origin)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class CompatShimIndependenceTests(unittest.TestCase):
    def test_compat_generation_never_references_compat_metrics(self):
        imported = _imported_module_names("pc_specific_psd.compat_generation")
        self.assertFalse(
            any("compat_metrics" in name for name in imported),
            f"compat_generation.py must not import compat_metrics; found imports: {imported}",
        )

    def test_compat_metrics_never_references_compat_generation(self):
        imported = _imported_module_names("pc_specific_psd.compat_metrics")
        self.assertFalse(
            any("compat_generation" in name for name in imported),
            f"compat_metrics.py must not import compat_generation; found imports: {imported}",
        )

    def test_compat_metrics_is_never_imported_at_module_top_level_elsewhere(self):
        """Every caller other than metrics.py itself must import compat_metrics
        lazily (inside a function), never at module top level -- otherwise
        validate-config/build-basis/probe/generate-psd --dry-run would
        require the metrics venv's stack just to import cli.py.
        """
        import pathlib

        package_dir = pathlib.Path(importlib.util.find_spec("pc_specific_psd").origin).parent
        offenders = []
        for path in package_dir.glob("*.py"):
            if path.stem in ("compat_metrics", "metrics"):
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in tree.body:  # module top level only
                if isinstance(node, ast.Import) and any(alias.name == "pc_specific_psd.compat_metrics" for alias in node.names):
                    offenders.append(path.name)
                elif isinstance(node, ast.ImportFrom) and node.module == "pc_specific_psd.compat_metrics":
                    offenders.append(path.name)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
