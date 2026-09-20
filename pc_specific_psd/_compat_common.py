"""Private helper shared by the two independent noise_init compat shims.

Kept tiny and dependency-free on purpose: ``compat_generation.py`` and
``compat_metrics.py`` must be importable without either one pulling in the
other, so this module must never import either of them.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

NOISE_INIT_ROOT = (Path(__file__).resolve().parent.parent / "noise_init").resolve()


def load_noise_init_module(name: str) -> types.ModuleType:
    """Import a top-level module from the sibling ``noise_init/`` package.

    ``noise_init``'s modules use flat same-directory imports (e.g.
    ``from io_utils import ...``), so this temporarily puts ``noise_init/`` on
    ``sys.path`` rather than importing it as a regular subpackage. Bytecode
    writing is suppressed for the duration so importing never creates a
    ``noise_init/__pycache__`` directory, and the loaded module's ``__file__``
    is asserted to actually resolve inside ``noise_init/`` before returning.
    """
    if not NOISE_INIT_ROOT.is_dir():
        raise RuntimeError(f"noise_init package not found at {NOISE_INIT_ROOT}")
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    root = str(NOISE_INIT_ROOT)
    path_added = root not in sys.path
    if path_added:
        sys.path.insert(0, root)
    try:
        module = importlib.import_module(name)
    finally:
        if path_added and root in sys.path:
            sys.path.remove(root)
        sys.dont_write_bytecode = previous_dont_write_bytecode
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"Module {name!r} has no __file__; cannot verify its origin")
    resolved = Path(module_file).resolve()
    if NOISE_INIT_ROOT not in resolved.parents:
        raise RuntimeError(f"Module {name!r} resolved outside noise_init/: {resolved}")
    return module
