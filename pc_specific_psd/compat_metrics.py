"""The only module that touches noise_init's metrics-side code.

Re-exports just ``MetricRunner``. This module must be imported lazily (inside
the function that needs it, never at another module's top level) so that
``validate-config``, ``build-basis``, ``probe`` and ``generate-psd --dry-run``
never require the metrics venv's pinned ``transformers``/HPSv3 stack to be
importable. Deliberately independent of ``compat_generation.py``: neither
module imports the other, so the generation venv's ``diffusers``/newer
``transformers`` pin and the metrics venv's older pin never need to coexist.
"""
from __future__ import annotations

from pc_specific_psd._compat_common import load_noise_init_module

_metric_runner = load_noise_init_module("metric_runner")

MetricRunner = _metric_runner.MetricRunner
evaluate_run = _metric_runner.evaluate_run
