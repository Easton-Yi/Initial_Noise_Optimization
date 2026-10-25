"""Frozen condition grids, run locations and presentation-only selections."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

from io_utils import condition_id, file_hash

NOISE_CONSTRUCTION_VERSION = "exact_white_endpoints_profiled_normalization_v2"
NOISE_FREQUENCY_GRID_VERSION = "divgen_integer_fft_bin_indices_v1"
FAMILY_SECTIONS = {"baseline": "baseline", "same_phase": "same_phase_floor", "independent_white": "independent_white"}


@dataclass(frozen=True)
class Condition:
    family: str
    alpha: float | None
    gamma: float | None

    @property
    def identifier(self):
        return condition_id(self.family, self.alpha, self.gamma)

    @property
    def method(self):
        return ("white" if self.alpha == 0 else "pink") if self.family == "baseline" else self.family


def conditions(config):
    from coupled_sampling import COUPLED_FAMILIES, validate_config
    validate_config(config)
    result = []
    for family, key in FAMILY_SECTIONS.items():
        section = config.get(key, {})
        if not section.get("enabled", family == "baseline"):
            continue
        alphas = list(map(float, section.get("alpha_values", [])))
        gammas = [None] if family == "baseline" else list(map(float, section.get("gamma_values", [])))
        if not alphas or not gammas:
            raise ValueError(f"Enabled {family} has an empty grid")
        if any(not math.isfinite(a) or a < 0 for a in alphas) or any(g is not None and (not math.isfinite(g) or not 0 <= g <= 1) for g in gammas):
            raise ValueError(f"Invalid {family} grid")
        result.extend(Condition(family, a, g) for a in alphas for g in gammas)
    result.extend(Condition(family, None, None) for family in COUPLED_FAMILIES
                  if config.get("coupled_sampling", {}).get(family, {}).get("enabled", False))
    if not result or len({c.identifier for c in result}) != len(result):
        raise ValueError("Frozen conditions are empty or contain duplicate/colliding condition IDs")
    return result


def resolve_run(run_id=None, run_dir=None, *, outputs_root=None):
    if run_dir is not None:
        path = Path(run_dir).resolve()
        if run_id and path.name != run_id:
            raise ValueError("--run-id and --run-dir disagree")
    elif run_id:
        if Path(run_id).name != run_id:
            raise ValueError("Use --run-dir for a path")
        path = Path(outputs_root or Path(__file__).parent / "outputs").resolve() / run_id
    else:
        raise ValueError("Provide --run-dir or --run-id")
    if not (path / "run_manifest.json").is_file():
        raise ValueError(f"Missing frozen run manifest: {path}")
    return path


def parse_selection(spec, grid, label):
    grid = sorted(set(grid))
    if spec is None:
        return grid
    try:
        if ":" in spec:
            lo, hi = map(float, spec.split(":"))
            if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                raise ValueError()
            selected = [v for v in grid if lo <= v <= hi]
        else:
            requested = list(map(float, spec.split(",")))
            if not requested or not all(math.isfinite(v) for v in requested):
                raise ValueError()
            selected = [v for v in grid if v in requested]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}={spec!r}; use finite MIN:MAX or comma-separated values") from exc
    if not selected:
        raise ValueError(f"{label}={spec!r} selects no points; available grid={grid}")
    return selected


def display_selection(config, options=None):
    options = options or {}
    result = {}
    grid = conditions(config)
    for family in FAMILY_SECTIONS:
        available = [c for c in grid if c.family == family]
        if not available:
            continue
        prefix = {"baseline": "baseline", "same_phase": "same", "independent_white": "independent"}[family]
        selected = {}
        for axis in ("alpha",) if family == "baseline" else ("alpha", "gamma"):
            specific = options.get(f"plot_{prefix}_{axis}_range")
            common = options.get(f"plot_{axis}_range") if family != "baseline" else None
            selected[axis + "_values"] = parse_selection(specific if specific is not None else common,
                [getattr(c, axis) for c in available], f"{family} {axis}")
        result[family] = selected
    for c in grid:
        if c.family in {"antithetic", "repulsive"}:
            result[c.family] = {"alpha_values": [None], "gamma_values": []}
    return result


def selected_condition(c, selection):
    if c.family in {"antithetic", "repulsive"}:
        return True
    grid = selection.get(c.family, {})
    return c.alpha in grid.get("alpha_values", []) and (c.gamma is None or c.gamma in grid.get("gamma_values", []))


def add_display_arguments(parser, *, include_metrics=True):
    for name in ("alpha", "gamma", "same-alpha", "same-gamma", "independent-alpha", "independent-gamma", "baseline-alpha"):
        parser.add_argument(f"--plot-{name}-range", help="Display only: inclusive MIN:MAX or comma-separated existing values")
    if include_metrics:
        parser.add_argument("--quality-metric")
        parser.add_argument("--diversity-metric")


def presentation_config(config, options=None):
    from result_integrity import enabled_metric_names
    result = copy.deepcopy(config)
    options = options or {}
    result["analysis"]["two_panel_display"] = display_selection(config, options)
    pair = result["analysis"]["primary_metric_pair"].copy()
    for axis in ("quality", "diversity"):
        if options.get(axis + "_metric"):
            pair[axis] = options[axis + "_metric"]
    q, p, g = enabled_metric_names(config)
    if pair["quality"] not in q or pair["diversity"] not in p + g:
        raise ValueError(f"Requested metric pair is not enabled in the frozen run: {pair}")
    result["analysis"]["primary_metric_pair"] = pair
    return result


def new_output(run_dir, kind):
    """Keep the original public output paths; reruns replace derived results."""
    suffix = {"overall": "", "zoom": "", "subsets": "prompt_subsets", "frequency_validation": "frequency_validation"}
    if kind not in suffix:
        raise ValueError(f"Unknown analysis output kind: {kind}")
    path = Path(run_dir) / "analysis" / suffix[kind]
    path.mkdir(parents=True, exist_ok=True)
    return path


def input_hashes(run_dir):
    root = Path(run_dir)
    paths = [root / "run_manifest.json", root / "blocks.jsonl", *sorted((root / "metrics").glob("*"))]
    return {str(p.relative_to(root)): file_hash(p) for p in paths if p.is_file()}
