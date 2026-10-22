"""Artist snapshots captured from the original renderers using identical inputs.

Only presentation is compared here. Statistical calculations have separate tests.
The fixture deliberately uses the old display grid so changing grid selection
cannot conceal a style regression. No model, source run, or external file needed.
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
import numpy as np
import torch

import analysis
import plot_zoomed
import analyze_prompt_subset as subsets
import noise_validation as spectral
from run_support import Condition

PAIR = ("hpsv3", "dreamsim_mean_pair_distance")
ALPHAS = (.8, .9)
GAMMAS = {"same_phase": (.025, .038, .05), "independent_white": (.75, .825, .9)}
BASELINE = (.2, .3, .4, .5)


def plot_rows():
    rows = []
    for family, alphas in (("baseline", BASELINE), ("same_phase", ALPHAS), ("independent_white", ALPHAS)):
        for alpha in alphas:
            for gamma in (None,) if family == "baseline" else GAMMAS[family]:
                meta = {"family": family, "alpha": alpha, "gamma": "" if gamma is None else gamma}
                rows.append({**meta, "condition_id": Condition(family, alpha, gamma).identifier,
                             "curve_id": analysis._curve_id(meta), "quality_metric": PAIR[0], "diversity_metric": PAIR[1],
                             "quality": 1 - .3 * alpha + .05 * (gamma or 0),
                             "diversity": .1 + .2 * alpha - .04 * (gamma or 0)})
    return rows


def rounded(value):
    return np.asarray(value, dtype=float).round(8).tolist()


def axis_snapshot(axis):
    legend = axis.get_legend()
    return {
        "title": axis.get_title(), "labels": [axis.get_xlabel(), axis.get_ylabel()],
        "limits": rounded([axis.get_xlim(), axis.get_ylim()]),
        "ticks": [rounded(axis.get_xticks()), rounded(axis.get_yticks())],
        "tick_fonts": [[t.get_fontsize() for t in axis.get_xticklabels()], [t.get_fontsize() for t in axis.get_yticklabels()]],
        "lines": [{"x": rounded(l.get_xdata()), "y": rounded(l.get_ydata()),
                   "color": rounded(to_rgba(l.get_color())), "marker": l.get_marker(),
                   "markersize": l.get_markersize(), "width": l.get_linewidth(),
                   "style": l.get_linestyle(), "alpha": l.get_alpha(), "label": l.get_label()} for l in axis.lines],
        "collections": [{"offsets": rounded(c.get_offsets()), "colors": rounded(c.get_facecolors()),
                         "sizes": rounded(c.get_sizes()) if hasattr(c, "get_sizes") else []} for c in axis.collections],
        "images": [{"values": rounded(i.get_array()), "cmap": i.get_cmap().name, "limits": rounded(i.get_clim())} for i in axis.images],
        "texts": [{"text": t.get_text(), "size": t.get_fontsize(), "position": rounded(t.get_position())} for t in axis.texts],
        "grid": [{"visible": l.get_visible(), "alpha": l.get_alpha()} for l in axis.get_xgridlines()],
        "legend": None if legend is None else {"texts": [t.get_text() for t in legend.get_texts()],
                    "sizes": [t.get_fontsize() for t in legend.get_texts()], "frame": legend.get_frame_on(),
                    "location": legend._loc, "columns": legend._ncols},
        "children": [axis_snapshot(a) for a in axis.child_axes],
    }


def capture_renderers(main=analysis, zoom=plot_zoomed, subset=subsets, noise=spectral, *, legacy=False, render_dir=None):
    captures = {}
    savefig = Figure.savefig
    def capture(figure, filename, **kwargs):
        figure.canvas.draw()
        key = Path(filename).name
        captures[key] = {"size": rounded(figure.get_size_inches()), "save": kwargs,
                         "title": None if figure._suptitle is None else figure._suptitle.get_text(),
                         "axes": [axis_snapshot(a) for a in figure.axes],
                         "shared_x": len(figure.axes) > 1 and figure.axes[0].get_shared_x_axes().joined(*figure.axes[:2]),
                         "shared_y": len(figure.axes) > 1 and figure.axes[0].get_shared_y_axes().joined(*figure.axes[:2])}
        if render_dir:
            render_dir.mkdir(parents=True, exist_ok=True)
            savefig(figure, render_dir / key, **kwargs)
    rows = plot_rows()
    display = {"baseline": {"alpha_values": BASELINE}, **{f: {"alpha_values": ALPHAS, "gamma_values": g} for f, g in GAMMAS.items()}}
    if legacy:
        display = {"alpha_values": ALPHAS, "gamma_values": tuple(sorted(set(sum(GAMMAS.values(), ()))))}
    with tempfile.TemporaryDirectory() as tmp, patch.object(Figure, "savefig", capture):
        output = Path(tmp)
        main._plot_baseline_qd(output / "baseline.png", rows, PAIR)
        main._plot_methods_two_panel(output / "methods.png", rows, PAIR, display)
        main._plot_fixed_alpha_gamma_two_panel(output / "fixed_alpha.png", rows, PAIR, display)
        gains = [{"family": f, "gamma": g, "available": True, "mean_quality_improvement": .1,
                  "bootstrap_ci_low": .05, "bootstrap_ci_high": .16} for f, grid in GAMMAS.items() for g in grid]
        main._plot_matched_diversity_gain(output / "matched_gain.png", gains)
        main._plot_single_curve_comparison(output / "detail.png", rows, PAIR, "same_phase_gamma_0.025")
        zoom.plot_fixed_gamma_two_panel(rows, output / "zoom_gamma.png", *PAIR)
        zoom.plot_fixed_alpha_two_panel(rows, output / "zoom_alpha.png", *PAIR)
        for is_zoom in (False, True):
            subset.plot_subsets(output, {"tag": rows}, {"tag": "Tag (n=2 prompts)"}, *PAIR, is_zoom)
        curves = {Condition(r["family"], r["alpha"], r["gamma"] or None):
                  torch.linspace(-1, -4, 24) * r["alpha"] for r in rows}
        noise._plot_radial_psd_selected(output / "radial.png", curves, 5., ALPHAS, GAMMAS)
        power = torch.linspace(2, 1, 24)
        views = {c: SimpleNamespace(raw_shell_power_sum=power * 4, sample_count=4,
                                    prediction_shell=power) for c in curves}
        if legacy:
            # Isolate rendering: the same already-computed spectra enter both versions.
            with patch.object(noise, "theoretical_shell_curve", return_value=power), patch.object(noise, "independent_white_theoretical_shell", return_value=power):
                noise._plot_theoretical_vs_empirical(output / "prediction.png", list(curves), views, None, None, None, 5., ALPHAS, GAMMAS)
        else:
            noise._plot_theoretical_vs_empirical(output / "prediction.png", list(curves), views, None, None, None, 5., ALPHAS, GAMMAS)
        alpha_grid = {f: list(ALPHAS) for f in GAMMAS}
        mapping = {(c.family, c.alpha, c.gamma): .2 + .3 * (c.gamma or 0) for c in curves if c.family != "baseline"}
        noise._plot_coherence_heatmap(output / "coherence.png", mapping, alpha_grid, GAMMAS)
        noise._plot_equivalent_alpha_heatmap(output / "nearest.png", mapping, alpha_grid, GAMMAS, list(BASELINE))
    return captures


class OriginalPlotStyleTests(unittest.TestCase):
    def test_original_axes_ticks_titles_legends_lines_and_export(self):
        reference = json.loads((Path(__file__).parent / "fixtures/original_plot_style.json").read_text())
        actual = capture_renderers()
        self.assertEqual(actual.keys(), reference.keys())
        for name in actual:
            with self.subTest(plot=name):
                self.assertEqual(actual[name], reference[name])

    def test_interval_not_containing_estimate_keeps_actual_endpoints(self):
        rows = [{"family": "same_phase", "gamma": .05, "available": True,
                 "mean_quality_improvement": .1, "bootstrap_ci_low": .2, "bootstrap_ci_high": .3}]
        def check(figure, *args, **kwargs):
            axis = figure.axes[0]
            interval = axis.collections[0].get_segments()[0]
            np.testing.assert_allclose(interval[:, 1], [.2, .3])
            self.assertTrue(any(np.allclose(line.get_ydata(), [.1]) for line in axis.lines))
        with tempfile.TemporaryDirectory() as tmp, patch.object(Figure, "savefig", autospec=True, side_effect=check) as save:
            analysis._plot_matched_diversity_gain(Path(tmp) / "gain.png", rows)
            self.assertTrue(save.called)

    def test_empty_subset_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            subsets.plot_subsets(Path(tmp), {}, {}, *PAIR, False)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
