#!/usr/bin/env python3
"""Generic metadata subsets using the full run's integrity and prompt statistics."""
from __future__ import annotations
import argparse
import hashlib
import re
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from analysis import analyze_run, _write_csv, _read_csv, _two_panel_rows, _metric_label
from io_utils import read_jsonl, write_json, write_jsonl, file_hash
from result_integrity import validate_analysis_inputs
from run_support import resolve_run, add_display_arguments, presentation_config, new_output, input_hashes


FAMILIES = ("same_phase", "independent_white")
LABELS = {"same_phase": "Same-phase", "independent_white": "Independent-white",
          **{name: _metric_label(name) for name in ("hpsv3", "clip_cosine", "dreamsim_mean_pair_distance", "lpips_alex_mean_pair_distance", "vendi_clip")}}

def draw(ax, points, family, mode):
    baseline = sorted([p for p in points if p["family"] == "baseline"], key=lambda p: p["alpha"])
    if baseline:
        ax.plot([p["diversity"] for p in baseline], [p["quality"] for p in baseline],
                marker="o", linestyle="None", color="black", markersize=4, label="Baseline")
        for p in baseline:
            ax.annotate(f"α={p['alpha']:g}", (p["diversity"], p["quality"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    ours = [] if family == "baseline" else [p for p in points if p["family"] == family]
    key, varying, symbol = ("alpha", "gamma", "α") if mode == "fixed_alpha" else ("gamma", "alpha", "γ")
    values = sorted({p[key] for p in ours})
    cmap = plt.get_cmap("Blues" if family == "same_phase" else "Oranges")
    for i, value in enumerate(values):
        series = sorted([p for p in ours if p[key] == value], key=lambda p: p[varying])
        ax.plot([p["diversity"] for p in series], [p["quality"] for p in series],
                "o-", color=cmap(.45 + .45 * i / max(1, len(values)-1)), label=f"{symbol}={value:g}")
        if mode == "fixed_alpha":
            for p in series:
                ax.annotate(f"γ={p['gamma']:g}", (p["diversity"], p["quality"]),
                            xytext=(4, -10), textcoords="offset points", fontsize=7)
    ax.grid(alpha=.22)
    ax.legend(fontsize=8, frameon=False)


def plot_subsets(output, data, titles, quality, diversity, zoom):
    # Keep one common viewport for every subset and both method panels.
    all_points = [p for rows in data.values() for p in rows]
    if not all_points:
        return
    limits = {}
    for name in ("diversity", "quality"):
        values = [p[name] for p in all_points]
        low, high = min(values), max(values)
        pad = max((high-low)*.12, 1e-5)
        limits[name] = low-pad, high+pad
    for mode in ("fixed_alpha", "fixed_gamma"):
        for subset, points in data.items():
            families = [f for f in FAMILIES if any(p["family"] == f for p in points)] or ["baseline"]
            fig, axes = plt.subplots(1, len(families), figsize=(14 if len(families) == 2 else 7, 5.5), sharex=True, sharey=True, squeeze=False, constrained_layout=True)
            for ax, family in zip(axes[0], families):
                draw(ax, points, family, mode)
                ax.set(title=LABELS.get(family, "Baseline"), xlabel=LABELS.get(diversity, diversity),
                       ylabel=LABELS.get(quality, quality), xlim=limits["diversity"], ylim=limits["quality"])
                ax.tick_params(labelleft=True)
            suffix = "zoomed" if zoom else "full"
            fig.suptitle(f"{titles[subset]} · {mode.replace('_', ' ')} · {suffix}\nEqual-weight prompt means; common axes across all subsets", fontsize=12)
            target = output / subset / f"{mode}_qd_two_panel_{suffix}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=180)
            plt.close(fig)



def category(block, field=None):
    def get(path):
        value = block
        for key in path.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        return value
    if field:
        value = get(field)
    else:
        tag, fallback = get("geneval_metadata.tag"), get("category")
        if tag is not None and fallback is not None and tag != fallback:
            raise ValueError(f"Conflicting category/tag for {block.get('block_id', block.get('prompt_id'))}")
        value = tag if tag is not None else fallback
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"Category must be nonempty text: {value!r}")
    return value


def classify(blocks, *, field=None, manifest=None, exclude_missing=False):
    labels = {bid: category(block, field) for bid, block in blocks.items()}
    if manifest:
        external = read_jsonl(manifest)
        seen = set()
        for row in external:
            # A manifest may label specific blocks or all groups of an identified prompt.
            key = ("block", str(row["block_id"])) if "block_id" in row else ("prompt", str(row.get("prompt_id")))
            if key in seen:
                raise ValueError(f"Duplicate external category identity: {key}")
            seen.add(key)
            matches = [b for b in blocks.values() if
                       (str(b["block_id"]) == key[1] if key[0] == "block" else str(b["prompt_id"]) == key[1])]
            if not matches or any(row.get("prompt_id") != b["prompt_id"] or row.get("prompt") != b["prompt"] for b in matches):
                raise ValueError(f"External category identity/text mismatch: {key}")
            value = category(row, field)
            for block in matches:
                bid = str(block["block_id"])
                if value is not None:
                    if labels[bid] is not None and labels[bid] != value:
                        raise ValueError(f"Category conflict: {bid}")
                    labels[bid] = value
    missing = [bid for bid, value in labels.items() if value is None]
    if len(missing) == len(labels):
        return {}, missing
    if missing and not exclude_missing:
        raise ValueError(f"Missing categories for {len(missing)} blocks; use --exclude-missing explicitly: {missing[:3]}")
    prompts = {}
    groups = defaultdict(list)
    for bid, value in labels.items():
        if value is None:
            continue
        pid = str(blocks[bid]["prompt_id"])
        if pid in prompts and prompts[pid] != value:
            raise ValueError(f"Conflicting categories across seed groups for prompt {pid}")
        prompts[pid] = value
        groups[value].append(bid)
    return dict(groups), missing


def analyze_subsets(run, *, options=None, field=None, manifest=None, exclude_missing=False):
    contract = validate_analysis_inputs(run)
    groups, missing = classify(contract.blocks, field=field, manifest=manifest, exclude_missing=exclude_missing)
    output = new_output(run, "subsets")
    provenance = {"source": str(run), "input_hashes": input_hashes(run),
                  "field": field or "geneval_metadata.tag, fallback category", "excluded_blocks": missing,
                  "external_manifest": str(manifest) if manifest else None,
                  "external_manifest_hash": file_hash(manifest) if manifest else None,
                  "status": "unavailable" if not groups else "partial" if missing else "passed",
                  "reason": "No category metadata; overall analysis remains available" if not groups else None}
    config = presentation_config(contract.config, options)
    provenance["aggregation_unit"] = "prompt_equal_after_seed_group_mean"
    provenance["bootstrap_unit"] = "prompt"
    provenance["bootstrap_settings"] = {key: contract.config["analysis"][key] for key in ("bootstrap_seed", "bootstrap_replicates", "confidence_level")}
    provenance["display_selection"] = config["analysis"]["two_panel_display"]
    mapping, curves, titles = [], {}, {}
    pair = config["analysis"]["primary_metric_pair"]
    quality, diversity = pair["quality"], pair["diversity"]
    for label, bids in sorted(groups.items()):
        token = (label if re.fullmatch(r"[A-Za-z0-9_-]+", label) and not label.startswith("category_")
                 else "category_" + hashlib.sha256(label.encode()).hexdigest())
        dest = output / token
        analyze_run(run, config, block_ids=bids, output_dir=dest, contract=contract, make_plots=False)
        aggregates = _read_csv(dest / "tables" / "all_curve_points.csv")
        metric_rows = {}
        for row in aggregates:
            for axis in ("quality", "diversity"):
                metric = row[axis + "_metric"]
                metric_rows[row["condition_id"], metric] = {"subset": label, "condition_id": row["condition_id"],
                    "family": row["family"], "alpha": row["alpha"], "gamma": row["gamma"], "metric": metric,
                    "mean": row[axis], "prompt_count": row["prompt_count"], "block_count": row["block_count"]}
        _write_csv(dest / "metric_summary.csv", list(metric_rows.values()))
        points = [dict(row, subset=label, alpha=float(row["alpha"]), gamma=float(row["gamma"]) if row["gamma"] else "",
                       quality=float(row["quality"]), diversity=float(row["diversity"])) for row in aggregates
                  if row["quality_metric"] == quality and row["diversity_metric"] == diversity]
        _write_csv(dest / "curve_points.csv", points)
        curves[token] = _two_panel_rows(points, config["analysis"]["two_panel_display"])
        count = len({contract.blocks[bid]["prompt_id"] for bid in bids})
        titles[token] = f"{label.replace('_', ' ').capitalize()} (n={count} prompts)"
        write_jsonl(dest / "blocks.jsonl", [contract.blocks[bid] for bid in bids])
        write_jsonl(dest / "prompts.jsonl", [{"prompt_id": pid, "prompt": text} for pid, text in sorted(
            {(str(contract.blocks[b]["prompt_id"]), contract.blocks[b]["prompt"]) for b in bids})])
        mapping.append({"category": label, "directory": token, "block_count": len(bids),
                        "prompt_count": len({contract.blocks[b]["prompt_id"] for b in bids})})
    plot_subsets(output, curves, titles, quality, diversity, False)
    plot_subsets(output, curves, titles, quality, diversity, True)
    _write_csv(output / "categories.csv", mapping)
    write_json(output / "analysis_info.json", {**provenance, "categories": mapping})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id"); parser.add_argument("--run-dir")
    parser.add_argument("--category-field")
    parser.add_argument("--category-manifest", "--manifest", dest="category_manifest")
    parser.add_argument("--exclude-missing", action="store_true")
    add_display_arguments(parser)
    args = parser.parse_args()
    try:
        print(analyze_subsets(resolve_run(args.run_id, args.run_dir), options=vars(args), field=args.category_field,
                              manifest=args.category_manifest, exclude_missing=args.exclude_missing))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__": main()
