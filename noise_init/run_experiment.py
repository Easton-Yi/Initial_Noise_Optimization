#!/usr/bin/env python3
"""One resumable CLI for validate, generate, metrics, analyze, and all."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

from analysis import analyze_run
from io_utils import alpha_token, condition_id, ensure_immutable_run, file_hash, read_json, read_jsonl, tensor_hash, write_json, write_jsonl
from metric_runner import evaluate_run
from model_adapters import GenerationConfig, T2IModelAdapter, build_adapter
from noise_methods import construct_noise, load_or_create_noise_batch, noise_statistics


REQUIRED_TOP_LEVEL = {"experiment", "model", "generation", "blocks", "baseline", "same_phase_floor", "independent_white", "quality_metrics", "diversity_metrics", "analysis"}
NOISE_FREQUENCY_GRID_VERSION = "divgen_integer_fft_bin_indices_v1"
NOISE_CONSTRUCTION_VERSION = "exact_white_endpoints_profiled_normalization_v2"
LATENT_INJECTION_VERIFICATION_VERSION = "prepare_latents_capture_v1"


@dataclass(frozen=True)
class Condition:
    family: str
    alpha: float
    gamma: float | None

    @property
    def identifier(self) -> str:
        return condition_id(self.family, self.alpha, self.gamma)

    @property
    def method(self) -> str:
        if self.family == "baseline":
            return "white" if self.alpha == 0 else "pink"
        return self.family


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict): raise ValueError("YAML root must be a mapping")
    # This is provenance, not a tunable parameter. It deliberately invalidates
    # runs made with the earlier normalized-frequency implementation.
    config.setdefault("experiment", {}).setdefault("noise_frequency_grid", NOISE_FREQUENCY_GRID_VERSION)
    config["experiment"].setdefault("noise_construction_version", NOISE_CONSTRUCTION_VERSION)
    # This marks runs whose adapter verified that every supplied initial noise
    # tensor reached the pipeline's latent preparation entry point.
    config["experiment"].setdefault("latent_injection_verification", LATENT_INJECTION_VERIFICATION_VERSION)
    # Paths in checked-in YAML are relative to the independent noise_init root,
    # not to the caller's shell directory or an adjacent DivGen checkout.
    cache_dir = config.get("model", {}).get("cache_dir")
    if cache_dir and not Path(cache_dir).is_absolute():
        config["model"]["cache_dir"] = str((path.parent.parent / cache_dir).resolve())
    return config


def validate_config(config: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    missing = REQUIRED_TOP_LEVEL - set(config)
    if missing: raise ValueError(f"Config missing sections: {sorted(missing)}")
    if config["experiment"]["gallery_size"] != 4: raise ValueError("This experiment contract requires gallery_size=4")
    if config["experiment"].get("noise_frequency_grid") != NOISE_FREQUENCY_GRID_VERSION: raise ValueError("Only DivGen integer FFT-bin frequency coordinates are supported")
    if config["experiment"].get("noise_construction_version") != NOISE_CONSTRUCTION_VERSION: raise ValueError("Unsupported noise construction version")
    if config["experiment"].get("latent_injection_verification") != LATENT_INJECTION_VERIFICATION_VERSION: raise ValueError("Unsupported latent injection verification version")
    if config["experiment"]["normalization_profile"] not in {"per_sample_per_channel_zero_mean_unit_std", "divgen_compat", "none"}: raise ValueError("Invalid normalization profile")
    primary_pair = config["analysis"].get("primary_metric_pair", {})
    if not isinstance(primary_pair, dict) or set(primary_pair) != {"quality", "diversity"}: raise ValueError("analysis.primary_metric_pair must contain quality and diversity")
    if primary_pair["quality"] not in {"hpsv3", "clip_cosine"} or primary_pair["diversity"] not in {"dreamsim_mean_pair_distance", "lpips_alex_mean_pair_distance", "vendi_clip"}:
        raise ValueError("analysis.primary_metric_pair contains an unknown metric name")
    if not isinstance(config["analysis"].get("optional_detail_curves", []), list): raise ValueError("analysis.optional_detail_curves must be a list")
    two_panel = config["analysis"].get("two_panel_display")
    if not isinstance(two_panel, dict) or set(two_panel) != {"alpha_values", "gamma_values"}:
        raise ValueError("analysis.two_panel_display must contain alpha_values and gamma_values")
    if not two_panel["alpha_values"] or not two_panel["gamma_values"] or not all(isinstance(value, (int, float)) for value in two_panel["alpha_values"]) or not all(isinstance(value, (int, float)) for value in two_panel["gamma_values"]):
        raise ValueError("analysis.two_panel_display values must be numeric")
    if config["model"]["adapter"] not in {"flux2_klein", "sdxl_turbo"}: raise ValueError("Only flux2_klein and sdxl_turbo adapters are supported")
    if config["diversity_metrics"]["vendi_clip"].get("enabled", False) and config["diversity_metrics"]["vendi_clip"]["embedding_checkpoint"] != config["quality_metrics"]["clip"]["checkpoint"]:
        raise ValueError("This compact runner uses one frozen CLIP encoder; vendi_clip.embedding_checkpoint must equal quality_metrics.clip.checkpoint")
    enabled_quality = {"clip_cosine" if name == "clip" else name for name, section in config["quality_metrics"].items() if section.get("enabled", False)}
    enabled_diversity = {"dreamsim_mean_pair_distance" if name == "dreamsim" else "lpips_alex_mean_pair_distance" if name == "lpips" else name for name, section in config["diversity_metrics"].items() if section.get("enabled", False)}
    if primary_pair["quality"] not in enabled_quality or primary_pair["diversity"] not in enabled_diversity:
        raise ValueError("analysis.primary_metric_pair must refer to enabled metrics")
    for name in ("baseline", "same_phase_floor", "independent_white"):
        section = config[name]
        for alpha in section["alpha_values"]:
            if float(alpha) < 0: raise ValueError(f"{name}: alpha must be non-negative")
        for gamma in section.get("gamma_values", []):
            if not 0 <= float(gamma) <= 1: raise ValueError(f"{name}: gamma must be in [0, 1]")
        if name != "baseline" and section.get("enabled", False) and any(float(gamma) in (0.0, 1.0) for gamma in section.get("gamma_values", [])):
            raise ValueError(f"{name}: gamma=0 and gamma=1 are tensor-test endpoints, not formal generation conditions")
    for name in ("same_phase_floor", "independent_white"):
        section = config[name]
        if not set(map(float, two_panel["alpha_values"])).issubset(set(map(float, section["alpha_values"]))):
            raise ValueError(f"analysis.two_panel_display alpha_values must be present in {name}.alpha_values")
        if not set(map(float, two_panel["gamma_values"])).issubset(set(map(float, section["gamma_values"]))):
            raise ValueError(f"analysis.two_panel_display gamma_values must be present in {name}.gamma_values")
    manifest = (config_path.parent.parent / config["blocks"]["manifest"]).resolve()
    blocks = read_jsonl(manifest)
    if not blocks: raise ValueError(f"Block manifest is empty: {manifest}")
    required = {"block_id", "prompt_id", "prompt", "seed_batch_id", "batch_seed"}
    for block in blocks:
        if required - set(block): raise ValueError(f"Block missing fields: {required - set(block)}")
    if len({block["block_id"] for block in blocks}) != len(blocks): raise ValueError("block_id values must be unique")
    return blocks


def all_conditions(config: dict[str, Any], selected: str | None = None) -> list[Condition]:
    values = [Condition("baseline", float(alpha), None) for alpha in config["baseline"]["alpha_values"] if config["baseline"].get("enabled", True)]
    for key, family in (("same_phase_floor", "same_phase"), ("independent_white", "independent_white")):
        section = config[key]
        if section.get("enabled", False):
            values += [Condition(family, float(alpha), float(gamma)) for alpha in section["alpha_values"] for gamma in section["gamma_values"]]
    if not selected: return values
    wanted = set(selected.split(",")); parsed = []
    for token in wanted:
        if token == "white": parsed.append(Condition("baseline", 0.0, None))
        elif token.startswith("pink:"): parsed.append(Condition("baseline", float(token.split(":", 1)[1]), None))
        elif token.startswith("same:") or token.startswith("independent:"):
            family, alpha, gamma = token.split(":")
            gamma_value = float(gamma)
            if gamma_value in (0.0, 1.0): raise ValueError("gamma=0 and gamma=1 are tensor-test endpoints, not generation conditions")
            parsed.append(Condition("same_phase" if family == "same" else "independent_white", float(alpha), gamma_value))
        else: raise ValueError(f"Unknown --conditions token: {token}")
    return parsed


def _run_dir(config: dict[str, Any], config_path: Path, run_id: str | None) -> Path:
    root = (config_path.parent.parent / config.get("paths", {}).get("outputs_root", "outputs")).resolve()
    return root / (run_id or config["experiment"]["name"])


def _sample_path(run_dir: Path, model_id: str, block_id: str, condition: Condition) -> Path:
    return run_dir / "generations" / model_id / block_id / condition.identifier


def _save_grid(images: list[Image.Image], target: Path) -> None:
    width, height = images[0].size
    grid = Image.new("RGB", (width * 4, height))
    for index, image in enumerate(images): grid.paste(image.convert("RGB"), (index * width, 0))
    grid.save(target, format="PNG")


def _is_complete(sample_dir: Path, latents: torch.Tensor) -> bool:
    records = read_jsonl(sample_dir / "samples.jsonl")
    if len(records) != len(latents): return False
    return all(Path(row["image_path"]).exists() and row["final_noise_hash"] == tensor_hash(latents[row["base_index"]]) for row in records)


def generate(config: dict[str, Any], config_path: Path, blocks: list[dict[str, Any]], *, run_id: str | None, selected_conditions: str | None, force: bool, adapter: T2IModelAdapter | None = None) -> Path:
    run_dir = _run_dir(config, config_path, run_id)
    ensure_immutable_run(run_dir, config, force=force)
    write_jsonl(run_dir / "blocks.jsonl", blocks)
    adapter = adapter or build_adapter(config["model"])
    spec = adapter.latent_spec(config["generation"]["height"], config["generation"]["width"], config["experiment"]["gallery_size"])
    generation_config = GenerationConfig(**{key: config["generation"][key] for key in ("height", "width", "num_inference_steps", "guidance_scale", "generation_batch_size", "output_format")})
    conditions = all_conditions(config, selected_conditions)
    for block in blocks:
        noise_cache = run_dir / "noise_cache" / block["block_id"]
        batch = load_or_create_noise_batch(noise_cache, int(config["experiment"]["master_seed"]), block["block_id"], spec.shape, batch_seed=int(block["batch_seed"]))
        for condition in conditions:
            latents = construct_noise(batch, condition.family, condition.alpha, condition.gamma, config["experiment"]["normalization_profile"])
            sample_dir = _sample_path(run_dir, adapter.model_id, block["block_id"], condition)
            if _is_complete(sample_dir, latents) and not force: continue
            if _is_same_phase_white_alias(condition):
                _write_same_phase_white_alias(run_dir, adapter.model_id, block, condition, batch, latents, config)
                continue
            # PNGs are source artefacts for all later metrics.  --force must not
            # replace them: use a new run id if a condition is incomplete/corrupt.
            if sample_dir.exists():
                raise RuntimeError(f"Incomplete existing generation at {sample_dir}; refusing to overwrite source images")
            sample_dir.mkdir(parents=True, exist_ok=True)
            images = adapter.generate(block["prompt"], latents, generation_config)
            if len(images) != 4: raise RuntimeError(f"Adapter returned {len(images)} images; expected 4")
            prepared_hashes = getattr(adapter, "last_generated_latent_hashes", None)
            if prepared_hashes is not None and len(prepared_hashes) != len(images):
                raise RuntimeError("Adapter returned incomplete initial-latent verification records")
            records = []
            for index, image in enumerate(images):
                target = sample_dir / f"image_{index:02d}.png"; image.save(target, format="PNG")
                records.append({"run_id": run_dir.name, "model_id": adapter.model_id, **block, "base_index": index,
                                "sample_seed": batch.sample_seeds[index], "method": condition.method, "family": condition.family,
                                "condition_id": condition.identifier, "alpha": condition.alpha, "gamma": condition.gamma,
                                "normalization_profile": config["experiment"]["normalization_profile"], "base_noise_hash": batch.base_hashes[index],
                                "eta_noise_hash": batch.eta_hashes[index], "final_noise_hash": tensor_hash(latents[index]),
                                "adapter_prepared_latent_hash": None if prepared_hashes is None else prepared_hashes[index],
                                "image_path": str(target.resolve()), "image_hash": file_hash(target), "generation_config_hash": _generation_hash(config)})
            _save_grid(images, sample_dir / "grid_1x4.png")
            write_jsonl(sample_dir / "samples.jsonl", records)
            write_json(sample_dir / "noise_statistics.json", {"pre_normalization": noise_statistics(_raw_noise(batch, condition)), "post_normalization": noise_statistics(latents)})
    if config["generation"].get("release_model_after_generation", True): adapter.close()
    return run_dir


def _is_same_phase_white_alias(condition: Condition) -> bool:
    """The alpha=0 same-phase floor is exactly the normalized base-white gallery.

    It remains a logical member of every fixed-gamma curve, but never requires a
    duplicate model invocation or duplicate PNG.  Gamma endpoints are handled
    by tensor tests and are intentionally absent from the formal YAML grids.
    """
    return condition.family == "same_phase" and condition.alpha == 0.0 and condition.gamma not in (None, 0.0, 1.0)


def _write_same_phase_white_alias(run_dir: Path, model_id: str, block: dict[str, Any], condition: Condition, batch, latents: torch.Tensor, config: dict[str, Any]) -> None:
    source_condition = Condition("baseline", 0.0, None)
    source_dir = _sample_path(run_dir, model_id, block["block_id"], source_condition)
    source_records = read_jsonl(source_dir / "samples.jsonl")
    if not _is_complete(source_dir, latents):
        raise RuntimeError(f"Same-phase white alias requires completed baseline white gallery: {source_dir}")
    target_dir = _sample_path(run_dir, model_id, block["block_id"], condition)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in source_records:
        records.append({**source, "method": condition.method, "family": condition.family, "condition_id": condition.identifier,
                        "alpha": condition.alpha, "gamma": condition.gamma, "final_noise_hash": tensor_hash(latents[source["base_index"]]),
                        "alias_of_condition_id": source_condition.identifier, "alias_of_image_path": source["image_path"], "is_exact_alias": True})
    write_jsonl(target_dir / "samples.jsonl", records)
    write_json(target_dir / "noise_statistics.json", {"alias_of_condition_id": source_condition.identifier,
               "pre_normalization": noise_statistics(_raw_noise(batch, condition)), "post_normalization": noise_statistics(latents)})


def _raw_noise(batch, condition: Condition) -> torch.Tensor:
    # Build with no post-processing purely for provenance statistics.
    return construct_noise(batch, condition.family, condition.alpha, condition.gamma, "none")


def _generation_hash(config: dict[str, Any]) -> str:
    from io_utils import sha256_text
    return sha256_text(config["model"], config["generation"])


def _analysis_config_for_existing_run(run_dir: Path, requested_config: dict[str, Any]) -> dict[str, Any]:
    """Use frozen statistical settings while permitting a newer figure layout.

    Analysis is derived from immutable metrics, so it must not silently adopt a
    new bootstrap policy.  Presentation-only options are safe to refresh and
    are recorded separately; this lets old complete runs be replotted after a
    plotting-code upgrade without regenerating images or metric values.
    """
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"No immutable run manifest found at {manifest_path}")
    manifest = read_json(manifest_path)
    frozen = manifest.get("resolved_config")
    if not isinstance(frozen, dict) or not isinstance(frozen.get("analysis"), dict):
        raise RuntimeError(f"Invalid resolved config in {manifest_path}")
    config = copy.deepcopy(frozen)
    presentation = requested_config["analysis"]
    config["analysis"].update({
        "primary_metric_pair": presentation["primary_metric_pair"],
        "optional_detail_curves": presentation.get("optional_detail_curves", []),
        "two_panel_display": presentation["two_panel_display"],
    })
    write_json(run_dir / "analysis" / "analysis_manifest.json", {
        "source_run_config_hash": manifest.get("config_hash"),
        "frozen_statistical_analysis": frozen["analysis"],
        "presentation_analysis": {"primary_metric_pair": config["analysis"]["primary_metric_pair"], "optional_detail_curves": config["analysis"]["optional_detail_curves"], "two_panel_display": config["analysis"]["two_panel_display"]},
    })
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--stage", required=True, choices=("validate", "generate", "metrics", "analyze", "all"))
    parser.add_argument("--run-id"); parser.add_argument("--prompt"); parser.add_argument("--batch-seed", type=int); parser.add_argument("--conditions"); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); config_path = Path(args.config).resolve(); config = load_config(config_path); blocks = validate_config(config, config_path)
    if args.prompt is not None:
        if args.batch_seed is None: parser.error("--prompt requires --batch-seed")
        blocks = [{"block_id": f"smoke_{args.batch_seed}", "prompt_id": "smoke", "prompt": args.prompt, "seed_batch_id": "smoke", "batch_seed": args.batch_seed}]
    if args.stage == "validate":
        print(f"valid: {len(blocks)} blocks, {len(all_conditions(config, args.conditions))} conditions"); return
    run_dir = _run_dir(config, config_path, args.run_id)
    if args.stage in ("generate", "all"):
        run_dir = generate(config, config_path, blocks, run_id=args.run_id, selected_conditions=args.conditions, force=args.force)
    elif args.stage == "metrics":
        # Metric values are scientific artefacts, so reject a run created by a
        # different noise or metric configuration.
        ensure_immutable_run(run_dir, config, force=args.force)
    elif args.stage == "analyze":
        config = _analysis_config_for_existing_run(run_dir, config)
    if args.stage in ("metrics", "all"): evaluate_run(run_dir, config, force=args.force)
    if args.stage in ("analyze", "all"): analyze_run(run_dir, config)


if __name__ == "__main__": main()
