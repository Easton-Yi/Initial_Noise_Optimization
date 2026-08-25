#!/usr/bin/env python3
"""One resumable CLI for validate, generate, metrics, analyze, and all."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

from analysis import analyze_run
from io_utils import alpha_token, condition_id, ensure_immutable_run, file_hash, read_jsonl, tensor_hash, write_json, write_jsonl
from metric_runner import evaluate_run
from model_adapters import GenerationConfig, T2IModelAdapter, build_adapter
from noise_methods import construct_noise, load_or_create_noise_batch, noise_statistics


REQUIRED_TOP_LEVEL = {"experiment", "model", "generation", "blocks", "baseline", "same_phase_floor", "independent_white", "quality_metrics", "diversity_metrics", "analysis"}


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
    if config["experiment"]["normalization_profile"] not in {"per_sample_per_channel_zero_mean_unit_std", "divgen_compat", "none"}: raise ValueError("Invalid normalization profile")
    if config["model"]["adapter"] not in {"flux2_klein", "sdxl_turbo"}: raise ValueError("Only flux2_klein and sdxl_turbo adapters are supported")
    if config["diversity_metrics"]["vendi_clip"].get("enabled", False) and config["diversity_metrics"]["vendi_clip"]["embedding_checkpoint"] != config["quality_metrics"]["clip"]["checkpoint"]:
        raise ValueError("This compact runner uses one frozen CLIP encoder; vendi_clip.embedding_checkpoint must equal quality_metrics.clip.checkpoint")
    for name in ("baseline", "same_phase_floor", "independent_white"):
        section = config[name]
        for alpha in section["alpha_values"]:
            if float(alpha) < 0: raise ValueError(f"{name}: alpha must be non-negative")
        for gamma in section.get("gamma_values", []):
            if not 0 <= float(gamma) <= 1: raise ValueError(f"{name}: gamma must be in [0, 1]")
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
            parsed.append(Condition("same_phase" if family == "same" else "independent_white", float(alpha), float(gamma)))
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
            # PNGs are source artefacts for all later metrics.  --force must not
            # replace them: use a new run id if a condition is incomplete/corrupt.
            if sample_dir.exists():
                raise RuntimeError(f"Incomplete existing generation at {sample_dir}; refusing to overwrite source images")
            sample_dir.mkdir(parents=True, exist_ok=True)
            images = adapter.generate(block["prompt"], latents, generation_config)
            if len(images) != 4: raise RuntimeError(f"Adapter returned {len(images)} images; expected 4")
            records = []
            for index, image in enumerate(images):
                target = sample_dir / f"image_{index:02d}.png"; image.save(target, format="PNG")
                records.append({"run_id": run_dir.name, "model_id": adapter.model_id, **block, "base_index": index,
                                "sample_seed": batch.sample_seeds[index], "method": condition.method, "family": condition.family,
                                "condition_id": condition.identifier, "alpha": condition.alpha, "gamma": condition.gamma,
                                "normalization_profile": config["experiment"]["normalization_profile"], "base_noise_hash": batch.base_hashes[index],
                                "eta_noise_hash": batch.eta_hashes[index], "final_noise_hash": tensor_hash(latents[index]),
                                "image_path": str(target.resolve()), "image_hash": file_hash(target), "generation_config_hash": _generation_hash(config)})
            _save_grid(images, sample_dir / "grid_1x4.png")
            write_jsonl(sample_dir / "samples.jsonl", records)
            write_json(sample_dir / "noise_statistics.json", {"pre_normalization": noise_statistics(_raw_noise(batch, condition)), "post_normalization": noise_statistics(latents)})
    if config["generation"].get("release_model_after_generation", True): adapter.close()
    return run_dir


def _raw_noise(batch, condition: Condition) -> torch.Tensor:
    # Build with no post-processing purely for provenance statistics.
    return construct_noise(batch, condition.family, condition.alpha, condition.gamma, "none")


def _generation_hash(config: dict[str, Any]) -> str:
    from io_utils import sha256_text
    return sha256_text(config["model"], config["generation"])


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
    if args.stage in ("metrics", "all"): evaluate_run(run_dir, config, force=args.force)
    if args.stage in ("analyze", "all"): analyze_run(run_dir, config)


if __name__ == "__main__": main()
