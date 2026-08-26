"""Versioned quality/diversity metric stage with lazy heavyweight imports."""
from __future__ import annotations

import csv
import gc
import importlib.metadata
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from io_utils import file_hash, read_jsonl, sha256_text, write_json


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


@dataclass
class MetricRunner:
    config: dict[str, Any]
    device: torch.device

    def __post_init__(self) -> None:
        self._clip = self._processor = self._lpips = self._dreamsim = self._hps = None
        self._embeddings: dict[str, np.ndarray] = {}

    def metric_versions(self) -> dict[str, str]:
        return {"torch": torch.__version__, "transformers": _package_version("transformers"), "lpips": _package_version("lpips"),
                "dreamsim": _package_version("dreamsim"), "vendi-score": _package_version("vendi-score"), "hpsv3": _package_version("hpsv3")}

    def release_models(self) -> None:
        """Release all GPU-backed metric models while retaining CPU embeddings."""
        self._clip = None
        self._processor = None
        self._lpips = None
        self._dreamsim = None
        self._hps = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _load_clip(self) -> None:
        if self._clip is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        checkpoint = self.config["quality_metrics"]["clip"]["checkpoint"]
        self._processor = CLIPProcessor.from_pretrained(checkpoint, cache_dir=self.config.get("cache_dir"))
        self._clip = CLIPModel.from_pretrained(checkpoint, cache_dir=self.config.get("cache_dir")).to(self.device).eval()

    def clip_embedding(self, image_path: Path) -> np.ndarray:
        image_digest = file_hash(image_path)
        key = sha256_text("clip-image", self.config["quality_metrics"]["clip"]["checkpoint"], image_digest)
        if key in self._embeddings:
            return self._embeddings[key]
        self._load_clip()
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vector = self._clip.get_image_features(**inputs)[0]
        vector = torch.nn.functional.normalize(vector, dim=0).float().cpu().numpy()
        self._embeddings[key] = vector
        return vector

    def clip_cosine(self, image_path: Path, prompt: str) -> float:
        self._load_clip()
        image = self.clip_embedding(image_path)
        text = self._processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            vector = self._clip.get_text_features(**text)[0]
        vector = torch.nn.functional.normalize(vector, dim=0).float().cpu().numpy()
        return float(np.dot(image, vector))

    def lpips_distance(self, a: Path, b: Path) -> float:
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net=self.config["diversity_metrics"]["lpips"]["backbone"], verbose=False).to(self.device).eval()
        tensors = []
        for path in (a, b):
            image = np.asarray(Image.open(path).convert("RGB").resize((256, 256)), dtype=np.float32) / 127.5 - 1.0
            tensors.append(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device))
        with torch.inference_mode():
            return float(self._lpips(tensors[0], tensors[1]).item())

    def dreamsim_distance(self, a: Path, b: Path) -> float:
        if self._dreamsim is None:
            from dreamsim import dreamsim
            self._dreamsim = dreamsim(pretrained=True, device=str(self.device))
        model, preprocess = self._dreamsim
        with torch.inference_mode():
            return float(model(preprocess(Image.open(a).convert("RGB")).unsqueeze(0).to(self.device), preprocess(Image.open(b).convert("RGB")).unsqueeze(0).to(self.device)).item())

    def hpsv3(self, image_path: Path, prompt: str) -> float:
        """Use only the official HPSv3 package; never fall back to HPSv2."""
        if self._hps is None:
            try:
                from hpsv3 import HPSv3RewardInferencer  # type: ignore
            except ImportError as error:
                raise ImportError("HPSv3 is enabled but unavailable; install its official package. HPSv2 is not substituted.") from error
            self._hps = HPSv3RewardInferencer(device=str(self.device))
        # The installed HPSv3 API accepts image_paths and prompts. It returns
        # (mu, sigma) reward tuples; this experiment records the scalar mu.
        with torch.inference_mode():
            rewards = self._hps.reward(image_paths=[str(image_path)], prompts=[prompt])
        return float(rewards[0][0].item())


def evaluate_run(run_dir: str | Path, config: dict[str, Any], *, force: bool = False) -> None:
    run_dir = Path(run_dir)
    rows = [row for path in run_dir.glob("generations/**/*.jsonl") for row in read_jsonl(path)]
    rows = [row for row in rows if Path(row["image_path"]).name.startswith("image_")]
    if not rows:
        raise RuntimeError("No generated sample records found")
    runner = MetricRunner(config, torch.device(config["model"]["device"]))
    metric_hash = sha256_text(config.get("quality_metrics"), config.get("diversity_metrics"), runner.metric_versions())
    metrics_dir = run_dir / "metrics"
    existing = [] if force else _read_csv(metrics_dir / "per_image.csv")
    # A score cache is keyed by immutable metric input, while the persisted
    # record is keyed by its experimental condition. Exact image aliases must
    # therefore reuse the score *and* receive their own condition record.
    score_cache = {(r["image_hash"], r["metric_config_hash"], r["metric"]): float(r["score"]) for r in existing}
    current_input = {(row["block_id"], row["condition_id"], row["base_index"]) for row in rows}
    per_image: list[dict[str, Any]] = [r for r in existing if (r["block_id"], r["condition_id"], int(r.get("base_index", -1))) in current_input and r.get("metric_config_hash") == metric_hash]
    existing_records = {(r["block_id"], r["condition_id"], int(r.get("base_index", -1)), r["image_hash"], r["metric_config_hash"], r["metric"]) for r in per_image}
    # Run one quality model over the complete dataset, release it, and only
    # then load the next model. HPSv3 and CLIP do not fit concurrently on a
    # 24 GB GPU.
    for metric, enabled in (("clip_cosine", config["quality_metrics"]["clip"]["enabled"]),
                            ("hpsv3", config["quality_metrics"]["hpsv3"]["enabled"])):
        if not enabled:
            continue
        checkpoint_counter = 0
        for row in rows:
            path, digest = Path(row["image_path"]), file_hash(row["image_path"])
            record_key = (row["block_id"], row["condition_id"], row["base_index"], digest, metric_hash, metric)
            if record_key in existing_records:
                continue
            cache_key = (digest, metric_hash, metric)
            score = score_cache.get(cache_key)
            if score is None:
                score = runner.clip_cosine(path, row["prompt"]) if metric == "clip_cosine" else runner.hpsv3(path, row["prompt"])
                score_cache[cache_key] = score
            per_image.append({**_common(row), "metric": metric, "score": score, "image_hash": digest, "metric_config_hash": metric_hash})
            existing_records.add(record_key)
            checkpoint_counter += 1
            if checkpoint_counter % 100 == 0:
                _write_csv(metrics_dir / "per_image.csv", per_image)
        _write_csv(metrics_dir / "per_image.csv", per_image)
        runner.release_models()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["block_id"], row["condition_id"]), []).append(row)
    pairs, per_group = [], _group_quality_rows(per_image)
    # Likewise, evaluate each pairwise diversity model in its own phase.
    for metric, enabled, fn in (("dreamsim_mean_pair_distance", config["diversity_metrics"]["dreamsim"]["enabled"], runner.dreamsim_distance),
                                ("lpips_alex_mean_pair_distance", config["diversity_metrics"]["lpips"]["enabled"], runner.lpips_distance)):
        if enabled:
            for (block_id, condition), gallery in groups.items():
                if len(gallery) != 4:
                    continue
                gallery.sort(key=lambda value: value["base_index"])
                values = []
                for left, right in combinations(gallery, 2):
                    value = fn(Path(left["image_path"]), Path(right["image_path"]))
                    values.append(value)
                    pairs.append({**_common(left), "metric": metric, "left_base_index": left["base_index"], "right_base_index": right["base_index"], "score": value, "metric_config_hash": metric_hash})
                per_group.append({**_common(gallery[0]), "metric": metric, "score": float(np.mean(values)), "n": 6, "metric_config_hash": metric_hash})
            _write_csv(metrics_dir / "per_pair.csv", pairs)
            _write_csv(metrics_dir / "per_group.csv", per_group)
            runner.release_models()

    if config["diversity_metrics"]["vendi_clip"]["enabled"]:
        for (block_id, condition), gallery in groups.items():
            if len(gallery) != 4:
                continue
            gallery.sort(key=lambda value: value["base_index"])
            vectors = np.stack([runner.clip_embedding(Path(item["image_path"])) for item in gallery])
            kernel = vectors @ vectors.T
            kernel = (kernel + kernel.T) / 2
            if not np.allclose(np.diag(kernel), 1.0, atol=1e-4):
                raise RuntimeError("Vendi CLIP kernel diagonal is not one")
            try:
                from vendi_score import vendi
            except ImportError as error:
                raise ImportError("vendi-score is required for vendi_clip") from error
            value = float(vendi.score_K(kernel))
            per_group.append({**_common(gallery[0]), "metric": "vendi_clip", "score": value, "n": 4, "metric_config_hash": metric_hash})
        _write_csv(metrics_dir / "per_group.csv", per_group)
        runner.release_models()
    _write_csv(metrics_dir / "per_image.csv", per_image)
    _write_csv(metrics_dir / "per_pair.csv", pairs)
    _write_csv(metrics_dir / "per_group.csv", per_group)
    write_json(metrics_dir / "metric_manifest.json", {"metric_config_hash": metric_hash, "versions": runner.metric_versions()})


def _common(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("run_id", "model_id", "block_id", "prompt_id", "prompt", "seed_batch_id", "base_index", "condition_id", "method", "alpha", "gamma", "normalization_profile") if key in row}


def _group_quality_rows(per_image: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in per_image:
        groups.setdefault((row["block_id"], row["condition_id"], row["metric"]), []).append(row)
    return [{**_common(values[0]), "metric": metric, "score": float(np.mean([float(v["score"]) for v in values])), "n": len(values), "std": float(np.std([float(v["score"]) for v in values], ddof=0)), "minimum": float(np.min([float(v["score"]) for v in values])), "maximum": float(np.max([float(v["score"]) for v in values])), "metric_config_hash": values[0]["metric_config_hash"]}
            for (_, _, metric), values in groups.items()]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
