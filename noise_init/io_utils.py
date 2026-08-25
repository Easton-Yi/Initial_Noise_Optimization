"""Small immutable-record and path helpers used by every experiment stage."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


def canonical_json(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def derived_seed(master_seed: int, *parts: object) -> int:
    """Stable 63-bit seed; deliberately never uses Python's randomized hash()."""
    return int(sha256_text(master_seed, *parts)[:16], 16) % (2**63 - 1)


def tensor_hash(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    header = canonical_json({"shape": list(tensor.shape), "dtype": "float32"}).encode()
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def write_json(path: str | Path, value: Any, *, replace: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to replace immutable record: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def alpha_token(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def condition_id(family: str, alpha: float, gamma: float | None = None) -> str:
    if family == "baseline":
        return f"baseline/alpha_{alpha_token(alpha)}"
    return f"{family}/alpha_{alpha_token(alpha)}_gamma_{alpha_token(float(gamma))}"


def environment_record() -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
    }


def ensure_immutable_run(run_dir: Path, config: dict[str, Any], *, force: bool) -> None:
    """Create the run provenance once; force never changes source provenance."""
    manifest = run_dir / "run_manifest.json"
    payload = {"resolved_config": config, "config_hash": sha256_text(canonical_json(config))}
    if manifest.exists():
        current = read_json(manifest)
        if current.get("config_hash") != payload["config_hash"]:
            raise RuntimeError(
                f"Run manifest differs at {manifest}; choose a different --run-id. "
                "--force only replaces derived outputs."
            )
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest, payload, replace=False)
    write_json(run_dir / "environment.json", environment_record(), replace=False)
