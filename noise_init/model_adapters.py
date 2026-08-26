"""Thin, lazy-loading adapters for the two approved diffusion pipelines."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Protocol

import torch
from PIL import Image


@dataclass(frozen=True)
class LatentSpec:
    shape: tuple[int, int, int, int]
    packing: str


@dataclass(frozen=True)
class GenerationConfig:
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    generation_batch_size: int
    output_format: str = "png"


class T2IModelAdapter(Protocol):
    model_id: str

    def latent_spec(self, height: int, width: int, batch_size: int) -> LatentSpec: ...
    def generate(self, prompt: str, latents: torch.Tensor, generation_config: GenerationConfig) -> list[Image.Image]: ...
    def close(self) -> None: ...


def _torch_dtype(name: str) -> torch.dtype:
    try:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {name}") from error


class _DiffusersAdapter:
    def __init__(self, model_config: dict, *, model_id: str):
        self.model_config, self.model_id = model_config, model_id
        self.device = torch.device(model_config["device"])
        self.dtype = _torch_dtype(model_config["dtype"])
        self.pipe = None
        self.last_supplied_latent_hash: str | None = None
        self.last_prepared_latent_hash: str | None = None
        self.last_generated_latent_hashes: list[str] = []

    def _load(self):
        raise NotImplementedError

    def close(self) -> None:
        self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _call_one(self, prompt: str, latent: torch.Tensor, config: GenerationConfig) -> Image.Image:
        if self.pipe is None:
            self._load()
        supplied = latent.detach().to(device=self.device, dtype=self.dtype).contiguous()
        self.last_supplied_latent_hash = _quick_tensor_hash(supplied)
        # ``latents=`` is the public initial-noise entry point.  Capture the
        # tensor that the pipeline subsequently hands to ``prepare_latents``.
        # This prevents an adapter/API mismatch from silently replacing every
        # supplied condition with pipeline-generated noise.
        original_prepare = getattr(self.pipe, "prepare_latents", None)
        if original_prepare is None or not callable(original_prepare):
            raise RuntimeError("Pipeline has no callable prepare_latents; cannot verify initial-noise injection")
        signature = inspect.signature(original_prepare)
        captured: dict[str, str | None] = {"hash": None}

        def checked_prepare_latents(*args, **kwargs):
            try:
                received = signature.bind_partial(*args, **kwargs).arguments.get("latents")
            except TypeError:
                received = kwargs.get("latents")
            if isinstance(received, torch.Tensor):
                captured["hash"] = _quick_tensor_hash(received)
            return original_prepare(*args, **kwargs)

        self.pipe.prepare_latents = checked_prepare_latents
        try:
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    latents=supplied,
                    height=config.height,
                    width=config.width,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    output_type="pil",
                )
        finally:
            self.pipe.prepare_latents = original_prepare
        self.last_prepared_latent_hash = captured["hash"]
        if self.last_prepared_latent_hash is None:
            raise RuntimeError("Pipeline did not pass supplied latents to prepare_latents")
        if self.last_prepared_latent_hash != self.last_supplied_latent_hash:
            raise RuntimeError("Pipeline prepare_latents received noise different from the supplied initial latent")
        image = result.images[0]
        if not isinstance(image, Image.Image):
            raise TypeError("Pipeline did not return a PIL image")
        return image

    def generate(self, prompt: str, latents: torch.Tensor, generation_config: GenerationConfig) -> list[Image.Image]:
        if latents.ndim != 4:
            raise ValueError("Adapters accept unpacked 4D initial latents only")
        # Sequential generation is intentional: it preserves base-index order and avoids
        # treating the gallery as a model batch requirement on a 24 GB GPU.
        self.last_generated_latent_hashes = []
        images = []
        for index in range(latents.shape[0]):
            images.append(self._call_one(prompt, latents[index:index + 1], generation_config))
            # _call_one raises before returning when injection cannot be
            # verified, so this record is always a verified pipeline input.
            assert self.last_prepared_latent_hash is not None
            self.last_generated_latent_hashes.append(self.last_prepared_latent_hash)
        return images


class Flux2KleinAdapter(_DiffusersAdapter):
    """FLUX.2 Klein receives an unpacked spatial latent; Diffusers packs exactly once."""
    def __init__(self, model_config: dict):
        super().__init__(model_config, model_id="flux2_klein")

    def _load(self) -> None:
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as error:
            raise ImportError("FLUX generation requires diffusers>=0.37 and its model dependencies") from error
        kwargs = {"torch_dtype": self.dtype}
        if self.model_config.get("revision"):
            kwargs["revision"] = self.model_config["revision"]
        if self.model_config.get("cache_dir"):
            kwargs["cache_dir"] = self.model_config["cache_dir"]
        token = self.model_config.get("token")
        if token:
            kwargs["token"] = token
        self.pipe = Flux2KleinPipeline.from_pretrained(self.model_config["checkpoint"], **kwargs)
        if self.model_config.get("cpu_offload", False):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

    def latent_spec(self, height: int, width: int, batch_size: int) -> LatentSpec:
        if self.pipe is None:
            self._load()
        scale = int(self.pipe.vae_scale_factor)
        channels = int(self.pipe.transformer.config.in_channels)
        if height % (scale * 2) or width % (scale * 2):
            raise ValueError("FLUX height/width must be divisible by vae_scale_factor * 2")
        return LatentSpec((batch_size, channels, height // (scale * 2), width // (scale * 2)), "pipeline_internal_2x2")


class SDXLTurboAdapter(_DiffusersAdapter):
    def __init__(self, model_config: dict):
        super().__init__(model_config, model_id="sdxl_turbo")

    def _load(self) -> None:
        try:
            from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline
        except ImportError as error:
            raise ImportError("SDXL generation requires diffusers and transformers") from error
        kwargs = {"torch_dtype": self.dtype}
        if self.model_config.get("cache_dir"):
            kwargs["cache_dir"] = self.model_config["cache_dir"]
        # This VAE is the validated adjacent-repository choice; record it in the config.
        vae_checkpoint = self.model_config.get("vae_checkpoint")
        if vae_checkpoint:
            self_vae = AutoencoderKL.from_pretrained(vae_checkpoint, torch_dtype=self.dtype, cache_dir=self.model_config.get("cache_dir"))
            kwargs["vae"] = self_vae
        self.pipe = StableDiffusionXLPipeline.from_pretrained(self.model_config["checkpoint"], **kwargs)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        if self.model_config.get("cpu_offload", False):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

    def latent_spec(self, height: int, width: int, batch_size: int) -> LatentSpec:
        if self.pipe is None:
            self._load()
        scale = int(self.pipe.vae_scale_factor)
        if height % scale or width % scale:
            raise ValueError("SDXL height/width must be divisible by vae_scale_factor")
        return LatentSpec((batch_size, int(self.pipe.unet.config.in_channels), height // scale, width // scale), "none")


def build_adapter(model_config: dict) -> T2IModelAdapter:
    adapter = model_config["adapter"]
    if adapter == "flux2_klein":
        return Flux2KleinAdapter(model_config)
    if adapter == "sdxl_turbo":
        return SDXLTurboAdapter(model_config)
    raise ValueError(f"Unknown model adapter: {adapter}")


def _quick_tensor_hash(tensor: torch.Tensor) -> str:
    # Avoid importing experiment utilities into a production model-loading path.
    import hashlib
    return hashlib.sha256(tensor.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()
