"""SDXL-Turbo adapter extensions required by the PCA-specific PSD study.

Wraps ``noise_init``'s ``SDXLTurboAdapter`` (via ``compat_generation``) to fix
two fragile behaviors identified during the compat research pass and never
patched in ``noise_init`` itself (out of this project's scope to modify):

1. ``SDXLTurboAdapter._load()`` builds its ``from_pretrained`` kwargs without
   ever reading ``model_config["revision"]`` -- a requested revision pin is
   silently dropped. ``SDXLTurboAdapterPCA`` passes it through explicitly and
   then *independently* confirms what commit was actually resolved by
   inspecting the local Hugging Face Hub cache (``try_to_load_from_cache``),
   rather than trusting the requested string back into provenance unchecked.
2. ``SDXLTurboAdapter.generate()``/``_call_one()`` never pass ``generator=``
   to the pipeline call, so scheduler randomness is uncontrolled. The PCA
   study's paired-comparison design (Reference vs A+/A-/B+/B- for the same
   draw) requires every condition sharing one ``(prompt_id, block_id,
   base_index)`` draw to see identical scheduler randomness, so only the
   injected PCA noise differs between compared conditions. ``generate()``
   here is the *only* method in this package permitted to invoke the
   pipeline; ``runner.py`` and ``probing.py`` must call it exclusively.

``SDXLTurboAdapterPCA.generate`` intentionally does not implement the generic
``T2IModelAdapter.generate(prompt, latents, generation_config)`` protocol
shape (it requires ``pair_keys`` and ``seed`` too) -- this adapter is used
directly by this package's own callers, never through that generic protocol.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from pc_specific_psd import manifests, psd_editor
from pc_specific_psd.compat_generation import GenerationConfig, SDXLTurboAdapter, derived_seed, same_phase_floor
from pc_specific_psd.patch_codec import OverlapCodec

if TYPE_CHECKING:
    from pc_specific_psd.config import PCASpecificPSDConfig

PAIRED_GENERATOR_NAMESPACE = "sdxl_pipeline_generator"
BASIS_ENCODING_IMAGE_SIZE = 512


class RevisionResolutionError(RuntimeError):
    """Raised when a requested ``revision`` cannot be independently confirmed."""


def resolve_cached_commit_hash(
    repo_id: str,
    *,
    cache_dir: str | None,
    revision: str,
    filename: str = "model_index.json",
) -> str:
    """Independently resolves what commit ``revision`` actually points to.

    Uses ``huggingface_hub``'s local cache index rather than trusting
    ``from_pretrained``'s own bookkeeping: the Hub cache lays snapshots out as
    ``<cache_dir>/models--org--repo/snapshots/<resolved_sha>/<filename>``
    regardless of whether ``revision`` was itself a branch, a tag, or already
    a full commit sha (``try_to_load_from_cache`` resolves a named ref via
    ``refs/<name>`` before ever touching ``snapshots/``), so the parent
    directory name of the resolved path is always the actual resolved commit
    hash. Raises if the cache cannot confirm a resolution at all.
    """
    from huggingface_hub import try_to_load_from_cache

    resolved_path = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir, revision=revision)
    if not isinstance(resolved_path, str):
        raise RevisionResolutionError(
            f"Could not confirm a cached resolution for {repo_id}@{revision} "
            f"(filename={filename!r}); the loaded pipeline's revision cannot be verified"
        )
    return Path(resolved_path).parent.name


@dataclass(frozen=True)
class RevisionProvenance:
    requested_revision: str | None
    resolved_commit_hash: str | None


def _quick_tensor_hash(tensor: torch.Tensor) -> str:
    # Deliberately duplicated from noise_init/model_adapters.py's private
    # helper of the same name rather than imported: compat_generation.py only
    # re-exports public symbols, and this one-line hash is cheap to keep in
    # sync by inspection rather than reaching past that boundary.
    return hashlib.sha256(tensor.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()


def _load_and_preprocess_image(path: str, size: int = BASIS_ENCODING_IMAGE_SIZE) -> torch.Tensor:
    """RGB equal-ratio resize + center crop to ``size``x``size``, normalized to
    the fixed ``[-1, 1]`` VAE input range (prompt.md:137). Returns a
    ``(3, size, size)`` float32 CPU tensor.
    """
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = size / min(width, height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    image = image.resize((resized_width, resized_height), Image.BICUBIC)
    left = (resized_width - size) // 2
    top = (resized_height - size) // 2
    image = image.crop((left, top, left + size, top + size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)  # (C, H, W)
    return tensor * 2.0 - 1.0


class SDXLTurboAdapterPCA(SDXLTurboAdapter):
    """Extends the base adapter with revision enforcement and a mandatory
    paired generator. See module docstring for the two fixed behaviors.
    """

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.revision_provenance: RevisionProvenance | None = None
        self.last_pair_keys: list[tuple[object, ...]] = []
        self.last_generator_seeds: list[int] = []

    def _load(self) -> None:
        try:
            from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline
        except ImportError as error:
            raise ImportError("SDXL generation requires diffusers and transformers") from error
        kwargs = {"torch_dtype": self.dtype}
        if self.model_config.get("cache_dir"):
            kwargs["cache_dir"] = self.model_config["cache_dir"]
        revision = self.model_config.get("revision")
        if revision:
            kwargs["revision"] = revision
        vae_checkpoint = self.model_config.get("vae_checkpoint")
        if vae_checkpoint:
            vae = AutoencoderKL.from_pretrained(
                vae_checkpoint, torch_dtype=self.dtype, cache_dir=self.model_config.get("cache_dir")
            )
            kwargs["vae"] = vae
        self.pipe = StableDiffusionXLPipeline.from_pretrained(self.model_config["checkpoint"], **kwargs)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        if self.model_config.get("cpu_offload", False):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        self._verify_and_record_revision(revision)

    def _verify_and_record_revision(self, revision: str | None) -> None:
        if not revision:
            self.revision_provenance = RevisionProvenance(requested_revision=None, resolved_commit_hash=None)
            return
        resolved = resolve_cached_commit_hash(
            self.model_config["checkpoint"], cache_dir=self.model_config.get("cache_dir"), revision=revision
        )
        self.revision_provenance = RevisionProvenance(requested_revision=revision, resolved_commit_hash=resolved)

    def encode_images_for_basis(self, image_paths: Sequence[str]) -> torch.Tensor:
        """Encoder callback for ``basis.build_pca_basis``: RGB equal-ratio
        resize + center crop to 512x512, VAE **posterior mean** (never a
        sampled draw) via ``vae.encode(batch).latent_dist.mean``, scaled by
        the pipeline's actual ``vae.config.scaling_factor`` -- the real
        latent-space convention, not an assumed constant (prompt.md:137).
        Loads the pipeline lazily (mirrors ``generate()``'s own lazy load) if
        it isn't already loaded. Always returns a float32 CPU tensor: basis
        fitting is not required to inherit generation's float16 instability
        (prompt.md:137 explicitly permits recording VAE precision separately).
        """
        if self.pipe is None:
            self._load()
        batch = torch.stack([_load_and_preprocess_image(path) for path in image_paths], dim=0)
        vae = self.pipe.vae
        with torch.inference_mode():
            batch = batch.to(device=self.device, dtype=self.dtype)
            latents = vae.encode(batch).latent_dist.mean * vae.config.scaling_factor
        return latents.detach().to("cpu", torch.float32).contiguous()

    def generate(
        self,
        prompt: str,
        latents: torch.Tensor,
        pair_keys: Sequence[tuple[object, ...]],
        seed: int,
        generation_config: GenerationConfig,
    ) -> list[Image.Image]:
        """The only method in this package permitted to invoke the pipeline.

        ``pair_keys[i]`` identifies the ``(prompt_id, block_id, base_index)``
        draw that ``latents[i]`` belongs to. Two separate ``generate()`` calls
        (for two different conditions being compared, e.g. Reference and A+)
        that pass the same ``pair_key`` for corresponding images get an
        *identical* scheduler generator seed -- this, not the injected noise,
        is what "paired" means here. A fresh ``torch.Generator`` is
        constructed every call from this deterministic rule alone; none is
        cached or reused across calls. ``seed`` is the caller-supplied master
        seed (mirrors ``derived_seed``'s explicit-master-seed convention used
        throughout this package and ``noise_init`` -- never a hidden global).
        """
        if latents.ndim != 4:
            raise ValueError("Adapters accept unpacked 4D initial latents only")
        if len(pair_keys) != latents.shape[0]:
            raise ValueError(f"Expected {latents.shape[0]} pair_keys (one per image), got {len(pair_keys)}")
        if self.pipe is None:
            self._load()
        self.last_generated_latent_hashes = []
        self.last_pair_keys = list(pair_keys)
        self.last_generator_seeds = []
        images = []
        for index in range(latents.shape[0]):
            generator_seed = derived_seed(seed, PAIRED_GENERATOR_NAMESPACE, pair_keys[index])
            generator = torch.Generator(device=self.device).manual_seed(generator_seed)
            images.append(self._call_one_paired(prompt, latents[index:index + 1], generation_config, generator))
            assert self.last_prepared_latent_hash is not None
            self.last_generated_latent_hashes.append(self.last_prepared_latent_hash)
            self.last_generator_seeds.append(generator_seed)
        return images

    def _call_one_paired(self, prompt: str, latent: torch.Tensor, config: GenerationConfig, generator: torch.Generator) -> Image.Image:
        # Deliberately duplicated from _DiffusersAdapter._call_one with
        # generator= added: noise_init/model_adapters.py is out of scope to
        # modify, and its _call_one has no generator parameter to extend.
        supplied = latent.detach().to(device=self.device, dtype=self.dtype).contiguous()
        self.last_supplied_latent_hash = _quick_tensor_hash(supplied)
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
                    generator=generator,
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


# == Real-model smoke check (plan Change 5) =====================================
# Runs before any real generation batch (the probe stage, and again before the
# full-pilot stage) to catch a latent-injection/shape/wiring bug before real
# GPU time is spent on a batch of images. Deliberately calibration-independent:
# it must pass with no calibration result present at all, since it runs before
# `calibrate` ever produces one for the very first (pre-probing) call site.


@dataclass(frozen=True)
class SmokeCheckResult:
    passed: bool
    reason: Optional[str] = None


_SMOKE_CHECK_PROMPT = manifests.PROMPTS[0]
_SMOKE_CHECK_BLOCK = manifests.SeedBlock(_SMOKE_CHECK_PROMPT.prompt_id, 0, _SMOKE_CHECK_PROMPT.batch_seeds[0])


def smoke_check_cache_key(config: "PCASpecificPSDConfig") -> tuple[str, str]:
    """``(config_hash, basis_hash)`` -- the one cache key both the pre-probing
    and post-preview smoke-check call sites use. ``calibration_hash`` is
    deliberately excluded: the check itself never reads a calibration result,
    so ``workflow.py`` can reuse the pre-probing pass's result once
    calibration exists, and only needs to re-run if config or basis changed.
    Basis-hash logic duplicates ``runner.basis_hash_for``/
    ``probing.probe_basis_hash_for`` rather than importing either -- both of
    those modules import this one, so importing back would close the cycle.
    """
    from pc_specific_psd.compat_generation import file_hash, sha256_text

    config_hash = sha256_text(config.model.as_model_config_dict(), config.generation.config)
    basis_path = config.resolve_root(config.basis.basis_output_path)
    if basis_path is None or not basis_path.exists():
        raise RuntimeError(f"basis file does not exist: {basis_path}; run build-basis first")
    return config_hash, file_hash(basis_path)


def smoke_check(config: "PCASpecificPSDConfig", adapter: SDXLTurboAdapterPCA, *, codec: OverlapCodec) -> SmokeCheckResult:
    """Proves the tau=0 latent path is wired correctly end to end, without
    requiring a ``SELECTED`` calibration registry. Reuses existing functions
    rather than adding new model logic: (1) builds the reference latent via
    ``psd_editor.apply_psd_edit_tau_zero`` and checks it is finite and
    correctly shaped; (2) confirms the tensor actually injected into the
    pipeline (via ``adapter.last_generated_latent_hashes``) is that same
    tensor; (3) compares it against ``same_phase_floor`` computed directly
    with the frozen constants -- the independent ground truth, not a second
    internal derivation of the same shortcut. Any exception or failed
    assertion is caught and returned as ``passed=False``; this never raises.
    """
    try:
        channels = config.basis.channels
        height, width = config.generation.config.height, config.generation.config.width
        if codec.channels != channels or codec.patch_size != config.basis.patch_size:
            return SmokeCheckResult(
                passed=False,
                reason=(
                    f"codec (channels={codec.channels}, patch_size={codec.patch_size}) does not "
                    f"match config.basis (channels={channels}, patch_size={config.basis.patch_size})"
                ),
            )

        base_white = psd_editor.base_white_for_draw(
            _SMOKE_CHECK_BLOCK, manifests.PROBING_BASE_INDEX, channels=channels, height=height, width=width
        )
        latent = psd_editor.apply_psd_edit_tau_zero(base_white)

        if not torch.isfinite(latent).all():
            return SmokeCheckResult(passed=False, reason="tau=0 latent contains non-finite values")
        expected_shape = (1, channels, height, width)
        if tuple(latent.shape) != expected_shape:
            return SmokeCheckResult(
                passed=False, reason=f"tau=0 latent shape {tuple(latent.shape)} != expected {expected_shape}"
            )

        pair_key = (_SMOKE_CHECK_BLOCK.prompt_id, _SMOKE_CHECK_BLOCK.block_id, manifests.PROBING_BASE_INDEX)
        adapter.generate(
            _SMOKE_CHECK_PROMPT.text, latent, [pair_key], seed=config.run.master_seed,
            generation_config=config.generation.config,
        )
        expected_hash = _quick_tensor_hash(latent.detach().to(device=adapter.device, dtype=adapter.dtype).contiguous())
        if not adapter.last_generated_latent_hashes or adapter.last_generated_latent_hashes[0] != expected_hash:
            return SmokeCheckResult(
                passed=False, reason="pipeline received a different latent than the tau=0 latent that was computed"
            )

        ground_truth = same_phase_floor(base_white, psd_editor.SAME_PHASE_ALPHA, psd_editor.SAME_PHASE_GAMMA)
        if not torch.allclose(latent, ground_truth, atol=1e-4, rtol=1e-4):
            return SmokeCheckResult(
                passed=False, reason="tau=0 latent disagrees with the independent same_phase_floor ground truth"
            )

        return SmokeCheckResult(passed=True)
    except Exception as exc:  # smoke_check must never raise -- only report
        return SmokeCheckResult(passed=False, reason=str(exc))
