"""The only module that touches noise_init's generation-side code.

Loads ``io_utils``, ``noise_methods`` and ``model_adapters`` from the sibling
``noise_init/`` package and re-exports exactly the symbols this package needs.
Deliberately independent of ``compat_metrics.py`` (see that module's
docstring) so commands that never need metrics never require the metrics
venv's pinned ``transformers`` version to be importable.
"""
from __future__ import annotations

from pc_specific_psd._compat_common import load_noise_init_module

_io_utils = load_noise_init_module("io_utils")
_noise_methods = load_noise_init_module("noise_methods")
_model_adapters = load_noise_init_module("model_adapters")

# -- noise_methods -----------------------------------------------------------
NoiseBatch = _noise_methods.NoiseBatch
sample_noise_batch = _noise_methods.sample_noise_batch
load_or_create_noise_batch = _noise_methods.load_or_create_noise_batch
cache_noise_batch = _noise_methods.cache_noise_batch
normalize = _noise_methods.normalize
pink = _noise_methods.pink
pink_filter = _noise_methods.pink_filter
same_phase_floor = _noise_methods.same_phase_floor
independent_white = _noise_methods.independent_white
radial_frequency_grid = _noise_methods.radial_frequency_grid
noise_statistics = _noise_methods.noise_statistics
construct_noise = _noise_methods.construct_noise

# -- model_adapters ------------------------------------------------------------
GenerationConfig = _model_adapters.GenerationConfig
LatentSpec = _model_adapters.LatentSpec
T2IModelAdapter = _model_adapters.T2IModelAdapter
build_adapter = _model_adapters.build_adapter
SDXLTurboAdapter = _model_adapters.SDXLTurboAdapter
Flux2KleinAdapter = _model_adapters.Flux2KleinAdapter

# -- io_utils ------------------------------------------------------------------
derived_seed = _io_utils.derived_seed
tensor_hash = _io_utils.tensor_hash
file_hash = _io_utils.file_hash
sha256_text = _io_utils.sha256_text
canonical_json = _io_utils.canonical_json
read_json = _io_utils.read_json
write_json = _io_utils.write_json
read_jsonl = _io_utils.read_jsonl
write_jsonl = _io_utils.write_jsonl
ensure_immutable_run = _io_utils.ensure_immutable_run
environment_record = _io_utils.environment_record
alpha_token = _io_utils.alpha_token
