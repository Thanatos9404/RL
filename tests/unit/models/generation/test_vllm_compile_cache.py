# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM side of the persistent per-type compile caches.

* ``configure_worker`` hands generation workers their own JIT cache directories
  and, when seeding is on, the cache type their GPU workers wait for.
* ``_patch_vllm_ray_worker_proc_init_hook`` makes RayExecutorV2 GPU workers run
  that wait before they compile.
* ``_patch_vllm_compile_factors_ignore_bundle_indices`` gives engines that
  differ only in their GPUs one compile cache key.
"""

import ast
import importlib.util
import logging
from pathlib import Path
from types import ModuleType

import pytest

from nemo_rl.models.generation.vllm import patches
from nemo_rl.models.generation.vllm.vllm_worker import (
    BaseVllmGenerationWorker,
    _actor_waits_for_compile_cache,
)
from nemo_rl.utils import compile_cache
from tests.unit.models.generation.vllm_patch_source_utils import (
    write_unpatched_copy,
)

REPO_ROOT = Path(__file__).parents[4]
CACHE_SYNC = REPO_ROOT / "tools/external_gym_vllm/cache_sync.sh"
_HOOK_SOURCE = "v1/executor/ray_executor_v2.py"
_HOOK_PATCH_FN = "_patch_vllm_ray_worker_proc_init_hook"
_HOOK_MARKER = "NeMo-RL: worker init hook"
_FACTORS_SOURCE = "envs.py"
_FACTORS_PATCH_FN = "_patch_vllm_compile_factors_ignore_bundle_indices"
# The shape of vllm/envs.py's ignored_factors set that the patch extends.
_SYNTHETIC_ENVS = """\
def compile_factors():
    ignored_factors: set[str] = {
        "MAX_JOBS",
        "VLLM_PORT",
        "VLLM_CACHE_ROOT",
    }
    return ignored_factors
"""
_LOGGER = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _no_ambient_cache_env(monkeypatch):
    for name in (
        compile_cache.CACHE_SYNC_ENV_VAR,
        compile_cache.PERSISTENT_ROOT_ENV_VAR,
        compile_cache.LOCAL_ROOT_ENV_VAR,
        "NRL_VLLM_CACHE_ROOT_PER_ENGINE",
        "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY",
    ):
        monkeypatch.delenv(name, raising=False)
    for var in compile_cache.CACHE_VARS:
        monkeypatch.delenv(f"NRL_VLLM_{var}", raising=False)
    monkeypatch.setenv("VLLM_CACHE_ROOT", "/local/policy/vllm")


@pytest.fixture
def seeding_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv(compile_cache.CACHE_SYNC_ENV_VAR, str(CACHE_SYNC))
    monkeypatch.setenv(compile_cache.PERSISTENT_ROOT_ENV_VAR, str(tmp_path / "p"))
    monkeypatch.setenv(compile_cache.LOCAL_ROOT_ENV_VAR, str(tmp_path / "l"))


def _engine_env_vars() -> dict[str, str]:
    _, env_vars, init_kwargs, _ = BaseVllmGenerationWorker.configure_worker(
        num_gpus=1, bundle_indices=(1, [4, 5, 6, 7]), num_gpus_per_node=8
    )
    assert init_kwargs["seed"] == 1025
    return env_vars


def test_configure_worker_is_unchanged_without_cache_settings():
    env_vars = _engine_env_vars()

    assert env_vars["VLLM_CACHE_ROOT"] == "/local/policy/vllm_1025"
    assert compile_cache.VLLM_CACHE_TYPE_ENV_VAR not in env_vars
    assert "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY" not in env_vars
    assert not set(compile_cache.CACHE_VARS) & set(env_vars)


def test_configure_worker_can_keep_a_stable_vllm_cache_root(monkeypatch):
    monkeypatch.setenv("NRL_VLLM_CACHE_ROOT_PER_ENGINE", "0")

    assert _engine_env_vars()["VLLM_CACHE_ROOT"] == "/local/policy/vllm"


def test_configure_worker_routes_generation_caches_and_seeding(
    monkeypatch, seeding_enabled
):
    monkeypatch.setenv("NRL_VLLM_TRITON_CACHE_DIR", "/local/policy/triton")
    monkeypatch.setenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "FOO,BAR")

    env_vars = _engine_env_vars()

    assert env_vars["TRITON_CACHE_DIR"] == "/local/policy/triton"
    assert env_vars[compile_cache.VLLM_CACHE_TYPE_ENV_VAR] == "policy"
    assert env_vars["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"].split(",") == [
        "FOO",
        "BAR",
        "TRITON_CACHE_DIR",
        compile_cache.VLLM_CACHE_TYPE_ENV_VAR,
    ]


@pytest.mark.parametrize(
    ("executor_backend", "ray_v2_executor", "actor_waits"),
    [
        (None, True, True),
        ("uni", True, True),
        ("ray", False, True),
        # The patched RayWorkerProc waits instead, overlapping engine startup.
        ("ray", True, False),
    ],
)
def test_actor_waits_unless_ray_v2_workers_do(
    executor_backend, ray_v2_executor, actor_waits
):
    assert (
        _actor_waits_for_compile_cache(
            executor_backend, ray_v2_executor=ray_v2_executor
        )
        is actor_waits
    )


def _initialize_worker(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    worker = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "RayWorkerProc"
    )
    return next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == "initialize_worker"
    )


@pytest.mark.vllm
def test_init_hook_runs_after_driver_env_and_before_worker_construction(
    monkeypatch, tmp_path, seeding_enabled
):
    copied = write_unpatched_copy(
        _HOOK_SOURCE, _HOOK_PATCH_FN, tmp_path / "ray_executor_v2.py"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))

    patches._patch_vllm_ray_worker_proc_init_hook(_LOGGER)

    content = copied.read_text()
    assert _HOOK_MARKER in content
    body = _initialize_worker(content).body
    env_loops = [
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.For) and "os.environ" in ast.unparse(node)
    ]
    hook = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.If) and "_nrl_on_ray_worker_init" in ast.unparse(node)
    )
    construct = next(
        i for i, node in enumerate(body) if "super().__init__" in ast.unparse(node)
    )
    assert env_loops and max(env_loops) < hook < construct
    # NeMo RL is imported only when the actor asked for the wait, so serving
    # containers without NeMo RL can run the patched file.
    hook_node = body[hook]
    assert isinstance(hook_node, ast.If)
    assert compile_cache.VLLM_CACHE_TYPE_ENV_VAR in ast.unparse(hook_node.test)


@pytest.mark.vllm
def test_init_hook_patch_is_idempotent(monkeypatch, tmp_path, seeding_enabled):
    copied = write_unpatched_copy(
        _HOOK_SOURCE, _HOOK_PATCH_FN, tmp_path / "ray_executor_v2.py"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_ray_worker_proc_init_hook(_LOGGER)
    once = copied.read_text()

    patches._patch_vllm_ray_worker_proc_init_hook(_LOGGER)

    assert copied.read_text() == once


def test_init_hook_patch_is_skipped_without_persistent_caches(monkeypatch):
    def _unexpected(_relative):
        raise AssertionError("must not touch vLLM without NRL_CACHE_SYNC")

    monkeypatch.setattr(patches, "_get_vllm_file", _unexpected)

    patches._patch_vllm_ray_worker_proc_init_hook(_LOGGER)


def test_init_hook_patch_fails_loudly_on_unknown_source(
    monkeypatch, tmp_path, seeding_enabled
):
    source = tmp_path / "ray_executor_v2.py"
    source.write_text("class RayWorkerProc:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))

    with pytest.raises(RuntimeError, match="init hook patch does not match"):
        patches._patch_vllm_ray_worker_proc_init_hook(_LOGGER)
    assert source.read_text() == "class RayWorkerProc:\n    pass\n"


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"envs_{path.parent.name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.vllm
def test_bundle_indices_drop_out_of_the_compile_cache_key(
    monkeypatch, tmp_path, seeding_enabled
):
    unpatched = write_unpatched_copy(
        _FACTORS_SOURCE, _FACTORS_PATCH_FN, tmp_path / "unpatched/envs.py"
    )
    patched = write_unpatched_copy(
        _FACTORS_SOURCE, _FACTORS_PATCH_FN, tmp_path / "patched/envs.py"
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(patched))

    patches._patch_vllm_compile_factors_ignore_bundle_indices(_LOGGER)

    def factors(source: Path, bundle_indices: str) -> dict[str, object]:
        monkeypatch.setenv("VLLM_RAY_BUNDLE_INDICES", bundle_indices)
        return _load_module(source).compile_factors()

    # Two engines placed on different GPUs of the same placement group.
    assert factors(unpatched, "0,1,2,3") != factors(unpatched, "4,5,6,7")
    assert factors(patched, "0,1,2,3") == factors(patched, "4,5,6,7")
    assert "VLLM_RAY_BUNDLE_INDICES" not in factors(patched, "0,1,2,3")


def test_bundle_indices_patch_is_idempotent(monkeypatch, tmp_path, seeding_enabled):
    source = tmp_path / "envs.py"
    source.write_text(_SYNTHETIC_ENVS)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))

    patches._patch_vllm_compile_factors_ignore_bundle_indices(_LOGGER)
    once = source.read_text()
    patches._patch_vllm_compile_factors_ignore_bundle_indices(_LOGGER)

    assert source.read_text() == once
    assert "VLLM_RAY_BUNDLE_INDICES" in _load_module(source).compile_factors()


def test_bundle_indices_patch_leaves_an_upstream_fix_alone(
    monkeypatch, tmp_path, seeding_enabled
):
    source = tmp_path / "envs.py"
    upstream = _SYNTHETIC_ENVS.replace(
        '        "VLLM_PORT",\n',
        '        "VLLM_PORT",\n        "VLLM_RAY_BUNDLE_INDICES",\n',
    )
    source.write_text(upstream)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))

    patches._patch_vllm_compile_factors_ignore_bundle_indices(_LOGGER)

    assert source.read_text() == upstream


def test_bundle_indices_patch_is_skipped_without_persistent_caches(monkeypatch):
    def _unexpected(_relative):
        raise AssertionError("must not touch vLLM without NRL_CACHE_SYNC")

    monkeypatch.setattr(patches, "_get_vllm_file", _unexpected)

    patches._patch_vllm_compile_factors_ignore_bundle_indices(_LOGGER)


def test_bundle_indices_patch_fails_loudly_on_unknown_source(
    monkeypatch, tmp_path, seeding_enabled
):
    source = tmp_path / "envs.py"
    source.write_text("def compile_factors():\n    return {}\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(source))

    with pytest.raises(RuntimeError, match="compile-factors patch does not match"):
        patches._patch_vllm_compile_factors_ignore_bundle_indices(_LOGGER)
