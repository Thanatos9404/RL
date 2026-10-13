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

"""Per-type persistent compile caches (nemo_rl/utils/compile_cache.py)."""

import shutil
import subprocess
from pathlib import Path

import pytest

from nemo_rl.models.generation.vllm import worker_init_hook
from nemo_rl.utils import compile_cache

REPO_ROOT = Path(__file__).parents[3]
CACHE_SYNC = REPO_ROOT / "tools/external_gym_vllm/cache_sync.sh"

needs_cache_sync_tools = pytest.mark.skipif(
    shutil.which("rsync") is None or shutil.which("flock") is None,
    reason="cache_sync.sh needs rsync and flock",
)


@pytest.fixture(autouse=True)
def _no_ambient_cache_env(monkeypatch):
    for name in (
        compile_cache.CACHE_SYNC_ENV_VAR,
        compile_cache.PERSISTENT_ROOT_ENV_VAR,
        compile_cache.LOCAL_ROOT_ENV_VAR,
        compile_cache.FORCE_FRESH_ENV_VAR,
        compile_cache.VLLM_CACHE_TYPE_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
    for prefix in ("VLLM", "MEGATRON"):
        for var in compile_cache.CACHE_VARS:
            monkeypatch.delenv(f"NRL_{prefix}_{var}", raising=False)


@pytest.fixture
def configured(monkeypatch, tmp_path):
    """Enable seeding with a persistent policy cache holding one entry."""
    persistent, local = tmp_path / "persistent", tmp_path / "local"
    (persistent / "policy").mkdir(parents=True)
    (persistent / "policy/entry.bin").write_text("compiled by an earlier job")
    monkeypatch.setenv(compile_cache.CACHE_SYNC_ENV_VAR, str(CACHE_SYNC))
    monkeypatch.setenv(compile_cache.PERSISTENT_ROOT_ENV_VAR, str(persistent))
    monkeypatch.setenv(compile_cache.LOCAL_ROOT_ENV_VAR, str(local))
    return local


def _forbid_subprocesses(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError(f"unexpected subprocess: {args}")

    monkeypatch.setattr(compile_cache.subprocess, "run", _fail)
    monkeypatch.setattr(compile_cache.subprocess, "Popen", _fail)


def test_env_overrides_maps_only_the_prefix_variables_that_are_set(monkeypatch):
    monkeypatch.setenv("NRL_VLLM_TRITON_CACHE_DIR", "/local/policy/triton")
    monkeypatch.setenv("NRL_VLLM_FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    monkeypatch.setenv("NRL_VLLM_CUDA_CACHE_PATH", "")
    monkeypatch.setenv("NRL_MEGATRON_TRITON_CACHE_DIR", "/local/megatron/triton")

    assert compile_cache.env_overrides("VLLM") == {
        "TRITON_CACHE_DIR": "/local/policy/triton",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
    }
    assert compile_cache.env_overrides("MEGATRON") == {
        "TRITON_CACHE_DIR": "/local/megatron/triton"
    }


def test_env_overrides_is_empty_when_unset():
    assert compile_cache.env_overrides("VLLM") == {}
    assert compile_cache.env_overrides("MEGATRON") == {}


@pytest.mark.parametrize("background", [False, True])
def test_seed_is_a_noop_when_unconfigured(monkeypatch, background):
    _forbid_subprocesses(monkeypatch)

    assert compile_cache.enabled() is False
    compile_cache.seed(compile_cache.POLICY_CACHE_TYPE, background=background)


def test_worker_init_hook_is_a_noop_without_a_cache_type(monkeypatch, configured):
    _forbid_subprocesses(monkeypatch)

    worker_init_hook.on_ray_worker_init()


def test_partial_configuration_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv(compile_cache.CACHE_SYNC_ENV_VAR, str(CACHE_SYNC))
    monkeypatch.setenv(compile_cache.LOCAL_ROOT_ENV_VAR, str(tmp_path))

    with pytest.raises(RuntimeError, match="NRL_COMPILE_CACHE_ROOT not set"):
        compile_cache.seed(compile_cache.POLICY_CACHE_TYPE)


def test_missing_cache_sync_script_fails_loudly(monkeypatch, configured, tmp_path):
    monkeypatch.setenv(compile_cache.CACHE_SYNC_ENV_VAR, str(tmp_path / "missing.sh"))

    with pytest.raises(RuntimeError, match="is not a file"):
        compile_cache.enabled()


@needs_cache_sync_tools
def test_seed_copies_the_type_onto_the_node_once(configured, tmp_path):
    compile_cache.seed(compile_cache.POLICY_CACHE_TYPE)

    assert (configured / "policy/entry.bin").exists()
    assert (configured / "policy.seeded").exists()

    # Seeded once per node: a later call does not copy newer entries.
    (tmp_path / "persistent/policy/later.bin").write_text("later")
    compile_cache.seed(compile_cache.POLICY_CACHE_TYPE)
    assert not (configured / "policy/later.bin").exists()


@needs_cache_sync_tools
def test_force_fresh_marks_the_type_seeded_without_copying(monkeypatch, configured):
    monkeypatch.setenv(compile_cache.FORCE_FRESH_ENV_VAR, "1")

    compile_cache.seed(compile_cache.POLICY_CACHE_TYPE)

    assert not (configured / "policy/entry.bin").exists()
    assert (configured / "policy.seeded").exists()


@needs_cache_sync_tools
def test_blocking_seed_returns_after_a_background_copy(configured):
    compile_cache.seed(compile_cache.POLICY_CACHE_TYPE, background=True)
    compile_cache.seed(compile_cache.POLICY_CACHE_TYPE)

    assert (configured / "policy/entry.bin").exists()
    assert (configured / "policy.seeded").exists()


@needs_cache_sync_tools
def test_worker_init_hook_waits_for_the_engine_cache_type(monkeypatch, configured):
    monkeypatch.setenv(
        compile_cache.VLLM_CACHE_TYPE_ENV_VAR, compile_cache.POLICY_CACHE_TYPE
    )

    worker_init_hook.on_ray_worker_init()

    assert (configured / "policy/entry.bin").exists()


@needs_cache_sync_tools
def test_blocking_seed_surfaces_cache_sync_failures(monkeypatch, configured, tmp_path):
    broken = tmp_path / "broken.sh"
    broken.write_text("exit 3\n")
    monkeypatch.setenv(compile_cache.CACHE_SYNC_ENV_VAR, str(broken))

    with pytest.raises(subprocess.CalledProcessError):
        compile_cache.seed(compile_cache.POLICY_CACHE_TYPE)
