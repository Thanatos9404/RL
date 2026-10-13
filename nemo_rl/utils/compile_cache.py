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
"""Per-type persistent JIT compile caches, copied onto a node by the workers that use them.

A Ray node only learns whether it hosts a generation engine or the trainer when
the driver places actors on it, so a launcher cannot copy the right cache onto it
before ``ray start``. Instead each worker type copies its own cache when its
actor starts:

    ${NRL_COMPILE_CACHE_ROOT}/<type>     persistent, on a shared filesystem
        -> ${NRL_LOCAL_COMPILE_CACHE}/<type>    node-local, used by the compilers

The copy runs once per node (``cache_sync.sh seed-once`` holds a node-local lock
and moves entries into place without replacing any). Publishing new entries back
is the launcher's job (``cache_sync.sh sidecar-types``, started on every node
before ``ray start``). Each worker type points its JIT caches at its own
directory through ``NRL_<PREFIX>_<VAR>`` variables (see ``env_overrides``).

Everything here is a no-op unless the launcher sets the variables. See
docs/guides/persistent-jit-caches.md.
"""

import os
import subprocess

# Path to tools/external_gym_vllm/cache_sync.sh. Together with the two roots
# below it turns per-type seeding on; patches.py reads the same variable.
CACHE_SYNC_ENV_VAR = "NRL_CACHE_SYNC"
PERSISTENT_ROOT_ENV_VAR = "NRL_COMPILE_CACHE_ROOT"
LOCAL_ROOT_ENV_VAR = "NRL_LOCAL_COMPILE_CACHE"
# "1": start from an empty cache (cold-start measurement) but still write back.
FORCE_FRESH_ENV_VAR = "CACHE_FORCE_FRESH"
# Set on vLLM generation actors when seeding is on; vLLM's Ray GPU workers wait
# for this type's copy before they compile (worker_init_hook.py). patches.py
# embeds the same name in the vLLM source it patches.
VLLM_CACHE_TYPE_ENV_VAR = "NRL_VLLM_COMPILE_CACHE_TYPE"

POLICY_CACHE_TYPE = "policy"
MEGATRON_CACHE_TYPE = "megatron"

# JIT cache variables a worker type can redirect with NRL_<PREFIX>_<VAR>.
CACHE_VARS = (
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "CUDA_CACHE_PATH",
    # FlashAttention-4 (CuTeDSL) kernels, compiled at engine warmup and CUDA-graph
    # capture. FA4 keeps no disk cache unless ..._ENABLED=1.
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED",
)


def env_overrides(prefix: str) -> dict[str, str]:
    """Map the set ``NRL_<prefix>_<VAR>`` variables to ``<VAR>`` for a worker's env.

    Args:
        prefix: Worker type, e.g. ``"VLLM"`` or ``"MEGATRON"``.

    Returns:
        ``{<VAR>: value}`` for every variable in ``CACHE_VARS`` whose
        ``NRL_<prefix>_<VAR>`` is set and non-empty; empty when none is.
    """
    overrides = {}
    for var in CACHE_VARS:
        value = os.environ.get(f"NRL_{prefix}_{var}")
        if value:
            overrides[var] = value
    return overrides


def enabled() -> bool:
    """Return whether per-type seeding is configured.

    Raises:
        RuntimeError: If only some of ``NRL_CACHE_SYNC``,
            ``NRL_COMPILE_CACHE_ROOT`` and ``NRL_LOCAL_COMPILE_CACHE`` are set,
            or ``NRL_CACHE_SYNC`` is not a file.
    """
    names = (CACHE_SYNC_ENV_VAR, PERSISTENT_ROOT_ENV_VAR, LOCAL_ROOT_ENV_VAR)
    values = {name: os.environ.get(name) for name in names}
    if not any(values.values()):
        return False
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            f"Persistent compile caches need {', '.join(names)}; "
            f"{', '.join(missing)} not set."
        )
    script = os.environ[CACHE_SYNC_ENV_VAR]
    if not os.path.isfile(script):
        raise RuntimeError(
            f"{CACHE_SYNC_ENV_VAR}={script} is not a file; point it at "
            "tools/external_gym_vllm/cache_sync.sh."
        )
    return True


def seed(cache_type: str, *, background: bool = False) -> None:
    """Copy this type's persistent cache onto the node, once per node.

    A blocking call returns once the node's copy is in place. The copy holds a
    node-local lock, so this also waits for a copy started by another process
    (for instance a background one), or makes the copy itself if none did.
    With ``CACHE_FORCE_FRESH=1`` nothing is copied, but the type is still
    marked seeded so that what this run compiles is written back.

    Args:
        cache_type: Subdirectory of both roots, e.g. ``"policy"``.
        background: Return at once and copy in a detached process. The user of
            the cache then calls ``seed`` again, blocking, right before it
            compiles.

    Raises:
        subprocess.CalledProcessError: If a blocking ``cache_sync.sh`` call
            fails (the copy itself is best effort and does not fail it).
    """
    if not enabled():
        return
    script = os.environ[CACHE_SYNC_ENV_VAR]
    local = os.path.join(os.environ[LOCAL_ROOT_ENV_VAR], cache_type)
    if os.environ.get(FORCE_FRESH_ENV_VAR) == "1":
        cmd = ["bash", script, "mark-fresh", local]
    else:
        persistent = os.path.join(os.environ[PERSISTENT_ROOT_ENV_VAR], cache_type)
        cmd = ["bash", script, "seed-once", persistent, local]
    if background:
        subprocess.Popen(cmd, start_new_session=True)
    else:
        subprocess.run(cmd, check=True)
