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
"""Runs in every vLLM ``RayWorkerProc`` (one per GPU) before the worker is built.

``RayWorkerProc.initialize_worker`` calls ``on_ray_worker_init`` through a vLLM
source patch (``patches._patch_vllm_ray_worker_proc_init_hook``), after the
driver's environment has been applied and before the worker loads or compiles
anything. The patched code imports this module only when
``NRL_VLLM_COMPILE_CACHE_TYPE`` is set.
"""

import os

from nemo_rl.utils import compile_cache


def on_ray_worker_init() -> None:
    """Wait until this node's copy of the engine's compile cache is in place.

    The NeMo RL vLLM actor starts that copy in the background when it is
    created, so the copy overlaps the rest of setup instead of running on the
    driver's serial worker-creation path. ``compile_cache.seed`` waits on the
    node-local lock the copy holds, or makes the copy itself if no actor on this
    node started it.
    """
    cache_type = os.environ.get(compile_cache.VLLM_CACHE_TYPE_ENV_VAR)
    if cache_type:
        compile_cache.seed(cache_type)
