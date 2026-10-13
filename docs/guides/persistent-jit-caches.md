# Persistent JIT Compile Caches

vLLM, TorchInductor, Triton, FlashInfer, FlashAttention-4 (CuTeDSL) and the CUDA driver compile kernels at run time and cache them on disk. By default those caches live in the container (`~/.cache`, `/tmp`) and disappear with the job, so every job compiles everything again: torch.compile graphs, autotuning and kernel warmup for each generation engine, and the trainer's Triton and Inductor kernels at the first training step. With a large model this adds minutes to every startup. For example, for a large MoE model served by 10 TP8 policy engines, vLLM engine-core initialization took 177 s per engine from a cold cache and 40–55 s from a warm one (measured with vLLM 0.25.1).

Persistent JIT caches keep these caches on a shared filesystem between jobs, while the compilers read and write a node-local copy. The feature is off unless a launcher sets the variables below; nothing changes for runs that do not.

## How It Works

Caches are kept per worker type, because the vLLM engines and the Megatron trainer compile different kernels, and each node only needs the types it runs:

| Type | Used by | Copied onto a node |
|---|---|---|
| `policy` | vLLM generation engines | by the NeMo RL vLLM actor, in the background when it starts; the engine waits for the copy right before it compiles |
| `megatron` | Megatron policy workers | by the Megatron worker, in the background when it starts (its kernels are first compiled at the first logprob or training step) |
| `<POOL>_CACHE_DIR` | external vLLM pools (`tools/external_gym_vllm`) | by every node of the pool's replicas, before Ray starts |

A Ray node only learns which workers it hosts when the driver places actors on it, so the launcher cannot copy the right cache before `ray start`. Each worker type therefore copies its own:

```text
${NRL_COMPILE_CACHE_ROOT}/<type>/      persistent, on a shared filesystem
        |  cache_sync.sh seed-once: once per node, by the first worker of that type
        v
${NRL_LOCAL_COMPILE_CACHE}/<type>/     node-local; the compilers read and write here
        |  cache_sync.sh sidecar-types: new files only, periodically
        v
${NRL_COMPILE_CACHE_ROOT}/<type>/
```

The policy copy runs off the driver's serial worker-creation path. The NeMo RL vLLM actor starts it in the background; the engine's GPU workers wait for it immediately before they build the model and compile. With vLLM's `RayExecutorV2`, the default for engines with more than one GPU, that wait runs inside each Ray GPU worker process, through a vLLM source patch that calls `nemo_rl/models/generation/vllm/worker_init_hook.py`, so the copy also overlaps the engine's own startup. Engines that compile in the actor's process wait in the actor before the engine is created.

`tools/external_gym_vllm/cache_sync.sh` does the copying and the write-back. `nemo_rl/utils/compile_cache.py` is the worker side.

## Enable It for NeMo RL Workers

Set these variables in the driver's environment; NeMo RL forwards the driver's environment to every Ray actor.

| Variable | Purpose |
|---|---|
| `NRL_CACHE_SYNC` | Path to `tools/external_gym_vllm/cache_sync.sh`, readable at the same path on every node. |
| `NRL_COMPILE_CACHE_ROOT` | Persistent root on the shared filesystem; one subdirectory per type. |
| `NRL_LOCAL_COMPILE_CACHE` | Node-local root, laid out like the persistent one. |
| `NRL_VLLM_<VAR>` | Sets `<VAR>` for the vLLM generation workers only. |
| `NRL_MEGATRON_<VAR>` | Sets `<VAR>` for the Megatron policy workers only. |
| `VLLM_CACHE_ROOT` | vLLM's torch.compile cache and FlashInfer autotune results. Point it into the `policy` type. |
| `NRL_VLLM_CACHE_ROOT_PER_ENGINE` | Default `1` appends `_<engine seed>` to `VLLM_CACHE_ROOT` for each engine, as before. `0` keeps the path unchanged, so it is the same in every job. Use `0` only when no two engines share a node. |
| `CACHE_FORCE_FRESH` | `1` starts every type from an empty cache, for cold-start measurements; what the run compiles is still written back. |
| `CACHE_SYNC_MIN_AGE_MIN` | Minutes a file must stay unchanged before it is written back (default `1`). |

`<VAR>` is one of `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `CUDA_CACHE_PATH`, `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR` and `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED`. FlashAttention-4 keeps no disk cache unless `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`. Setting only some of `NRL_CACHE_SYNC`, `NRL_COMPILE_CACHE_ROOT` and `NRL_LOCAL_COMPILE_CACHE` fails at startup.

The write-back runs on every node, started before `ray start`. With `ray.sub`, use `SETUP_COMMAND`, which runs on every node before Ray starts (and again if Ray start is retried, hence the PID file). `ray.sub` writes it to `<log dir>/setup_command.sh` and creates `<log dir>/ENDED` when the job ends, which triggers the final write-back pass:

```bash
CACHE_SYNC=/path/to/RL/tools/external_gym_vllm/cache_sync.sh
PERSISTENT=/shared/compile_cache   # on the shared filesystem
LOCAL=/tmp/nemo_rl_compile_cache   # node-local

read -r -d '' SETUP_COMMAND <<EOF || true
rm -rf ${LOCAL}
mkdir -p ${LOCAL}
log_dir="\$(dirname "\$0")"
pid_file="\${log_dir}/cache_sync_\$(hostname -s).pid"
if ! kill -0 "\$(cat "\${pid_file}" 2>/dev/null)" 2>/dev/null; then
  bash ${CACHE_SYNC} sidecar-types ${LOCAL} ${PERSISTENT} 60 "\${log_dir}/ENDED" policy megatron \
    >> "\${log_dir}/cache_sync_\$(hostname -s).log" 2>&1 &
  echo \$! > "\${pid_file}"
fi
EOF
export SETUP_COMMAND

COMMAND="NRL_CACHE_SYNC=${CACHE_SYNC} \
NRL_COMPILE_CACHE_ROOT=${PERSISTENT} \
NRL_LOCAL_COMPILE_CACHE=${LOCAL} \
VLLM_CACHE_ROOT=${LOCAL}/policy/vllm \
NRL_VLLM_TORCHINDUCTOR_CACHE_DIR=${LOCAL}/policy/inductor \
NRL_VLLM_TRITON_CACHE_DIR=${LOCAL}/policy/triton \
NRL_VLLM_FLASHINFER_WORKSPACE_BASE=${LOCAL}/policy/flashinfer \
NRL_VLLM_CUDA_CACHE_PATH=${LOCAL}/policy/cuda \
NRL_VLLM_FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
NRL_VLLM_FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${LOCAL}/policy/fa4_cute \
NRL_MEGATRON_TORCHINDUCTOR_CACHE_DIR=${LOCAL}/megatron/inductor \
NRL_MEGATRON_TRITON_CACHE_DIR=${LOCAL}/megatron/triton \
NRL_MEGATRON_FLASHINFER_WORKSPACE_BASE=${LOCAL}/megatron/flashinfer \
NRL_MEGATRON_CUDA_CACHE_PATH=${LOCAL}/megatron/cuda \
uv run examples/run_grpo.py --config <config.yaml>"
```

Add `NRL_VLLM_CACHE_ROOT_PER_ENGINE=0` when every node runs at most one generation engine. Without it each engine reads and writes `${LOCAL}/policy/vllm_<seed>` with its own engine seed, which still persists, but an engine then only reuses what the engine with the same seed compiled in earlier jobs.

Clearing the node-local root at job start matters: `seed-once` skips a type whose `<type>.seeded` marker survives from an earlier job on the same node, and that node would then miss what other jobs published since.

## External vLLM Pools

Pools started by `tools/external_gym_vllm/run_in_allocation.sh` get their own cache with `register_external_vllm_pool ... --cache-dir <dir>` (the `<POOL>_CACHE_DIR` variable). Every node of every replica copies it before Ray starts, the vLLM workers inherit the paths, and a sidecar writes new entries back. `CACHE_FORCE_FRESH`, `CACHE_SYNC_INTERVAL` and `CACHE_SYNC_MIN_AGE_MIN` apply to pools too. See [Persistent compile caches](https://github.com/NVIDIA-NeMo/RL/blob/main/tools/external_gym_vllm/README.md#persistent-compile-caches) in the tool's README.

## Write-Back Rules

`cache_sync.sh writeback`, which `sidecar` and `sidecar-types` repeat, publishes a file only if all of these hold:

- it does not exist in the persistent copy yet: existing files are never replaced;
- it has not changed for `CACHE_SYNC_MIN_AGE_MIN` minutes (default 1), on every pass including the final one at the end of a job: because existing files are never replaced, publishing a file that is still being written would make the truncated entry permanent, so files from a job's last minute are compiled again and published by a later run;
- it is newer than the node's seed (files copied in by `seed-once` already exist in the persistent copy);
- for `sidecar-types`, its type was seeded on this node, so a node only publishes the types it runs.

One writer at a time updates a persistent directory, holding `<dir>.lock` next to it; a writer that finds the lock taken skips that pass, and a lock older than 10 minutes is removed as stale. rsync writes each file under a temporary name and renames it, so a job that seeds while another writes back never copies a partial file. Seeding also copies with rsync, which leaves out files it could not copy instead of leaving them truncated, and moves entries into the node-local directory without replacing files a compiler already wrote there.

Merging entries from many nodes and jobs into one directory is safe because these caches are content- or hash-addressed: the key covers the model, the engine settings and the software versions.

## Stable vLLM Cache Keys

vLLM hashes most `VLLM_*` environment variables into its torch.compile and FlashInfer autotune cache keys. NeMo RL sets `VLLM_RAY_BUNDLE_INDICES` per engine to place it on its GPUs, so each engine had its own keys: a cache written by one engine never hit for the others, and every run compiled and tuned once per engine. When `NRL_CACHE_SYNC` is set, NeMo RL patches vLLM to leave this variable out of the key (`_patch_vllm_compile_factors_ignore_bundle_indices` in `nemo_rl/models/generation/vllm/patches.py`), so all engines of a policy share one key. The patch is applied only with persistent caches on, so other runs keep their existing cache keys, and it fails at engine start if the vLLM source no longer matches.

## Caveats

- **No pruning.** Persistent directories only grow. Entries for older vLLM, PyTorch, Triton or FlashInfer versions, other models or other engine settings stay until you delete them. Start a new `NRL_COMPILE_CACHE_ROOT` (or `--cache-dir`) when you change the container, or remove old ones yourself. An interrupted write-back can leave rsync temporary files (`.<name>.<random>`), which are never read.
- **Shared filesystem.** `NRL_COMPILE_CACHE_ROOT`, the pool cache directories and `NRL_CACHE_SYNC` must be visible at the same path on every node. `NRL_LOCAL_COMPILE_CACHE` must be node-local: the once-per-node seeding takes a `flock` lock there.
- **Tools.** Containers that seed or write back need `rsync` and `flock`. A missing tool fails the command loudly.
- **Megatron copies in the background** and nothing waits for it, on the assumption that the trainer compiles nothing before its first logprob or training step. A kernel compiled while the copy is running is compiled from scratch.
- **Only the `policy` and `megatron` types are wired into NeMo RL.** Other trainer backends keep their default caches.
- **Launcher-side seeding.** Some example launchers, such as `examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh`, seed node-local caches themselves: their `SETUP_COMMAND` extracts prebuilt tarballs onto every node before Ray starts. That approach needs no NeMo RL support, but copies every cache to every node whatever it runs, and builds and refreshes the tarballs outside the job. It and this feature are independent; a launcher should use one of them per cache directory.
