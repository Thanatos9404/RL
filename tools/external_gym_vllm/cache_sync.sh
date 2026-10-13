#!/usr/bin/env bash
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

# Copy JIT compile caches from a persistent directory on a shared filesystem to
# node-local storage, and publish new entries back. Used by run_in_allocation.sh
# (external vLLM pools), nemo_rl/utils/compile_cache.py (NeMo RL workers) and
# launcher setup commands (write-back on Ray nodes). See
# docs/guides/persistent-jit-caches.md.
#
#   cache_sync.sh seed          <persistent_dir> <local_dir>
#   cache_sync.sh seed-once     <persistent_dir> <local_dir>
#   cache_sync.sh mark-fresh    <local_dir>
#   cache_sync.sh writeback     <local_dir> <persistent_dir>
#   cache_sync.sh sidecar       <local_dir> <persistent_dir> [interval_s] [stop_file]
#   cache_sync.sh sidecar-types <local_root> <persistent_root> <interval_s> <stop_file> <type>...
#
# seed copies <persistent_dir> into <local_dir> (no lock, no marker), for a
# launcher that seeds a node before anything on it compiles.
# seed-once copies <persistent_dir> onto the node once: the first caller takes a
# node-local lock (<local_dir>.seedlock), copies into a temporary directory and
# moves entries into <local_dir> without replacing anything already there (a
# compiler may have started writing), then creates <local_dir>.seeded. Later
# callers wait on the lock and return, so a blocking call doubles as "wait for
# the copy someone else started".
# mark-fresh is seed-once without the copy (CACHE_FORCE_FRESH=1, a cold-start
# measurement): <local_dir> starts empty but is marked seeded, so what the run
# compiles is still written back.
#
# writeback publishes files of <local_dir> that are newer than its .seeded
# marker (if any), have not been modified for CACHE_SYNC_MIN_AGE_MIN minutes
# (default 1) and do not exist in <persistent_dir> yet. It never replaces a
# persistent file, so publishing a file that is still being written would make
# the truncated entry permanent; the age rule prevents that, including on the
# final pass at the end of a job (files from a job's last minute are compiled
# again and published by a later run). rsync writes each file under a temporary
# name and renames it, so readers never see partial files. Writers serialize on
# a lock directory next to <persistent_dir> (<persistent_dir>.lock); a writer
# that finds it taken skips the pass, and a lock older than 10 minutes is
# treated as stale. The cached artifacts (vLLM torch.compile, TorchInductor,
# Triton, FlashInfer, FlashAttention-4) are content- or hash-addressed, so
# merging entries written by several nodes and jobs is safe.
# sidecar repeats writeback every interval_s seconds (default 60) and once more
# when stop_file appears. sidecar-types does the same for <local_root>/<type>
# -> <persistent_root>/<type>, for each listed type once it has been seeded on
# this node, so a node only publishes the types it actually runs.
set -uo pipefail

log() { echo "[cache_sync $(hostname -s)] $*"; }

require_tools() {
  local tool
  for tool in "$@"; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
      log "ERROR: '${tool}' is required but not installed"
      exit 1
    fi
  done
}

writeback() {
  local src="$1" dst="$2" lock="$2.lock" list out err n rc
  local -a newer=()
  [[ -d "${src}" ]] || return 0
  list="$(mktemp)"
  # Entries copied in by seed-once keep their mtimes and already exist in the
  # persistent copy; only files created after the seed are candidates.
  [[ -e "${src}.seeded" ]] && newer=(-newer "${src}.seeded")
  (cd "${src}" && find . -type f -mmin "+${CACHE_SYNC_MIN_AGE_MIN:-1}" "${newer[@]}" -print) >"${list}"
  if [[ ! -s "${list}" ]]; then
    rm -f "${list}"
    return 0
  fi
  mkdir -p "$(dirname "${dst}")"
  if ! mkdir "${lock}" 2>/dev/null; then
    if [[ -n "$(find "${lock}" -maxdepth 0 -mmin +10 2>/dev/null)" ]]; then
      log "removing stale lock ${lock}"
      rmdir "${lock}" 2>/dev/null || true
    fi
    rm -f "${list}"
    return 0
  fi
  mkdir -p "${dst}"
  out="$(mktemp)"
  err="$(mktemp)"
  rsync -a --ignore-existing --files-from="${list}" --out-format='%n' "${src}/" "${dst}/" >"${out}" 2>"${err}"
  rc=$?
  rmdir "${lock}" 2>/dev/null || true
  # 24: a listed file vanished before it was copied (a compiler replaced it).
  if (( rc != 0 && rc != 24 )); then
    log "WARNING: write-back ${src} -> ${dst} failed (rsync exit ${rc}): $(tail -n 1 "${err}")"
  fi
  n="$(grep -vc '/$' "${out}")"
  (( n > 0 )) && log "wrote back ${n} new file(s) ${src} -> ${dst}"
  rm -f "${list}" "${out}" "${err}"
  return 0
}

sidecar() {
  local src="$1" dst="$2" interval="${3:-60}" stop="${4:-}" last
  while true; do
    sleep "${interval}"
    last=0
    [[ -n "${stop}" && -e "${stop}" ]] && last=1
    # The pass after stop_file appeared is the last one; same age rule.
    writeback "${src}" "${dst}"
    (( last )) && break
  done
}

# Copy $1 into $2 with rsync, not cp: a file that fails to copy (or vanishes,
# such as another writer's temporary file) is left out rather than left
# truncated, so whatever arrived is complete and can be used.
copy_complete_files() {
  local src="$1" dst="$2" rc
  rsync -a "${src}/" "${dst}/" 2>/dev/null
  rc=$?
  if (( rc != 0 && rc != 24 )); then
    log "WARNING: copy ${src} -> ${dst} incomplete (rsync exit ${rc}); missing entries are compiled again"
  fi
}

seed() {
  local src="$1" dst="$2"
  mkdir -p "${dst}"
  if [[ -d "${src}" ]] && [[ -n "$(ls -A "${src}" 2>/dev/null)" ]]; then
    copy_complete_files "${src}" "${dst}"
    log "seeded ${dst} from ${src} ($(du -sh "${dst}" 2>/dev/null | cut -f1))"
  else
    log "no persistent cache at ${src} yet; ${dst} starts cold"
  fi
}

# Move every entry of $1 into $2 that $2 does not have yet; recurse into
# directories both have. Each mv is a same-filesystem rename, so a reader of $2
# sees a complete file or nothing.
merge_into() {
  local src="$1" dst="$2" entry name
  mkdir -p "${dst}"
  for entry in "${src}"/* "${src}"/.[!.]*; do
    [[ -e "${entry}" || -L "${entry}" ]] || continue
    name="$(basename "${entry}")"
    if [[ ! -e "${dst}/${name}" ]]; then
      mv -T "${entry}" "${dst}/${name}" 2>/dev/null || true
    elif [[ -d "${entry}" && -d "${dst}/${name}" ]]; then
      merge_into "${entry}" "${dst}/${name}"
    fi
  done
}

seed_once() {
  local src="$1" dst="$2" tmp t0
  mkdir -p "$(dirname "${dst}")"
  exec 9>"${dst}.seedlock"
  flock 9
  if [[ -e "${dst}.seeded" ]]; then
    exec 9>&-
    return 0
  fi
  t0=$(date +%s.%N)
  if [[ -d "${src}" ]] && [[ -n "$(ls -A "${src}" 2>/dev/null)" ]]; then
    tmp="$(dirname "${dst}")/.seed-$(basename "${dst}").$$"
    rm -rf "${tmp}"
    mkdir -p "${tmp}"
    copy_complete_files "${src}" "${tmp}"
    merge_into "${tmp}" "${dst}"
    rm -rf "${tmp}"
    log "seeded ${dst} from ${src} ($(du -sh "${dst}" 2>/dev/null | cut -f1), $(awk -v a="${t0}" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}') s)"
  else
    mkdir -p "${dst}"
    log "no persistent cache at ${src} yet; ${dst} starts cold"
  fi
  touch "${dst}.seeded"
  exec 9>&-
}

mark_fresh() {
  local dst="$1"
  mkdir -p "$(dirname "${dst}")"
  exec 9>"${dst}.seedlock"
  flock 9
  if [[ ! -e "${dst}.seeded" ]]; then
    mkdir -p "${dst}"
    touch "${dst}.seeded"
    log "fresh ${dst} (CACHE_FORCE_FRESH=1: nothing copied; new entries are written back)"
  fi
  exec 9>&-
}

sidecar_types() {
  local root="$1" proot="$2" interval="$3" stop="$4" cache_type last
  shift 4
  while true; do
    sleep "${interval}"
    last=0
    [[ -n "${stop}" && -e "${stop}" ]] && last=1
    for cache_type in "$@"; do
      [[ -e "${root}/${cache_type}.seeded" ]] || continue
      writeback "${root}/${cache_type}" "${proot}/${cache_type}"
    done
    (( last )) && break
  done
}

usage() {
  echo "usage: cache_sync.sh {seed|seed-once|mark-fresh|writeback|sidecar|sidecar-types} ..." >&2
  exit 2
}

cmd="${1:-}"
shift || true
case "${cmd}" in
  seed)
    (( $# == 2 )) || usage
    require_tools rsync
    seed "$@"
    ;;
  seed-once)
    (( $# == 2 )) || usage
    require_tools flock rsync
    seed_once "$@"
    ;;
  mark-fresh)
    (( $# == 1 )) || usage
    require_tools flock
    mark_fresh "$@"
    ;;
  writeback)
    (( $# == 2 )) || usage
    require_tools rsync
    writeback "$@"
    ;;
  sidecar)
    (( $# >= 2 && $# <= 4 )) || usage
    require_tools rsync
    sidecar "$@"
    ;;
  sidecar-types)
    (( $# >= 5 )) || usage
    require_tools rsync
    sidecar_types "$@"
    ;;
  *) usage ;;
esac
