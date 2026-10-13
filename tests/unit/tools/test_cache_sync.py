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

"""tools/external_gym_vllm/cache_sync.sh: seeding and write-back of compile caches.

Write-back never replaces a file in the persistent cache, so publishing a file
that a compiler is still writing would make a truncated entry permanent for
every later job. These tests pin the rules that prevent that (the age rule, on
every pass including the last) and the node-local seeding lock.
"""

import fcntl
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
CACHE_SYNC = REPO_ROOT / "tools/external_gym_vllm/cache_sync.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("rsync") is None or shutil.which("flock") is None,
    reason="cache_sync.sh needs rsync and flock",
)


def _cache_sync(*args, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(CACHE_SYNC), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=60,
        **kwargs,
    )


def _write(path: Path, content: str, *, age_minutes: float = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if age_minutes:
        stamp = time.time() - age_minutes * 60
        os.utime(path, (stamp, stamp))
    return path


def test_writeback_publishes_only_settled_files_and_never_overwrites(tmp_path):
    local, persistent = tmp_path / "local", tmp_path / "persistent"
    _write(local / "settled.bin", "settled", age_minutes=10)
    _write(local / "nested/settled.bin", "nested", age_minutes=10)
    _write(local / "being-written.bin", "partial")
    _write(local / "existing.bin", "local copy", age_minutes=10)
    _write(persistent / "existing.bin", "persistent copy")

    result = _cache_sync("writeback", local, persistent)

    assert result.returncode == 0, result.stderr
    assert (persistent / "settled.bin").read_text() == "settled"
    assert (persistent / "nested/settled.bin").read_text() == "nested"
    assert not (persistent / "being-written.bin").exists()
    assert (persistent / "existing.bin").read_text() == "persistent copy"
    assert not (tmp_path / "persistent.lock").exists()
    assert "wrote back 2 new file(s)" in result.stdout


def test_writeback_honors_the_configured_minimum_age(tmp_path):
    local, persistent = tmp_path / "local", tmp_path / "persistent"
    _write(local / "three-minutes-old.bin", "x", age_minutes=3)

    _cache_sync(
        "writeback",
        local,
        persistent,
        env={**os.environ, "CACHE_SYNC_MIN_AGE_MIN": "5"},
    )
    assert not (persistent / "three-minutes-old.bin").exists()

    _cache_sync("writeback", local, persistent)
    assert (persistent / "three-minutes-old.bin").exists()


def test_writeback_skips_files_that_predate_the_seed(tmp_path):
    """Seeded entries already exist in the persistent copy; don't re-check them."""
    local, persistent = tmp_path / "local", tmp_path / "persistent"
    _write(local / "copied-in-by-seed.bin", "old", age_minutes=30)
    _write(tmp_path / "local.seeded", "", age_minutes=20)
    _write(local / "compiled-by-this-job.bin", "new", age_minutes=10)

    _cache_sync("writeback", local, persistent)

    assert (persistent / "compiled-by-this-job.bin").exists()
    assert not (persistent / "copied-in-by-seed.bin").exists()


def test_writeback_waits_for_another_writer_and_breaks_a_stale_lock(tmp_path):
    local, persistent = tmp_path / "local", tmp_path / "persistent"
    lock = tmp_path / "persistent.lock"
    _write(local / "entry.bin", "x", age_minutes=10)
    lock.mkdir()

    _cache_sync("writeback", local, persistent)
    assert not (persistent / "entry.bin").exists()
    assert lock.exists()

    stale = time.time() - 20 * 60
    os.utime(lock, (stale, stale))
    _cache_sync("writeback", local, persistent)
    assert not lock.exists()
    assert not (persistent / "entry.bin").exists()

    _cache_sync("writeback", local, persistent)
    assert (persistent / "entry.bin").exists()


def test_sidecar_final_pass_keeps_the_age_rule(tmp_path):
    """The pass at job end must not publish files a compiler may still write."""
    local, persistent = tmp_path / "local", tmp_path / "persistent"
    stop = _write(tmp_path / "ENDED", "")
    _write(local / "settled.bin", "settled", age_minutes=10)
    _write(local / "being-written.bin", "partial")

    result = _cache_sync("sidecar", local, persistent, 0, stop)

    assert result.returncode == 0, result.stderr
    assert (persistent / "settled.bin").exists()
    assert not (persistent / "being-written.bin").exists()


def test_seed_copies_complete_files_into_the_local_dir(tmp_path):
    persistent, local = tmp_path / "persistent", tmp_path / "local"
    _write(persistent / "dir/entry.bin", "entry")

    result = _cache_sync("seed", persistent, local)

    assert result.returncode == 0, result.stderr
    assert (local / "dir/entry.bin").read_text() == "entry"
    assert not (tmp_path / "local.seeded").exists()


def test_seed_once_copies_once_per_node_without_replacing_local_files(tmp_path):
    persistent, local = tmp_path / "persistent", tmp_path / "local" / "policy"
    _write(persistent / "shared.bin", "persistent")
    _write(persistent / "dir/entry.bin", "entry")
    # A compiler on this node already wrote this entry; the seed must keep it.
    _write(local / "shared.bin", "compiled here")

    first = _cache_sync("seed-once", persistent, local)

    assert first.returncode == 0, first.stderr
    assert (local / "shared.bin").read_text() == "compiled here"
    assert (local / "dir/entry.bin").read_text() == "entry"
    assert (tmp_path / "local" / "policy.seeded").exists()
    assert not list((tmp_path / "local").glob(".seed-*"))

    _write(persistent / "published-later.bin", "later")
    second = _cache_sync("seed-once", persistent, local)

    assert second.returncode == 0, second.stderr
    assert not (local / "published-later.bin").exists()


def test_seed_once_without_a_persistent_cache_starts_cold(tmp_path):
    local = tmp_path / "local" / "policy"

    result = _cache_sync("seed-once", tmp_path / "missing", local)

    assert result.returncode == 0, result.stderr
    assert local.is_dir() and not any(local.iterdir())
    assert (tmp_path / "local" / "policy.seeded").exists()
    assert "starts cold" in result.stdout


def test_seed_once_waits_for_the_copy_holding_the_node_lock(tmp_path):
    """A blocking seed-once is how workers wait for a background copy."""
    persistent, local = tmp_path / "persistent", tmp_path / "local" / "policy"
    _write(persistent / "entry.bin", "entry")
    seeded = tmp_path / "local" / "policy.seeded"
    seed_lock = tmp_path / "local" / "policy.seedlock"
    seed_lock.parent.mkdir(parents=True)

    with open(seed_lock, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        waiter = subprocess.Popen(
            ["bash", str(CACHE_SYNC), "seed-once", str(persistent), str(local)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.5)
            assert waiter.poll() is None
            assert not seeded.exists()
        finally:
            fcntl.flock(held, fcntl.LOCK_UN)
    assert waiter.wait(timeout=30) == 0
    assert seeded.exists()
    assert (local / "entry.bin").exists()


def test_mark_fresh_copies_nothing_but_enables_writeback(tmp_path):
    persistent, local = tmp_path / "persistent", tmp_path / "local" / "policy"
    _write(persistent / "warm.bin", "warm")

    result = _cache_sync("mark-fresh", local)

    assert result.returncode == 0, result.stderr
    assert local.is_dir() and not any(local.iterdir())
    marker = tmp_path / "local" / "policy.seeded"
    assert marker.exists()

    # A file compiled by the cold run (after the marker) is still published.
    stamp = time.time() - 20 * 60
    os.utime(marker, (stamp, stamp))
    _write(local / "cold-compiled.bin", "cold", age_minutes=10)
    _cache_sync("writeback", local, persistent)
    assert (persistent / "cold-compiled.bin").read_text() == "cold"


def test_sidecar_types_writes_back_only_types_seeded_on_this_node(tmp_path):
    root, persistent = tmp_path / "local", tmp_path / "persistent"
    stop = _write(tmp_path / "ENDED", "")
    _write(root / "policy.seeded", "", age_minutes=20)
    _write(root / "policy/kernel.bin", "policy", age_minutes=10)
    _write(root / "policy/being-written.bin", "partial")
    # Never seeded here: this node does not run the trainer.
    _write(root / "megatron/kernel.bin", "megatron", age_minutes=10)

    result = _cache_sync(
        "sidecar-types", root, persistent, 0, stop, "policy", "megatron"
    )

    assert result.returncode == 0, result.stderr
    assert (persistent / "policy/kernel.bin").read_text() == "policy"
    assert not (persistent / "policy/being-written.bin").exists()
    assert not (persistent / "megatron").exists()


def test_missing_rsync_fails_loudly(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "hostname").symlink_to(shutil.which("hostname"))

    result = subprocess.run(
        [shutil.which("bash"), str(CACHE_SYNC), "writeback", "a", "b"],
        env={"PATH": str(bin_dir)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "'rsync' is required" in result.stdout


@pytest.mark.parametrize(
    "args", [(), ("seed",), ("seed-once", "only-one"), ("writeback", "a")]
)
def test_usage_errors_exit_2(args):
    assert _cache_sync(*args).returncode == 2
