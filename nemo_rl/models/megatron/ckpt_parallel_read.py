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
"""Parallel reads for torch_dist (torch.distributed.checkpoint) checkpoint loads.

PyTorch's ``FileSystemReader.read_data`` reads a rank's checkpoint items one at a
time from a single file stream, so each rank has one storage request in flight.
:func:`install` replaces it, process-wide, with :func:`parallel_read_data`, which
reads the same byte ranges with a thread pool (``os.preadv``, ``CHUNK_BYTES`` per
request, in file order, at most ``WINDOW_BYTES`` read ahead) while the calling
thread deserializes and copies the items in file order. Everything after the read
(stream transforms, ``torch.load``, narrowing, ``copy_``, ``commit_tensor``) is
the same code as PyTorch's.

Because this replaces PyTorch's own implementation, :func:`install` only does so
on a torch version in ``[2.11, 2.14)`` whose ``read_data`` source matches the one
this module mirrors, and raises otherwise. Readers that do not read local files
(for example ``FsspecReader``, which inherits ``read_data``) keep PyTorch's
implementation.

This is an opt-in bridge until PyTorch's reader, or megatron-core's choice of
reader, can keep several reads in flight per rank.
"""

import collections
import hashlib
import inspect
import io
import os
import textwrap
import warnings
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import cast

import torch
from torch.distributed.checkpoint.filesystem import (
    FileSystem,
    FileSystemReader,
    narrow_tensor_by_index,
)
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
)

# Bytes per read request.
CHUNK_BYTES = 16 << 20
# Bytes read ahead of the item being deserialized (an item larger than this is
# still read whole, in parallel chunks).
WINDOW_BYTES = 1 << 30

# torch versions [min, max) whose FileSystemReader.read_data this module mirrors.
_MIN_TORCH_VERSION = (2, 11)
_MAX_TORCH_VERSION = (2, 14)
# sha256 of textwrap.dedent(inspect.getsource(FileSystemReader.read_data)); the
# same in torch 2.11.0 and 2.13.0. A reader with other source raises in install().
_READ_DATA_SOURCE_SHA256 = (
    "ae6cd4ef21fa688cc16401091adbc2ea3a7e112905de52f4f46933f770b95168"
)

# Thread count of the installed reader; None while PyTorch's reader is in place.
G_INSTALLED_NUM_THREADS: int | None = None


def check_torch_supported() -> None:
    """Raise unless this torch's ``FileSystemReader.read_data`` is the one mirrored here.

    Raises:
        RuntimeError: If the torch version is outside ``[2.11, 2.14)`` or the source
            of ``FileSystemReader.read_data`` differs from the mirrored version.
    """
    major, minor = (int(part) for part in torch.__version__.split(".")[:2])
    if not _MIN_TORCH_VERSION <= (major, minor) < _MAX_TORCH_VERSION:
        raise RuntimeError(
            f"megatron_cfg.checkpoint.parallel_read_threads supports torch "
            f"{_MIN_TORCH_VERSION[0]}.{_MIN_TORCH_VERSION[1]} up to (not including) "
            f"{_MAX_TORCH_VERSION[0]}.{_MAX_TORCH_VERSION[1]}, found "
            f"{torch.__version__}. Set it to 1 to use PyTorch's checkpoint reader."
        )
    try:
        source = inspect.getsource(FileSystemReader.read_data)
    except OSError as e:
        raise RuntimeError(
            "megatron_cfg.checkpoint.parallel_read_threads needs the source of "
            "torch's FileSystemReader.read_data to verify it. Set it to 1 to use "
            "PyTorch's checkpoint reader."
        ) from e
    digest = hashlib.sha256(textwrap.dedent(source).encode()).hexdigest()
    if digest != _READ_DATA_SOURCE_SHA256:
        raise RuntimeError(
            f"megatron_cfg.checkpoint.parallel_read_threads: torch {torch.__version__} "
            f"has a FileSystemReader.read_data that {__name__} does not mirror "
            f"(source sha256 {digest}). Set it to 1 to use PyTorch's checkpoint reader."
        )


def install(num_threads: int) -> None:
    """Read torch_dist checkpoint files with ``num_threads`` threads per load.

    Replaces ``FileSystemReader.read_data`` for the whole process, so every later
    load through a local-file ``FileSystemReader`` (or a subclass that keeps
    ``read_data``) uses :func:`parallel_read_data`. Idempotent for the same
    ``num_threads``.

    Args:
        num_threads: Read threads per load. 1 keeps PyTorch's reader.

    Raises:
        ValueError: If ``num_threads`` is not a positive integer.
        RuntimeError: If a different thread count is already installed, or the
            installed torch is not supported (see :func:`check_torch_supported`).
    """
    global G_INSTALLED_NUM_THREADS
    if isinstance(num_threads, bool) or not isinstance(num_threads, int):
        raise ValueError(
            "megatron_cfg.checkpoint.parallel_read_threads must be an integer, "
            f"got {num_threads!r}."
        )
    if num_threads < 1:
        raise ValueError(
            "megatron_cfg.checkpoint.parallel_read_threads must be >= 1 "
            f"(1 keeps PyTorch's reader), got {num_threads}."
        )
    if G_INSTALLED_NUM_THREADS is not None:
        if num_threads != G_INSTALLED_NUM_THREADS:
            raise RuntimeError(
                f"Parallel checkpoint reads are already installed in this process "
                f"with {G_INSTALLED_NUM_THREADS} threads; cannot switch to {num_threads}."
            )
        return
    if num_threads == 1:
        return
    check_torch_supported()

    pytorch_read_data = FileSystemReader.read_data

    def read_data(
        self: FileSystemReader, plan: LoadPlan, planner: LoadPlanner
    ) -> torch.futures.Future[None]:
        if (
            type(self.fs) is not FileSystem
            or type(self)._slice_file is not FileSystemReader._slice_file
        ):
            warnings.warn(
                f"{type(self).__name__} does not read local files the way "
                f"FileSystemReader does; using PyTorch's single-stream reader.",
                stacklevel=2,
            )
            return pytorch_read_data(self, plan, planner)
        return parallel_read_data(
            self, plan, planner, num_threads=num_threads, window_bytes=WINDOW_BYTES
        )

    FileSystemReader.read_data = read_data
    G_INSTALLED_NUM_THREADS = num_threads
    print(
        f"Parallel checkpoint reads: {num_threads} threads, "
        f"{CHUNK_BYTES >> 20} MB requests, {WINDOW_BYTES >> 20} MB read-ahead",
        flush=True,
    )


def parallel_read_data(
    reader: FileSystemReader,
    plan: LoadPlan,
    planner: LoadPlanner,
    *,
    num_threads: int,
    window_bytes: int,
) -> torch.futures.Future[None]:
    """``FileSystemReader.read_data`` with ``num_threads`` reads in flight.

    Args:
        reader: The reader whose ``read_data`` this replaces; reads local files.
        plan: This rank's load plan.
        planner: The load planner that receives the items.
        num_threads: Threads issuing ``os.preadv`` requests of ``CHUNK_BYTES``.
        window_bytes: Bytes that may be read ahead of the item being deserialized.

    Returns:
        A completed future, as PyTorch's ``read_data`` returns.
    """
    per_file: dict[str, list[ReadItem]] = {}
    for read_item in plan.items:
        item_md = reader.storage_data[read_item.storage_index]
        per_file.setdefault(item_md.relative_path, []).append(read_item)

    with ThreadPoolExecutor(
        max_workers=num_threads, thread_name_prefix="nrl-ckpt-read"
    ) as pool:
        for relative_path, reqs in per_file.items():
            path = os.fspath(reader.fs.concat_path(reader.path, relative_path))
            _read_file(reader, planner, pool, path, reqs, window_bytes)

    fut: torch.futures.Future[None] = torch.futures.Future()
    fut.set_result(None)
    return fut


def _read_file(
    reader: FileSystemReader,
    planner: LoadPlanner,
    pool: ThreadPoolExecutor,
    path: str,
    reqs: list[ReadItem],
    window_bytes: int,
) -> None:
    reqs = sorted(reqs, key=lambda r: reader.storage_data[r.storage_index].offset)
    pending: collections.deque[tuple[ReadItem, bytearray, list[Future[None]]]] = (
        collections.deque()
    )
    fd = os.open(path, os.O_RDONLY)
    try:
        pending_bytes = 0
        for req in reqs:
            item_md = reader.storage_data[req.storage_index]
            buf = bytearray(item_md.length)
            view = memoryview(buf)
            chunks = [
                pool.submit(
                    _pread_into,
                    fd,
                    view[start : start + CHUNK_BYTES],
                    item_md.offset + start,
                    path,
                )
                for start in range(0, item_md.length, CHUNK_BYTES)
            ]
            pending.append((req, buf, chunks))
            pending_bytes += item_md.length
            while pending and pending_bytes > window_bytes:
                pending_bytes -= _load_next(reader, planner, pending)
        while pending:
            _load_next(reader, planner, pending)
    finally:
        # On error, drop reads that have not started and let running ones finish
        # before their fd is closed.
        unfinished = [chunk for _, _, chunks in pending for chunk in chunks]
        for chunk in unfinished:
            chunk.cancel()
        wait(unfinished)
        os.close(fd)


def _pread_into(fd: int, view: memoryview, offset: int, path: str) -> None:
    done = 0
    while done < len(view):
        n = os.preadv(fd, [view[done:]], offset + done)
        if n == 0:
            raise EOFError(f"{path}: unexpected end of file at byte {offset + done}")
        done += n


def _load_next(
    reader: FileSystemReader,
    planner: LoadPlanner,
    pending: collections.deque[tuple[ReadItem, bytearray, list[Future[None]]]],
) -> int:
    """Wait for the oldest pending item's reads, load it and return its size."""
    req, buf, chunks = pending[0]
    for chunk in chunks:
        chunk.result()
    pending.popleft()
    _load_item(reader, planner, req, buf)
    return len(buf)


def _load_item(
    reader: FileSystemReader, planner: LoadPlanner, req: ReadItem, data: bytearray
) -> None:
    # The body of PyTorch's read_data loop, with the file slice replaced by the
    # bytes already read.
    item_md = reader.storage_data[req.storage_index]
    transform_from = reader.transforms.transform_load_stream(
        req, item_md.transform_descriptors or (), io.BytesIO(data)
    )
    if req.type == LoadItemType.BYTE_IO:
        read_bytes = io.BytesIO(transform_from.read(-1))
        read_bytes.seek(0)
        planner.load_bytes(req, read_bytes)
        return
    if transform_from.seekable():
        seekable = transform_from
    else:
        # torch.load requires a seekable input.
        seekable = io.BytesIO(transform_from.read(-1))
        seekable.seek(0)
    tensor = cast(
        torch.Tensor, torch.load(seekable, map_location="cpu", weights_only=True)
    )
    tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
    target_tensor = planner.resolve_tensor(req).detach()
    if target_tensor.size() != tensor.size():
        raise AssertionError(
            f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
        )
    target_tensor.copy_(tensor)
    planner.commit_tensor(req, target_tensor)
