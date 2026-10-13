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
"""Parallel torch_dist checkpoint reads, against real torch.distributed.checkpoint on CPU."""

import io
import os
from pathlib import Path
from typing import IO, Any, cast

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    CheckpointException,
    FileSystemReader,
    FileSystemWriter,
)
from torch.distributed.checkpoint._extension import (
    ExtensionRegistry,
    StreamTransformExtension,
)
from torch.distributed.checkpoint.filesystem import FileSystem
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner

from nemo_rl.models.megatron import ckpt_parallel_read

# A float32 tensor of 1.5 read requests plus a remainder; DCP stores it as one item.
_BIG_NUMEL = ckpt_parallel_read.CHUNK_BYTES * 3 // 8 + 3
# Smaller than the largest item, so the read-ahead window is exceeded.
_SMALL_WINDOW_BYTES = 4 << 20
# An odd request size, so request boundaries fall inside every larger item.
_ODD_CHUNK_BYTES = (1 << 20) + 7
_XOR_TABLE = bytes(b ^ 0x5A for b in range(256))


class _XorWriter(io.RawIOBase):
    def __init__(self, output: IO[bytes]) -> None:
        super().__init__()
        self._output = output

    def writable(self) -> bool:
        return True

    def write(self, b: Any) -> int:
        data = bytes(b)
        self._output.write(data.translate(_XOR_TABLE))
        return len(data)


class _XorReader(io.RawIOBase):
    """Not seekable, like a decompressing stream."""

    def __init__(self, source: IO[bytes]) -> None:
        super().__init__()
        self._source = source

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        view = memoryview(b)
        data = self._source.read(len(view))
        view[: len(data)] = data.translate(_XOR_TABLE)
        return len(data)


class _XorExtension(StreamTransformExtension):
    """A stream transform, so items carry transform descriptors."""

    @staticmethod
    def registry_name() -> str:
        return "nemo_rl_test.xor"

    @staticmethod
    def from_descriptor(version: str) -> "_XorExtension":
        return _XorExtension()

    def get_descriptor(self) -> str:
        return f"{self.registry_name()}/1"

    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        return cast(IO[bytes], _XorWriter(output))

    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        return cast(IO[bytes], _XorReader(input))


def _state_dict() -> dict[str, Any]:
    gen = torch.Generator().manual_seed(0)
    return {
        "big": torch.randn(_BIG_NUMEL, generator=gen),
        "matrix": torch.randn(257, 129, generator=gen).to(torch.bfloat16),
        "ints": torch.randint(-1000, 1000, (4097,), generator=gen),
        "scalar": torch.tensor(3.5),
        "empty": torch.empty(0),
        "nested": {
            "a": torch.arange(10, dtype=torch.int32),
            "b": {"c": torch.ones(3, 3)},
        },
        # Stored as BYTE_IO items.
        "blob": io.BytesIO(b"raw bytes \x00\x01" * 1000),
        "step": 7,
        "meta": {"name": "ckpt", "values": [1, 2.5, "x"]},
    }


def _destination(value: Any) -> Any:
    """Same structure with zeroed tensors and None for everything DCP stores as bytes."""
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    if isinstance(value, dict):
        return {k: _destination(v) for k, v in value.items()}
    return None


def _assert_same(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
    elif isinstance(expected, io.BytesIO):
        assert isinstance(actual, io.BytesIO)
        assert actual.getvalue() == expected.getvalue()
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same(actual[key], expected[key])
    else:
        assert actual == expected


def _registry() -> ExtensionRegistry:
    registry = ExtensionRegistry()
    registry.register(_XorExtension)
    return registry


class _ParallelReader(FileSystemReader):
    def __init__(self, path: Path, *, window_bytes: int) -> None:
        super().__init__(path, _extension_registry=_registry())
        self._window_bytes = window_bytes

    def read_data(
        self, plan: LoadPlan, planner: LoadPlanner
    ) -> torch.futures.Future[None]:
        return ckpt_parallel_read.parallel_read_data(
            self, plan, planner, num_threads=4, window_bytes=self._window_bytes
        )


def _save(path: Path, *, single_file_per_rank: bool, xor: bool) -> None:
    writer = FileSystemWriter(
        path,
        single_file_per_rank=single_file_per_rank,
        _extensions=[_XorExtension()] if xor else None,
    )
    dcp.save(_state_dict(), storage_writer=writer, no_dist=True)


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("dcp")
    layouts = {
        "single_file": {"single_file_per_rank": True, "xor": False},
        "file_per_item": {"single_file_per_rank": False, "xor": False},
        "transformed": {"single_file_per_rank": True, "xor": True},
    }
    for name, kwargs in layouts.items():
        _save(root / name, **kwargs)
    return {name: root / name for name in layouts}


@pytest.fixture
def restore_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo install() after the test."""
    monkeypatch.setattr(FileSystemReader, "read_data", FileSystemReader.read_data)
    monkeypatch.setattr(ckpt_parallel_read, "G_INSTALLED_NUM_THREADS", None)


@pytest.mark.parametrize("layout", ["single_file", "file_per_item", "transformed"])
@pytest.mark.parametrize(
    ("chunk_bytes", "window_bytes"),
    [
        (ckpt_parallel_read.CHUNK_BYTES, _SMALL_WINDOW_BYTES),
        (_ODD_CHUNK_BYTES, ckpt_parallel_read.WINDOW_BYTES),
    ],
)
def test_parallel_read_matches_pytorch_reader(
    monkeypatch: pytest.MonkeyPatch,
    checkpoints: dict[str, Path],
    layout: str,
    chunk_bytes: int,
    window_bytes: int,
) -> None:
    monkeypatch.setattr(ckpt_parallel_read, "CHUNK_BYTES", chunk_bytes)
    path = checkpoints[layout]
    expected = _state_dict()

    default = _destination(expected)
    dcp.load(
        default,
        storage_reader=FileSystemReader(path, _extension_registry=_registry()),
        no_dist=True,
    )
    parallel = _destination(expected)
    dcp.load(
        parallel,
        storage_reader=_ParallelReader(path, window_bytes=window_bytes),
        no_dist=True,
    )

    _assert_same(default, expected)
    _assert_same(parallel, default)


def test_install_routes_dcp_load_through_parallel_reader(
    monkeypatch: pytest.MonkeyPatch,
    restore_reader: None,
    checkpoints: dict[str, Path],
) -> None:
    calls: list[dict[str, Any]] = []
    parallel_read_data = ckpt_parallel_read.parallel_read_data

    def spy(*args: Any, **kwargs: Any) -> torch.futures.Future[None]:
        calls.append(kwargs)
        return parallel_read_data(*args, **kwargs)

    monkeypatch.setattr(ckpt_parallel_read, "parallel_read_data", spy)
    ckpt_parallel_read.install(4)
    ckpt_parallel_read.install(4)  # idempotent

    loaded = _destination(_state_dict())
    dcp.load(
        loaded,
        storage_reader=FileSystemReader(checkpoints["single_file"]),
        no_dist=True,
    )

    assert [c["num_threads"] for c in calls] == [4]
    _assert_same(loaded, _state_dict())
    with pytest.raises(RuntimeError, match="already installed"):
        ckpt_parallel_read.install(2)


def test_install_one_thread_keeps_pytorch_reader(restore_reader: None) -> None:
    original = FileSystemReader.read_data
    ckpt_parallel_read.install(1)
    assert FileSystemReader.read_data is original


@pytest.mark.parametrize("value", [0, -1, True, 2.0, "4"])
def test_install_rejects_invalid_thread_count(restore_reader: None, value: Any) -> None:
    original = FileSystemReader.read_data
    with pytest.raises(ValueError, match="parallel_read_threads"):
        ckpt_parallel_read.install(value)
    assert FileSystemReader.read_data is original


def test_installed_torch_is_supported() -> None:
    # Fails when CI moves to a torch whose reader this module was not checked against.
    ckpt_parallel_read.check_torch_supported()


@pytest.mark.parametrize("version", ["2.10.0", "2.14.0", "2.14.0a0+git0", "3.0.0"])
def test_unsupported_torch_version_raises(
    monkeypatch: pytest.MonkeyPatch, restore_reader: None, version: str
) -> None:
    monkeypatch.setattr(torch, "__version__", version)
    original = FileSystemReader.read_data
    with pytest.raises(RuntimeError, match="supports torch"):
        ckpt_parallel_read.install(4)
    assert FileSystemReader.read_data is original


def test_changed_pytorch_reader_raises(
    monkeypatch: pytest.MonkeyPatch, restore_reader: None
) -> None:
    def read_data(
        self: FileSystemReader, plan: LoadPlan, planner: LoadPlanner
    ) -> torch.futures.Future[None]:
        raise NotImplementedError

    monkeypatch.setattr(FileSystemReader, "read_data", read_data)
    with pytest.raises(RuntimeError, match="does not mirror"):
        ckpt_parallel_read.install(4)
    assert FileSystemReader.read_data is read_data


class _OtherFileSystem(FileSystem):
    pass


def test_reader_without_local_files_keeps_pytorch_reader(
    monkeypatch: pytest.MonkeyPatch,
    restore_reader: None,
    checkpoints: dict[str, Path],
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("parallel_read_data must not be used")

    ckpt_parallel_read.install(4)
    monkeypatch.setattr(ckpt_parallel_read, "parallel_read_data", fail)
    reader = FileSystemReader(checkpoints["single_file"])
    reader.fs = _OtherFileSystem()

    loaded = _destination(_state_dict())
    with pytest.warns(UserWarning, match="single-stream reader"):
        dcp.load(loaded, storage_reader=reader, no_dist=True)
    _assert_same(loaded, _state_dict())


def test_truncated_file_raises(tmp_path: Path) -> None:
    _save(tmp_path, single_file_per_rank=True, xor=False)
    (data_file,) = tmp_path.glob("*.distcp")
    os.truncate(data_file, data_file.stat().st_size - 1000)

    # DCP wraps the reader's EOFError.
    with pytest.raises(CheckpointException, match="unexpected end of file"):
        dcp.load(
            _destination(_state_dict()),
            storage_reader=_ParallelReader(tmp_path, window_bytes=_SMALL_WINDOW_BYTES),
            no_dist=True,
        )
