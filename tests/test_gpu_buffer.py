"""Unit tests for GPU buffer classes."""

# pylint: disable=protected-access

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from sb3_extra_buffers.gpu_buffers import GpuReplayBuffer, GpuRolloutBuffer, find_gpu_buffer_dtypes, has_zstd
from sb3_extra_buffers.gpu_buffers.compression_methods import COMPRESSION_METHOD_MAP
from sb3_extra_buffers.gpu_buffers.observation_store import (
    DenseObservationStore,
    RawObservationStore,
    create_observation_store,
)
from sb3_extra_buffers.gpu_buffers.raw_buffer import RawBuffer, SharedRawHeap
from sb3_extra_buffers.gpu_buffers.utils import estimate_max_slot_bytes, estimate_total_heap_bytes


def _compression_methods():
    methods = ["none", "rle"]
    if has_zstd():
        methods.append("zstd")
    return methods


@pytest.mark.parametrize("compression_method", _compression_methods())
def test_raw_heap_no_cross_cell_corruption(compression_method):
    buffer_size = 4
    n_envs = 2
    flat_len = 8
    dtypes = find_gpu_buffer_dtypes((2, 2, 2), compression_method=compression_method)
    elem_type = dtypes["elem_type"]
    arr_configs = {"size": flat_len, "dtype": elem_type}

    if compression_method == "none":
        store = create_observation_store(
            "none",
            buffer_size,
            n_envs,
            flat_len,
            elem_type,
            "cpu",
        )
    else:
        codec = COMPRESSION_METHOD_MAP[compression_method]
        n_cells = buffer_size * n_envs
        heap_bytes = estimate_total_heap_bytes(
            n_cells,
            flat_len,
            elem_type,
            dtypes["runs_type"],
            compression_method,
        )
        shared_heap = SharedRawHeap(n_cells, heap_bytes, device="cpu")

        def compress(arr, buffer, byte_start, **kwargs):
            return codec.compress(arr, buffer, byte_start, **kwargs)

        def decompress(buffer, meta, **kwargs):
            return codec.decompress(
                buffer,
                meta,
                elem_type=elem_type,
                runs_type=dtypes["runs_type"],
                arr_configs=arr_configs,
                **kwargs,
            )

        store = RawObservationStore(
            buffer_size,
            n_envs,
            flat_len,
            elem_type,
            "cpu",
            field_offset=0,
            compress=compress,
            decompress=decompress,
            compression_method=compression_method,
            shared_heap=shared_heap,
        )

    first = th.arange(flat_len, dtype=elem_type)
    second = th.flip(first, dims=(0,))
    store.write(0, 0, first)
    store.write(1, 1, second)

    th.testing.assert_close(store.read(0, 0), first)
    th.testing.assert_close(store.read(1, 1), second)


@pytest.mark.parametrize("compression_method", _compression_methods())
def test_raw_heap_growth_does_not_corrupt_neighbor(compression_method):
    if compression_method == "none":
        pytest.skip("dense store has no heap growth path")
    buffer_size = 2
    n_envs = 1
    flat_len = 64
    dtypes = find_gpu_buffer_dtypes((8, 8), compression_method=compression_method)
    elem_type = dtypes["elem_type"]
    arr_configs = {"size": flat_len, "dtype": elem_type}
    codec = COMPRESSION_METHOD_MAP[compression_method]
    n_cells = buffer_size * n_envs
    heap_bytes = estimate_total_heap_bytes(
        n_cells,
        flat_len,
        elem_type,
        dtypes["runs_type"],
        compression_method,
    )
    shared_heap = SharedRawHeap(n_cells, heap_bytes, device="cpu")

    def compress(arr, buffer, byte_start, **kwargs):
        return codec.compress(arr, buffer, byte_start, **kwargs)

    def decompress(buffer, meta, **kwargs):
        return codec.decompress(
            buffer,
            meta,
            elem_type=elem_type,
            runs_type=dtypes["runs_type"],
            arr_configs=arr_configs,
            **kwargs,
        )

    store = RawObservationStore(
        buffer_size,
        n_envs,
        flat_len,
        elem_type,
        "cpu",
        field_offset=0,
        compress=compress,
        decompress=decompress,
        compression_method=compression_method,
        shared_heap=shared_heap,
    )

    small = th.zeros(flat_len, dtype=elem_type)
    neighbor = th.randint(0, 255, (flat_len,), dtype=elem_type)
    store.write(0, 0, small)
    store.write(1, 0, neighbor)
    neighbor_before = store.read(1, 0).clone()
    grown = th.zeros(flat_len, dtype=elem_type)
    grown[0] = 1
    store.write(0, 0, grown)
    th.testing.assert_close(store.read(1, 0), neighbor_before)
    th.testing.assert_close(store.read(0, 0), grown)


@pytest.mark.parametrize("compression_method", _compression_methods())
def test_gpu_replay_add_sample(compression_method):
    obs_space = spaces.Box(low=0, high=255, shape=(3, 4), dtype=np.uint8)
    action_space = spaces.Discrete(2)
    buffer_size = 16
    n_envs = 2
    dtypes = find_gpu_buffer_dtypes(obs_space.shape, compression_method=compression_method)

    buffer = GpuReplayBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=action_space,
        n_envs=n_envs,
        compression_method=compression_method,
        dtypes=dtypes,
        buffer_device="cpu",
        device="cpu",
    )

    for step in range(buffer_size):
        obs = np.full((n_envs, 3, 4), step, dtype=np.uint8)
        next_obs = np.full((n_envs, 3, 4), step + 1, dtype=np.uint8)
        action = np.zeros(n_envs, dtype=np.int64)
        reward = np.ones(n_envs, dtype=np.float32)
        done = np.zeros(n_envs, dtype=np.float32)
        buffer.add(obs, next_obs, action, reward, done, [{} for _ in range(n_envs)])

    samples = buffer.sample(8)
    assert samples.observations.shape == (8, 3, 4)
    assert samples.next_observations.shape == (8, 3, 4)
    assert samples.observations.device.type == "cpu"


@pytest.mark.parametrize("compression_method", ["rle"])
def test_gpu_replay_compact_on_wrap(compression_method):
    obs_space = spaces.Box(low=0, high=255, shape=(2, 2), dtype=np.uint8)
    action_space = spaces.Discrete(2)
    buffer_size = 4
    n_envs = 1
    dtypes = find_gpu_buffer_dtypes(obs_space.shape, compression_method=compression_method)

    buffer = GpuReplayBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=action_space,
        n_envs=n_envs,
        compression_method=compression_method,
        dtypes=dtypes,
        buffer_device="cpu",
        device="cpu",
    )
    assert buffer.shared_heap is not None

    for step in range(buffer_size * 2):
        obs = np.full((n_envs, 2, 2), step % 3, dtype=np.uint8)
        next_obs = obs + 1
        buffer.add(obs, next_obs, np.array([0]), np.array([1.0]), np.array([0.0]), [{}])

    assert buffer.full
    assert buffer.shared_heap.data_end <= buffer.shared_heap.buffer.size
    samples = buffer.sample(2)
    assert samples.observations.shape == (2, 2, 2)


@pytest.mark.parametrize("compression_method", _compression_methods())
def test_gpu_rollout_get(compression_method):
    obs_space = spaces.Box(low=0, high=255, shape=(2, 2), dtype=np.uint8)
    action_space = spaces.Discrete(2)
    buffer_size = 8
    n_envs = 2
    dtypes = find_gpu_buffer_dtypes(obs_space.shape, compression_method=compression_method)

    buffer = GpuRolloutBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=action_space,
        n_envs=n_envs,
        compression_method=compression_method,
        dtypes=dtypes,
        buffer_device="cpu",
        device="cpu",
    )

    for step in range(buffer_size):
        obs = np.full((n_envs, 2, 2), step, dtype=np.uint8)
        action = np.zeros(n_envs, dtype=np.int64)
        reward = np.ones(n_envs, dtype=np.float32)
        episode_start = np.zeros(n_envs, dtype=np.float32)
        value = th.zeros(n_envs)
        log_prob = th.zeros(n_envs)
        buffer.add(obs, action, reward, episode_start, value, log_prob)

    batches = list(buffer.get(batch_size=4))
    assert len(batches) == buffer_size * n_envs // 4
    assert batches[0].observations.shape[0] == 4
    assert batches[0].observations.device.type == "cpu"
    if buffer.shared_heap is not None:
        assert buffer.shared_heap.data_end <= buffer.shared_heap.buffer.size


def test_dense_observation_store_flatten():
    store = DenseObservationStore(4, 2, 4, th.uint8, "cpu")
    store.write(0, 0, th.tensor([1, 2, 3, 4], dtype=th.uint8))
    store.write(1, 1, th.tensor([5, 6, 7, 8], dtype=th.uint8))
    store.flatten()
    th.testing.assert_close(store.read_flat(0), th.tensor([1, 2, 3, 4], dtype=th.uint8))
    th.testing.assert_close(store.read_flat(5), th.tensor([5, 6, 7, 8], dtype=th.uint8))


def test_zstd_roundtrip():
    if not has_zstd():
        pytest.skip("zstd not installed")
    from sb3_extra_buffers.gpu_buffers.compression_methods.compression_methods_zstd import (
        zstd_compress,
        zstd_decompress,
    )

    flat_len = 64
    elem_type = th.uint8
    input_arr = th.randint(0, 255, (flat_len,), dtype=elem_type)
    heap_bytes = max(
        estimate_max_slot_bytes(flat_len, elem_type, th.uint16, "zstd"),
        flat_len + 32,
    )
    buffer = RawBuffer(heap_bytes, device="cpu")
    meta = zstd_compress(input_arr, buffer, byte_start=0, elem_type=elem_type)
    output = zstd_decompress(
        buffer,
        meta,
        elem_type=elem_type,
        runs_type=th.uint16,
        arr_configs={"size": flat_len, "dtype": elem_type},
    )
    th.testing.assert_close(output, input_arr)


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("compression_method", _compression_methods())
def test_gpu_replay_cuda(compression_method):
    obs_space = spaces.Box(low=0, high=255, shape=(2, 2), dtype=np.uint8)
    action_space = spaces.Discrete(2)
    dtypes = find_gpu_buffer_dtypes(obs_space.shape, compression_method=compression_method)
    buffer = GpuReplayBuffer(
        buffer_size=8,
        observation_space=obs_space,
        action_space=action_space,
        n_envs=1,
        compression_method=compression_method,
        dtypes=dtypes,
        buffer_device="cuda",
        device="cuda",
    )
    obs = np.full((1, 2, 2), 7, dtype=np.uint8)
    buffer.add(obs, obs, np.array([0]), np.array([1.0]), np.array([0.0]), [{}])
    sample = buffer.sample(1)
    assert sample.observations.device.type == "cuda"
