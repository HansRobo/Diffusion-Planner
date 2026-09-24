"""An ONNX session that fell back to CPU must say so, not run unaccelerated in silence."""

from __future__ import annotations

import sys
import types

import pytest

from scenario_generation.simulate import CPU_EP, CUDA_EP, TENSORRT_EP, _require_accelerator

# Distinct from every index the cases ask for, so a device_id off the wrong source shows up.
_CURRENT_DEVICE = 2
_TRT_CACHE = "/engines"
_TRT_CACHE_OPTIONS = {
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": _TRT_CACHE,
    "trt_timing_cache_enable": True,
}


class _SessionOptions:
    """Stands in for ``ort.SessionOptions``; only the thread count is read back."""

    def __init__(self):
        self.intra_op_num_threads = 0


class _Session:
    """Stands in for ``ort.InferenceSession``, reporting the providers it was handed as active."""

    def __init__(self, providers, provider_options=None, sess_options=None):
        self.providers = providers
        self.provider_options = provider_options
        self.sess_options = sess_options

    def get_providers(self):
        return self.providers

    def get_inputs(self):
        return []

    def get_outputs(self):
        return []


def _open_session(monkeypatch, device, providers=None, **kwargs):
    """Build an ``_OnnxModel`` against a fake onnxruntime and hand back the session it opened."""
    import scenario_generation.simulate as simulate

    def _factory(path, sess_options, providers, provider_options):
        return _Session(providers, provider_options, sess_options)

    monkeypatch.setattr(simulate.torch.cuda, "current_device", lambda: _CURRENT_DEVICE)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(InferenceSession=_factory, SessionOptions=_SessionOptions),
    )
    return simulate._OnnxModel("m.onnx", device, providers, **kwargs).session


def test_a_gpu_request_that_landed_on_cpu_raises():
    """ORT lists providers the build contains, not the ones that loaded; it warns and continues."""
    session = _Session([CPU_EP])

    with pytest.raises(RuntimeError, match="runs on"):
        _require_accelerator(session, [TENSORRT_EP, CUDA_EP, CPU_EP], "m.onnx")


def test_the_message_names_the_way_out():
    session = _Session([CPU_EP])

    with pytest.raises(RuntimeError, match=r"providers=\['CPUExecutionProvider'\]"):
        _require_accelerator(session, [CUDA_EP, CPU_EP], "m.onnx")


@pytest.mark.parametrize("active", [[TENSORRT_EP, CPU_EP], [CUDA_EP, CPU_EP]])
def test_either_accelerator_satisfies_the_request(active):
    """TensorRT partitions the graph and leaves the rest to CPU, so CPU being present is normal."""
    _require_accelerator(_Session(active), [TENSORRT_EP, CUDA_EP, CPU_EP], "m.onnx")


def test_asking_for_cpu_is_not_a_failure():
    _require_accelerator(_Session([CPU_EP]), [CPU_EP], "m.onnx")


def test_the_default_does_not_reach_for_tensorrt(monkeypatch):
    """TensorRT partitions the graph up front and refuses ops it cannot build, so defaulting to
    it turns a model it dislikes into a session that never opens. It has to be asked for."""
    assert _open_session(monkeypatch, "cuda").providers == [CUDA_EP, CPU_EP]


def test_asking_for_cpu_by_device_does_not_request_a_gpu_provider(monkeypatch):
    assert _open_session(monkeypatch, "cpu").providers == [CPU_EP]


@pytest.mark.parametrize(
    "device,providers,expected",
    [
        # 0 is a real GPU, not "unset": it must not read as absent and fall back to the default.
        ("cuda:0", [CUDA_EP, CPU_EP], {CUDA_EP: {"device_id": 0}, CPU_EP: {}}),
        ("cuda", [CUDA_EP, CPU_EP], {CUDA_EP: {"device_id": _CURRENT_DEVICE}, CPU_EP: {}}),
        (
            "cuda:1",
            [TENSORRT_EP, CUDA_EP, CPU_EP],
            {
                TENSORRT_EP: {"device_id": 1, **_TRT_CACHE_OPTIONS, "trt_fp16_enable": False},
                CUDA_EP: {"device_id": 1},
                CPU_EP: {},
            },
        ),
    ],
)
def test_gpu_providers_follow_requested_device(monkeypatch, device, providers, expected):
    """ORT ignores torch.cuda.set_device() and defaults to GPU 0, so every rank of a distributed
    run piles onto the first visible GPU unless the provider itself carries the index."""
    session = _open_session(monkeypatch, device, providers, engine_cache_dir=_TRT_CACHE)

    assert dict(zip(session.providers, session.provider_options)) == expected


def test_intra_op_threads_default_to_one(monkeypatch):
    """Left unset, ORT sizes the pool from the machine's core count in every rank at once."""
    assert _open_session(monkeypatch, "cuda").sess_options.intra_op_num_threads == 1


def test_intra_op_threads_follow_the_environment(monkeypatch):
    """A host that wants a wider pool has to be able to say so without editing the adapter."""
    monkeypatch.setenv("SCENARIO_SIM_ORT_INTRA", "4")

    assert _open_session(monkeypatch, "cuda").sess_options.intra_op_num_threads == 4


def test_tensorrt_is_chosen_by_the_environment(monkeypatch):
    monkeypatch.setenv("SCENARIO_SIM_ORT_EP", "trt")
    session = _open_session(monkeypatch, "cuda:0", engine_cache_dir=_TRT_CACHE)
    assert session.providers == [TENSORRT_EP, CUDA_EP, CPU_EP]


def test_a_tensorrt_request_that_landed_on_cuda_raises():
    """A TensorRT run that quietly became a CUDA run would be reported under the wrong name."""
    from scenario_generation.simulate import _require_tensorrt

    with pytest.raises(RuntimeError, match="asked for TensorrtExecutionProvider"):
        _require_tensorrt(_Session([CUDA_EP, CPU_EP]), "m.onnx")


def test_each_export_gets_its_own_engine_directory(tmp_path):
    """TensorRT's cache does not notice new weights under the same graph, so two checkpoints of
    one run -- same file name, same size -- must not share an engine."""
    import os

    from scenario_generation.simulate import _graph_key

    a = tmp_path / "epoch0091" / "diffusion_planner.onnx"
    b = tmp_path / "epoch0092" / "diffusion_planner.onnx"
    for p in (a, b):
        p.parent.mkdir()
        p.write_bytes(b"x" * 16)
        os.utime(p, ns=(1, 1))
    assert _graph_key(a) != _graph_key(b)

    before = _graph_key(a)
    os.utime(a, ns=(2, 2))  # re-exported in place
    assert _graph_key(a) != before
