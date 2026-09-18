"""Tests for the GPU/CUDA startup path.

Two defects found by actually running on a GPU:

1. ctranslate2 dlopen()s libcublas at *encode* time, but the nvidia-* wheels
   install it under site-packages/nvidia/<lib>/lib, which is not on the loader
   path. The model loaded fine and every job then failed with
   "Library libcublas.so.12 is not found or cannot be loaded".
2. load_model() inserted the new model into the cache before evicting, so a
   cache cap of 1 briefly held two models in VRAM.

Neither is visible without exercising the real path, so both are pinned here
with fakes rather than a GPU.
"""

from __future__ import annotations

import sys
import types

import pytest

import transcribe_server as s


class FakeInfo:
    duration = 1.0
    language = "en"


class RecordingWhisper:
    """Stands in for faster_whisper.WhisperModel, noting cache size on build."""

    constructed: list[int] = []

    def __init__(self, name, device=None, compute_type=None):  # noqa: ARG002
        RecordingWhisper.constructed.append(len(s._MODEL_CACHE))
        self.name = name

    def transcribe(self, *args, **kwargs):  # noqa: ARG002
        return iter([]), FakeInfo()


@pytest.fixture
def fake_whisper(monkeypatch):
    RecordingWhisper.constructed = []
    module = types.SimpleNamespace(WhisperModel=RecordingWhisper)
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    s._MODEL_CACHE.clear()
    yield RecordingWhisper
    s._MODEL_CACHE.clear()


# --------------------------------------------------------------------------- #
# Eviction ordering
# --------------------------------------------------------------------------- #


def test_model_is_evicted_before_the_next_one_is_constructed(configured, fake_whisper):
    """A cap of 1 must never hold two models in VRAM, not even briefly."""
    assert s.ARGS.model_cache == 1

    s.load_model("base", "cpu", "int8")  # cache empty
    s.load_model("base", "cpu", "float16")  # at capacity: must evict first

    assert fake_whisper.constructed == [0, 0], (
        "the second model was built while the first was still cached"
    )
    assert len(s._MODEL_CACHE) == 1


def test_model_cache_respects_a_higher_cap(configured, fake_whisper, monkeypatch):
    monkeypatch.setattr(s.ARGS, "model_cache", 2)

    s.load_model("base", "cpu", "int8")
    s.load_model("base", "cpu", "float16")
    assert fake_whisper.constructed == [0, 1], "cap 2 should keep both"
    assert len(s._MODEL_CACHE) == 2

    s.load_model("base", "cpu", "float32")  # now at capacity
    assert fake_whisper.constructed == [0, 1, 1], "evict one before building"
    assert len(s._MODEL_CACHE) == 2


def test_cache_hit_does_not_reload(configured, fake_whisper):
    first = s.load_model("base", "cpu", "int8")
    second = s.load_model("base", "cpu", "int8")
    assert first is second
    assert len(fake_whisper.constructed) == 1


def test_eviction_is_audited(configured, fake_whisper):
    s.load_model("base", "cpu", "int8")
    s.load_model("base", "cpu", "float16")

    events = [e["event"] for e in configured.events()]
    assert "model.loaded" in events
    assert "model.unloaded" in events


# --------------------------------------------------------------------------- #
# Preload must prove the device can encode, not just load
# --------------------------------------------------------------------------- #


class UnencodableModel:
    """Loads fine, then fails on encode — exactly the missing-cuBLAS case."""

    def transcribe(self, *args, **kwargs):  # noqa: ARG002
        raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")


def test_preload_exits_when_the_model_cannot_encode(configured, monkeypatch, capsys):
    monkeypatch.setattr(s.ARGS, "device", "cuda")
    monkeypatch.setattr(s, "load_model", lambda *a, **k: UnencodableModel())

    with pytest.raises(SystemExit) as exc:
        s.preload_model(s.ARGS)

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "cannot run" in out
    assert "libcublas" in out
    assert "--device cpu" in out, "the operator needs a way out"
    assert "Model ready" not in out, "must not print a false all-clear"


def test_preload_succeeds_when_the_model_encodes(configured, monkeypatch, capsys):
    monkeypatch.setattr(s.ARGS, "device", "cuda")
    monkeypatch.setattr(s, "load_model", lambda *a, **k: RecordingWhisper("base"))

    s.preload_model(s.ARGS)

    assert "Model ready." in capsys.readouterr().out


def test_verify_device_rejects_an_unencodable_model(configured):
    with pytest.raises(RuntimeError, match="libcublas"):
        s.verify_device(UnencodableModel(), "base")


def test_verify_device_accepts_a_working_model(configured):
    s.verify_device(RecordingWhisper("base"), "base")  # must not raise


# --------------------------------------------------------------------------- #
# CUDA library preloading
# --------------------------------------------------------------------------- #


def test_enable_cuda_libraries_reports_what_it_loaded(configured):
    """With the wheels installed this must find libcublas/libcudnn."""
    loaded = s.enable_cuda_libraries()

    if not loaded:
        pytest.skip("nvidia-* wheels are not installed in this environment")

    assert any("cublas" in name for name in loaded), loaded
    assert any("cudnn" in name for name in loaded), loaded


def test_enable_cuda_libraries_is_safe_when_the_wheels_are_absent(
    configured, monkeypatch
):
    """A CPU-only install must not blow up at startup."""
    monkeypatch.setattr(s.importlib.util, "find_spec", lambda name: None)
    assert s.enable_cuda_libraries() == []


def test_enable_cuda_libraries_makes_the_soname_resolvable(configured):
    """The whole point: a bare dlopen of the soname must work afterwards."""
    import ctypes

    if not s.enable_cuda_libraries():
        pytest.skip("nvidia-* wheels are not installed in this environment")

    ctypes.CDLL("libcublas.so.12")  # would raise OSError if not preloaded
