# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for pickle-related patches in env_override.py.

1. _apply_fxgraphcache_pickle_patch: wraps a pickler's dumps method to
   convert ValueError into a bypass exception.
2. _patch_autograd_cache_pickle_for_triton: patches AOTAutogradCache._pickle_entry
   to handle CachingAutotuner objects with unpicklable exec'd launcher functions.
"""

import dataclasses
import pickle

import pytest

import vllm.env_override  # noqa: F401 — triggers patches
from vllm.env_override import _apply_fxgraphcache_pickle_patch

functorch_config = pytest.importorskip(
    "torch._functorch.config",
    reason="requires torch._functorch",
)
AOTAutogradCache = pytest.importorskip(
    "torch._functorch._aot_autograd.autograd_cache",
    reason="requires torch._functorch",
).AOTAutogradCache
CachingAutotuner = pytest.importorskip(
    "torch._inductor.runtime.triton_heuristics",
    reason="requires torch._inductor",
).CachingAutotuner


class _BypassStub(Exception):
    """Stand-in for BypassFxGraphCache in unit tests."""


class TestApplyFxgraphcachePicklePatch:
    def test_valueerror_converted_to_bypass(self):
        class Pickler:
            def dumps(self, obj):
                raise ValueError("can't serialize blocked layout")

        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)

        with pytest.raises(_BypassStub, match="Failed to pickle cache key"):
            Pickler().dumps(object())

    def test_original_valueerror_chained(self):
        class Pickler:
            def dumps(self, obj):
                raise ValueError("bad tensor layout")

        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)

        with pytest.raises(_BypassStub) as exc_info:
            Pickler().dumps(object())

        cause = exc_info.value.__cause__
        assert isinstance(cause, ValueError)
        assert str(cause) == "bad tensor layout"

    def test_non_valueerror_propagates(self):
        class Pickler:
            def dumps(self, obj):
                raise TypeError("unexpected type")

        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)

        with pytest.raises(TypeError, match="unexpected type"):
            Pickler().dumps(object())

    def test_normal_return_preserved(self):
        sentinel = b"serialized-graph-key"

        class Pickler:
            def dumps(self, obj):
                return sentinel

        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)

        assert Pickler().dumps(object()) is sentinel

    def test_idempotent(self):
        class Pickler:
            def dumps(self, obj):
                return b"ok"

        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)
        first_dumps = Pickler.dumps
        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)

        assert Pickler.dumps is first_dumps

    def test_sentinel_attribute_set(self):
        class Pickler:
            def dumps(self, obj):
                return b"ok"

        assert not hasattr(Pickler.dumps, "_vllm_patched")
        assert not getattr(Pickler, "_vllm_fxgraph_dumps_patched", False)

        _apply_fxgraphcache_pickle_patch(Pickler, _BypassStub)

        assert Pickler.dumps._vllm_patched is True  # type: ignore[attr-defined]
        assert Pickler._vllm_fxgraph_dumps_patched is True  # type: ignore[attr-defined]


def test_patch_applied_in_current_environment():
    """Integration: verify patch state matches current torch version."""
    from torch._inductor.codecache import FxGraphCachePickler

    from vllm.utils.torch_utils import is_torch_equal_or_newer

    should_be_patched = is_torch_equal_or_newer(
        "2.10.0"
    ) and not is_torch_equal_or_newer("2.11.0")

    assert getattr(FxGraphCachePickler, "_vllm_fxgraph_dumps_patched", False) == (
        should_be_patched
    )
    assert hasattr(FxGraphCachePickler.dumps, "_vllm_patched") == should_be_patched


# --- AOTAutogradCache._pickle_entry triton launcher patch ---

# Module-level so pickle can resolve by name.


@dataclasses.dataclass
class _EntryWithField:
    fn: object = None


@dataclasses.dataclass
class _EntryWithAutotuner:
    kernel: object = None


class _FakeFn:
    fn = None
    __globals__ = None
    used_global_vals = None
    _hash_lock = None

    def repr(self, *a):
        return "fake"


class _FakeAutotuner(CachingAutotuner):  # type: ignore[valid-type, misc]
    """Minimal subclass; inherits prepare_for_pickle / restore_after_unpickle."""

    def __init__(self, launchers=None):
        self.launchers = launchers or []
        self.fn = _FakeFn()
        self.lock = None
        self.compile_results = []
        self.size_hints = None
        self.triton_meta = {}
        self.inductor_meta = {}
        self.heuristic_type = None
        self.device_props = None
        self._reload_kernel = None

    def __getstate__(self):
        assert not self.launchers
        return {**self.__dict__, "lock": None}

    def __setstate__(self, state):
        self.__dict__.update(state)


def _make_exec_function():
    scope: dict[str, object] = {}
    exec("def launcher(x): return x + 1", scope)
    return scope["launcher"]


def test_autograd_cache_pickle_patch_applied():
    assert getattr(AOTAutogradCache, "_vllm_pickle_patched", False)


def test_autotuner_with_launchers_pickles_and_restores():
    launcher = _make_exec_function()
    autotuner = _FakeAutotuner(launchers=[launcher])
    entry = _EntryWithAutotuner(kernel=autotuner)

    original = functorch_config.strict_autograd_cache
    try:
        functorch_config.strict_autograd_cache = True
        result = AOTAutogradCache._pickle_entry(entry, False)
        assert result is not None
    finally:
        functorch_config.strict_autograd_cache = original

    assert autotuner.launchers == [launcher]
    assert pickle.loads(result).kernel.launchers == []


def test_exec_function_replaced_with_none():
    """Unpicklable exec'd functions are replaced with None in the output."""
    result = AOTAutogradCache._pickle_entry(
        _EntryWithField(fn=_make_exec_function()), False
    )
    assert result is not None
    assert pickle.loads(result).fn is None


def test_autotuner_restored_on_pickle_failure():
    """CachingAutotuner launchers are restored even when pickle fails."""
    launcher = _make_exec_function()
    autotuner = _FakeAutotuner(launchers=[launcher])
    entry = _EntryWithAutotuner(kernel=autotuner)
    # Add an object that can't be pickled and isn't caught by our reducer
    import threading

    entry.bad = threading.Lock()  # type: ignore[attr-defined]

    result = AOTAutogradCache._pickle_entry(entry, False)
    assert result is None
    assert autotuner.launchers == [launcher]


def test_aot_compile_pickler_patch_applied():
    from torch._dynamo.aot_compile import AOTCompilePickler

    assert getattr(AOTCompilePickler, "_vllm_patched", False)


def test_aot_compile_pickler_replaces_exec_function():
    import io

    from torch._dynamo.aot_compile import AOTCompilePickler

    buf = io.BytesIO()
    p = AOTCompilePickler({}, buf)
    p.dump({"fn": _make_exec_function()})
    assert pickle.loads(buf.getvalue())["fn"] is None
