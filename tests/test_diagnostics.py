from __future__ import annotations

from parakeet_transcribe.diagnostics import _linux_triton_compiler_ready


def test_linux_triton_compiler_ok_when_gcc_on_path(monkeypatch) -> None:
    monkeypatch.setattr("parakeet_transcribe.diagnostics.sys.platform", "linux")
    monkeypatch.setitem(__import__("sys").modules, "triton", object())
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(
        "parakeet_transcribe.diagnostics.shutil.which",
        lambda tool: "/usr/bin/gcc" if tool == "gcc" else None,
    )
    ok, line = _linux_triton_compiler_ready()
    assert ok
    assert "gcc" in line


def test_linux_triton_compiler_missing_when_no_compiler(monkeypatch) -> None:
    monkeypatch.setattr("parakeet_transcribe.diagnostics.sys.platform", "linux")
    monkeypatch.setitem(__import__("sys").modules, "triton", object())
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr("parakeet_transcribe.diagnostics.shutil.which", lambda _tool: None)
    ok, line = _linux_triton_compiler_ready()
    assert not ok
    assert "MISSING Triton C compiler" in line


def test_linux_triton_compiler_ok_via_cc_env(monkeypatch) -> None:
    monkeypatch.setattr("parakeet_transcribe.diagnostics.sys.platform", "linux")
    monkeypatch.setitem(__import__("sys").modules, "triton", object())
    monkeypatch.setenv("CC", "custom-cc")
    monkeypatch.setattr(
        "parakeet_transcribe.diagnostics.shutil.which",
        lambda tool: "/opt/custom-cc" if tool == "custom-cc" else None,
    )
    ok, line = _linux_triton_compiler_ready()
    assert ok
    assert "CC=custom-cc" in line


def test_non_linux_skips_triton_compiler_check(monkeypatch) -> None:
    monkeypatch.setattr("parakeet_transcribe.diagnostics.sys.platform", "win32")
    ok, line = _linux_triton_compiler_ready()
    assert ok
    assert line == ""
