"""The sizing report that documents the reference machine.

SpeechInt is dimensioned for six CPU cores with AVX2 and 16 GB RAM. These tests
pin the contract: the minimum is reported, a machine below it is flagged but
never blocked, and the values survive a missing /proc or an unknown platform.
"""

from pathlib import Path

from app import host


def test_the_reference_minimum_is_six_cores_and_sixteen_gigabytes():
    assert host.MIN_CORES == 6
    assert host.MIN_MEMORY_GB == 16.0


def test_inspect_reports_the_minimum_and_the_budget():
    info = host.inspect()

    assert info["minimum"] == {"cores": 6, "memory_gb": 16.0, "avx2": True}
    assert set(info["budget_gb"]) == {"gateway", "whisper", "llm"}
    # The whole stack must fit into the reference machine with room to spare.
    assert sum(info["budget_gb"].values()) < host.MIN_MEMORY_GB / 2
    assert info["cores"] > 0


def test_a_small_machine_is_flagged_but_still_reported(monkeypatch):
    monkeypatch.setattr(host.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(host, "_host_memory_gb", lambda: 4.0)

    info = host.inspect()

    assert info["meets_minimum"] is False
    assert info["cores"] == 2
    assert info["memory_gb"] == 4.0
    assert any("2 CPU-Kerne" in warning for warning in info["warnings"])
    assert any("4.0 GB RAM" in warning for warning in info["warnings"])


def test_the_reference_machine_has_no_warnings(monkeypatch):
    monkeypatch.setattr(host.os, "cpu_count", lambda: 6)
    monkeypatch.setattr(host, "_host_memory_gb", lambda: 16.0)
    monkeypatch.setattr(host, "_has_avx2", lambda: True)

    info = host.inspect()

    assert info["warnings"] == []
    assert info["meets_minimum"] is True


def test_a_small_container_cap_is_not_held_against_the_host(monkeypatch):
    # The gateway container is capped on purpose; only the memory of the
    # machine the models run on decides whether the endpoint is sized right.
    monkeypatch.setattr(host, "_host_memory_gb", lambda: 32.0)
    monkeypatch.setattr(host, "_container_memory_gb", lambda: 0.5)

    info = host.inspect()

    assert info["memory_gb"] == 32.0
    assert info["container_memory_gb"] == 0.5
    assert not any("RAM" in warning for warning in info["warnings"])


def test_missing_avx2_is_only_reported_on_x86(monkeypatch):
    monkeypatch.setattr(host, "_has_avx2", lambda: False)

    assert any("AVX2" in warning for warning in host.warnings())

    monkeypatch.setattr(host, "_has_avx2", lambda: None)  # ARM host

    assert not any("AVX2" in warning for warning in host.warnings())


def test_unknown_memory_is_reported_as_unknown(monkeypatch):
    monkeypatch.setattr(host, "_host_memory_gb", lambda: None)
    monkeypatch.setattr(host, "_container_memory_gb", lambda: None)

    info = host.inspect()

    assert info["memory_gb"] is None
    assert info["container_memory_gb"] is None
    assert not any("RAM" in warning for warning in info["warnings"])


def test_host_memory_falls_back_to_proc_meminfo(monkeypatch):
    monkeypatch.setattr(host.os, "sysconf", lambda name: 0)

    memory_gb = host._host_memory_gb()

    if Path("/proc/meminfo").exists():
        assert memory_gb > 0
    else:
        assert memory_gb is None


def test_log_sizing_reports_without_failing(caplog):
    with caplog.at_level("INFO", logger="speechint.host"):
        host.log_sizing()

    assert "Dimensionierung" in caplog.text
    assert "Referenz: 6 Kerne" in caplog.text
