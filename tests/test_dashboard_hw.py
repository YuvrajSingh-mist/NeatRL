"""Tests for the dashboard's hardware counters.

Apple Silicon has no NVML equivalent, so GPU utilization used to be reported as
a hardcoded 0 and "VRAM" was the whole Python process RSS against total RAM.
These tests pin the macOS path to the values macOS actually exposes.
"""

from __future__ import annotations

import subprocess

import pytest

from neatrl.cli import dashboard


@pytest.fixture(autouse=True)
def _fresh_cache():
    """The ioreg probe is throttled; clear it between tests."""
    dashboard._appleGpuCache = (0.0, 0.0)
    yield
    dashboard._appleGpuCache = (0.0, 0.0)


def _fake_ioreg(output: str):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    return run


def test_apple_gpu_util_parses_the_metal_counter(monkeypatch):
    """ioreg publishes the driver's own utilization percentage."""
    monkeypatch.setattr(
        dashboard.subprocess,
        "run",
        _fake_ioreg(
            '  "PerformanceStatistics" = {"Device Utilization %"=73,'
            '"Renderer Utilization %"=70}'
        ),
    )
    assert dashboard._apple_gpu_util() == 73.0


def test_apple_gpu_util_returns_zero_when_ioreg_is_missing(monkeypatch):
    """A non-macOS box, or a denied ioreg, must not crash the dashboard."""

    def boom(*a, **k):
        raise FileNotFoundError("ioreg")

    monkeypatch.setattr(dashboard.subprocess, "run", boom)
    assert dashboard._apple_gpu_util() == 0.0


def test_apple_gpu_util_handles_unexpected_output(monkeypatch):
    """No AGX node (an Intel Mac) simply yields no GPU figure."""
    monkeypatch.setattr(dashboard.subprocess, "run", _fake_ioreg(""))
    assert dashboard._apple_gpu_util() == 0.0


def test_apple_gpu_util_is_throttled(monkeypatch):
    """The probe spawns a process, so it is cached for about a second."""
    calls = []

    def counting_run(*a, **k):
        calls.append(1)
        return subprocess.CompletedProcess(
            a, 0, stdout='"Device Utilization %"=5', stderr=""
        )

    monkeypatch.setattr(dashboard.subprocess, "run", counting_run)
    for _ in range(10):
        dashboard._apple_gpu_util()
    assert len(calls) == 1, "ioreg should be sampled once per cache window"


def test_mps_reports_gpu_and_driver_memory(monkeypatch):
    """On MPS, VRAM is the share of the MPS budget the driver has allocated."""
    import torch

    monkeypatch.setattr(dashboard, "DT", ("GPU", "VRAM", "mps"))
    monkeypatch.setattr(dashboard, "_apple_gpu_util", lambda: 42.0)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 10_000)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 2_500)
    monkeypatch.setattr(dashboard, "HAS_PSUTIL", False)

    _cpu, gpu, _ram, vram = dashboard.getHw()
    assert gpu == 42.0
    assert vram == pytest.approx(25.0)


def test_mps_survives_a_zero_budget(monkeypatch):
    """Never divide by zero when the MPS budget is unavailable."""
    import torch

    monkeypatch.setattr(dashboard, "DT", ("GPU", "VRAM", "mps"))
    monkeypatch.setattr(dashboard, "_apple_gpu_util", lambda: 10.0)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 0)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 5)
    monkeypatch.setattr(dashboard, "HAS_PSUTIL", False)

    _cpu, gpu, _ram, vram = dashboard.getHw()
    assert vram == 0.0
    assert gpu == 10.0


def test_ram_is_reported_as_ram_not_dram():
    """The header says RAM; DRAM is accurate but reads as jargon for the disk."""
    from neatrl.cli.dashboard import Dashboard

    frame = Dashboard("DQN", "CartPole-v1", 1000).renderDash(0, 0, {}, None, "")
    assert "RAM:" in frame
    assert "DRAM" not in frame


def test_mps_memory_is_reported_in_gigabytes(monkeypatch):
    """A percentage of PyTorch's soft budget read as an oddly small number."""
    import torch

    monkeypatch.setattr(dashboard, "DT", ("GPU", "MPS", "mps"))
    monkeypatch.setattr(
        torch.mps, "driver_allocated_memory", lambda: int(2.5 * 1024**3)
    )
    assert dashboard.gpuMemCell(18.4) == "MPS: [cyan]2.5G[/cyan]"


def test_cuda_memory_stays_a_percentage(monkeypatch):
    """On CUDA the percentage is of the card's real VRAM, so keep it."""
    monkeypatch.setattr(dashboard, "DT", ("GPU", "VRAM", "cuda"))
    assert dashboard.gpuMemCell(18.4) == "VRAM: [cyan]18.4%[/cyan]"


def test_mps_memory_survives_an_unavailable_counter(monkeypatch):
    import torch

    def boom():
        raise RuntimeError("MPS not initialised")

    monkeypatch.setattr(dashboard, "DT", ("GPU", "MPS", "mps"))
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", boom)
    assert "n/a" in dashboard.gpuMemCell(0.0)


def test_long_algorithm_names_are_not_cut_from_the_dashboard():
    """`Dueling-DQN` used to vanish: the header row had no room for it."""
    from neatrl.cli.dashboard import Dashboard

    for algo in ("DQN", "DQN-CNN", "Dueling-DQN", "Q-learning"):
        frame = Dashboard(algo, "ALE/Pong-v5", 1000).renderDash(0, 0, {}, None, "")
        assert algo in frame, f"{algo} missing from the dashboard"


def test_header_fits_an_80_column_terminal(monkeypatch):
    """The memory value is the last cell, so overflow silently ate it."""
    import re

    import torch

    from neatrl.cli.dashboard import Dashboard

    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 240 * 1024**2)
    frame = Dashboard("Dueling-DQN", "ALE/Pong-v5", 1000).renderDash(0, 0, {}, None, "")
    header = next(line for line in frame.splitlines() if "CPU" in line)

    assert len(header) <= 80, header
    assert header.endswith("\u2502"), header
    # The full value must survive, unit included.
    assert re.search(r"MPS: \d+(?:\.\d+)?[MG]\s*\u2502$", header), header


# --- non-Apple backends -----------------------------------------------------


@pytest.fixture()
def fresh_nvml(monkeypatch):
    """NVML state is cached per process; reset it so tests do not leak."""
    monkeypatch.setattr(dashboard, "_nvmlHandle", None)
    monkeypatch.setattr(dashboard, "_nvmlTried", False)
    yield


def test_nvml_is_initialised_once_per_process(monkeypatch, fresh_nvml):
    """Per-frame nvmlInit() was a driver call 5x a second; cache the handle."""
    calls = {"init": 0}

    class FakeNVML:
        @staticmethod
        def nvmlInit():
            calls["init"] += 1

        @staticmethod
        def nvmlDeviceGetHandleByIndex(i):
            return f"dev{i}"

        @staticmethod
        def nvmlDeviceGetUtilizationRates(handle):
            return type("Rates", (), {"gpu": 64})()

    monkeypatch.setitem(__import__("sys").modules, "pynvml", FakeNVML)

    assert dashboard._nvidia_gpu_util() == 64.0
    assert dashboard._nvidia_gpu_util() == 64.0
    assert calls["init"] == 1, "NVML should be initialised once, not per frame"


def test_a_missing_nvml_does_not_retry_every_frame(monkeypatch, fresh_nvml):
    """A box without the NVIDIA driver must not pay for that probe 5x/sec."""
    attempts = []

    class Boom:
        @staticmethod
        def nvmlInit():
            attempts.append(1)
            raise RuntimeError("libnvidia-ml.so not found")

    monkeypatch.setitem(__import__("sys").modules, "pynvml", Boom)

    for _ in range(5):
        assert dashboard._nvidia_gpu_util() == dashboard.GPU_UNKNOWN
    assert len(attempts) == 1


def test_rocm_falls_back_to_amdsmi(monkeypatch, fresh_nvml):
    """AMD cards on Linux expose no NVML; amdsmi is the ROCm equivalent."""
    import sys as _sys

    monkeypatch.setattr(dashboard, "DT", ("GPU", "VRAM", "cuda"))
    monkeypatch.setattr(dashboard, "_nvidia_gpu_util", lambda: dashboard.GPU_UNKNOWN)
    monkeypatch.setattr(dashboard, "_amd_gpu_util", lambda: 77.0)

    assert dashboard.gpuUtil() == 77.0
    assert "amdsmi" not in _sys.modules  # the stub is used, not the real package


def test_gpu_util_reports_unknown_when_no_backend_answers(monkeypatch, fresh_nvml):
    monkeypatch.setattr(dashboard, "DT", ("GPU", "VRAM", "cuda"))
    monkeypatch.setattr(dashboard, "_nvidia_gpu_util", lambda: dashboard.GPU_UNKNOWN)
    monkeypatch.setattr(dashboard, "_amd_gpu_util", lambda: dashboard.GPU_UNKNOWN)
    assert dashboard.gpuUtil() == dashboard.GPU_UNKNOWN


def test_apple_probe_is_skipped_off_darwin(monkeypatch):
    """ioreg does not exist on Linux or Windows; do not spawn it there."""
    ran = []
    monkeypatch.setattr(dashboard.sys, "platform", "linux")
    monkeypatch.setattr(dashboard.subprocess, "run", lambda *a, **k: ran.append(1))

    assert dashboard._apple_gpu_util() == 0.0
    assert not ran, "ioreg was shelled out on a non-Apple host"
