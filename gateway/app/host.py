"""Host sizing: what a speech endpoint must provide, and how this one compares.

SpeechInt is dimensioned for the smallest machine that is expected to host it:
**six CPU cores with AVX2 and 16 GB RAM**. That is the reference the thread
defaults, the container budgets and the estimates in the README are based on.

This module turns that assumption into something observable: :func:`inspect`
reports the cores and memory the process can actually see (cgroup-aware inside
Docker) plus whether AVX2 is available, :func:`warnings` names every deviation
and the startup hook logs them. Nothing here is fatal – an undersized endpoint
still works, it is just slower than documented – so a client can show the
administrator a hint instead of an error.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("speechint.host")

#: Minimum cores of a supported endpoint. Also the default thread count of both
#: model servers, because each one runs alone on the machine.
MIN_CORES = 6

#: Minimum RAM of a supported endpoint, in GB.
MIN_MEMORY_GB = 16.0

#: The AVX2 vector instructions whisper.cpp and llama.cpp use on x86-64. ARM
#: hosts (Apple Silicon, Graviton) use NEON instead and need no check.
REQUIRES_AVX2 = True

#: Steady-state footprint of each container on the reference machine, in GB.
#: Summed up in the README; kept here so the API can report the same numbers.
BUDGET_GB = {
    "gateway": 0.3,
    "whisper": 1.2,
    "llm": 3.0,
}

#: Values above this are "unlimited" rather than a real limit.
_UNLIMITED = 1 << 50

_CGROUP_LIMITS = (
    "/sys/fs/cgroup/memory.max",  # cgroup v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
)


def _container_memory_gb() -> Optional[float]:
    """Memory ceiling of this container, if one is set (cgroup v2 then v1)."""

    for path in _CGROUP_LIMITS:
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw == "" or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= _UNLIMITED:
            continue
        return value / (1024**3)
    return None


def _host_memory_gb() -> Optional[float]:
    """Physical memory of the machine the model servers run on.

    Deliberately not the cgroup limit: the gateway container is capped at a few
    hundred megabytes on purpose, while the question here is how much memory the
    whisper and llama.cpp containers may use.
    """

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    else:
        if pages > 0 and size > 0:
            return pages * size / (1024**3)

    # Docker without lxcfs reports the host total here; a fallback for the few
    # platforms where sysconf has no SC_PHYS_PAGES.
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024**2)
    except (OSError, IndexError, ValueError):
        pass
    return None


def _has_avx2() -> Optional[bool]:
    """``True``/``False`` on x86-64, ``None`` when the question does not apply."""

    if platform.machine().lower() not in ("x86_64", "amd64", "i386", "i686"):
        return None
    try:
        flags = Path("/proc/cpuinfo").read_text(errors="ignore")
    except OSError:
        return None
    return " avx2 " in f" {flags} "


def inspect() -> dict:
    """Cores, memory and AVX2 of this endpoint, plus the reference minimum."""

    cores = os.cpu_count() or 0
    memory_gb = _host_memory_gb()
    container_gb = _container_memory_gb()

    found = warnings()

    return {
        "cores": cores,
        "memory_gb": round(memory_gb, 1) if memory_gb else None,
        # Informational: the gateway itself is capped on purpose and that cap
        # says nothing about the machine the models run on.
        "container_memory_gb": round(container_gb, 2) if container_gb else None,
        "avx2": _has_avx2(),
        "architecture": platform.machine(),
        "minimum": {
            "cores": MIN_CORES,
            "memory_gb": MIN_MEMORY_GB,
            "avx2": REQUIRES_AVX2,
        },
        "budget_gb": dict(BUDGET_GB),
        "meets_minimum": not found,
        "warnings": found,
    }


def warnings() -> list:
    """Human-readable deviations from the reference machine, in German."""

    found = []

    cores = os.cpu_count() or 0
    if cores and cores < MIN_CORES:
        found.append(
            f"Nur {cores} CPU-Kerne verfügbar – dimensioniert ist dieser Dienst "
            f"für mindestens {MIN_CORES}."
        )

    memory_gb = _host_memory_gb()
    if memory_gb and memory_gb < MIN_MEMORY_GB:
        found.append(
            f"Nur {memory_gb:.1f} GB RAM verfügbar – dimensioniert ist dieser "
            f"Dienst für mindestens {MIN_MEMORY_GB:.0f} GB."
        )

    if _has_avx2() is False:
        found.append(
            "Die CPU meldet kein AVX2 – auf x86-64 werden Spracherkennung und "
            "Diktat dadurch deutlich langsamer."
        )

    return found


def log_sizing() -> None:
    """Log the reference values and any deviation once, at startup."""

    info = inspect()
    memory = f"{info['memory_gb']} GB" if info["memory_gb"] else "unbekannt"
    log.info(
        "Dimensionierung: %s Kerne, %s RAM, AVX2 %s. Referenz: %s Kerne, %.0f GB, AVX2.",
        info["cores"],
        memory,
        {True: "ja", False: "nein", None: "entfällt"}[info["avx2"]],
        MIN_CORES,
        MIN_MEMORY_GB,
    )
    for warning in info["warnings"]:
        log.warning("Dimensionierung: %s", warning)
