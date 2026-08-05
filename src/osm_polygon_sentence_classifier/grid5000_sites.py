"""Factual multi-site Grid'5000 discovery and deterministic selection."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final, cast

from .grid5000 import (
    _HOME_QUOTA_COMMAND,
    MINIMUM_CUDA_CAPABILITY,
    CommandResult,
    CommandRunner,
    Grid5000ConfigurationError,
    SubprocessCommandRunner,
    _ssh_argv,
    parse_quota_output,
)

DEFAULT_SITES: Final[tuple[str, ...]] = (
    "bordeaux",
    "grenoble",
    "lille",
    "louvain",
    "luxembourg",
    "lyon",
    "nancy",
    "nantes",
    "rennes",
    "sophia",
    "strasbourg",
    "toulouse",
)
DEFAULT_MAX_WORKERS: Final[int] = 4
DEFAULT_GPU_MEMORY_MB: Final[int] = 8_000
DEFAULT_CUDA_CAPABILITY: Final[tuple[int, int]] = MINIMUM_CUDA_CAPABILITY
_PROBE_TIMEOUT_SECONDS: Final[float] = 30.0
_OAR_NODES_COMMAND: Final[str] = "oarnodes -J"
_FREE_HOME_COMMAND: Final[str] = (
    "set -eu; value=$(df -Pk \"$HOME\" | awk 'NR==2 {print $4}'); "
    "case \"$value\" in ''|*[!0-9]*) exit 2;; esac; printf '%s\\n' \"$value\""
)
_QUEUE_DEPTH_COMMAND: Final[str] = (
    "set -eu; oarstat -u 2>/dev/null | awk '$5 ~ /Waiting|Hold/ {n++} END {print n+0}'"
)


@dataclass(frozen=True, slots=True)
class SiteRequirements:
    """Hard GPU and persistent-storage requirements for one run."""

    gpu_memory_mb: int = DEFAULT_GPU_MEMORY_MB
    cuda_capability: tuple[int, int] = DEFAULT_CUDA_CAPABILITY
    persistent_free_bytes: int = 8 * 1024**3

    def __post_init__(self) -> None:
        if self.gpu_memory_mb <= 0:
            raise Grid5000ConfigurationError("gpu_memory_mb must be positive")
        if self.cuda_capability < (0, 0):
            raise Grid5000ConfigurationError("cuda_capability must be non-negative")
        if self.persistent_free_bytes < 0:
            raise Grid5000ConfigurationError(
                "persistent_free_bytes must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class GpuResource:
    """One complete Alive GPU resource record from ``oarnodes -J``."""

    gpu_memory_mb: int
    cuda_capability: tuple[int, int]
    jobs_assigned: int
    production: bool
    exotic: bool


@dataclass(frozen=True, slots=True)
class SiteProbe:
    """Read-only observations for one Grid'5000 frontend."""

    name: str
    reachable: bool
    resources: tuple[GpuResource, ...]
    persistent_free_bytes: int
    queued_jobs: int
    error: str | None = None

    def compatible_resources(
        self, requirements: SiteRequirements
    ) -> tuple[GpuResource, ...]:
        """Return resources satisfying the hard GPU requirements."""

        return tuple(
            resource
            for resource in self.resources
            if resource.gpu_memory_mb >= requirements.gpu_memory_mb
            and resource.cuda_capability >= requirements.cuda_capability
        )

    def has_compatible(self, requirements: SiteRequirements) -> bool:
        """Return whether any observed resource satisfies the GPU contract."""

        return bool(self.compatible_resources(requirements))

    def idle_compatible(self, requirements: SiteRequirements) -> bool:
        """Return whether a compatible resource currently has no job assigned."""

        return any(
            resource.jobs_assigned == 0
            for resource in self.compatible_resources(requirements)
        )

    @property
    def allocation(self) -> dict[str, str] | None:
        """Return the scheduler resource choice inferred from compatible facts.

        This mirrors the reference project's submit helper: choose the default
        queue when a compatible non-production resource exists, otherwise the
        production queue; then choose standard unless every matching resource
        requires the exotic type.
        """

        return choose_allocation(self.resources)

    def to_dict(self, requirements: SiteRequirements) -> dict[str, object]:
        """Return a credential-free JSON representation for durable evidence."""

        return {
            "allocation": choose_allocation(self.resources, requirements=requirements),
            "cuda_capability": [
                list(resource.cuda_capability) for resource in self.resources
            ],
            "error": self.error,
            "has_compatible": self.has_compatible(requirements),
            "idle_compatible": self.idle_compatible(requirements),
            "name": self.name,
            "persistent_free_bytes": self.persistent_free_bytes,
            "queued_jobs": self.queued_jobs,
            "reachable": self.reachable,
            "resource_count": len(self.resources),
            "resources": [
                {
                    "cuda_capability": list(resource.cuda_capability),
                    "exotic": resource.exotic,
                    "gpu_memory_mb": resource.gpu_memory_mb,
                    "jobs_assigned": resource.jobs_assigned,
                    "production": resource.production,
                }
                for resource in self.resources
            ],
        }


def _coerce_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _coerce_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).upper() == "YES"


def _coerce_jobs(value: object) -> int | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    if value is None:
        return 0
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.casefold() in {"no", "none"}:
            return 0
        if normalized == "0":
            return 0
        if normalized.isdigit():
            return 1
    return _coerce_non_negative_int(value)


def _raw_records(payload: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(payload, Mapping):
        records = (payload,) if "state" in payload else tuple(payload.values())
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        records = payload
    else:
        raise ValueError("oarnodes payload must be a mapping or sequence")
    parsed: list[Mapping[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("oarnodes record must be a mapping")
        parsed.append(cast(Mapping[str, object], record))
    return tuple(parsed)


def _coerce_capability_minor(record: Mapping[str, object]) -> int | None:
    explicit = record.get("gpu_compute_capability_minor")
    if explicit is not None:
        return _coerce_non_negative_int(explicit)
    combined = record.get("gpu_compute_capability")
    if isinstance(combined, str) and "." in combined:
        _major, minor = combined.split(".", maxsplit=1)
        return _coerce_non_negative_int(minor)
    return 0


def parse_oarnodes_stdout(stdout: str) -> tuple[GpuResource, ...]:
    """Parse one JSON ``oarnodes -J`` response, failing closed on bad JSON."""

    if not isinstance(stdout, str):
        raise ValueError("oarnodes output must be text")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ValueError("invalid oarnodes JSON") from error
    resources: list[GpuResource] = []
    for record in _raw_records(payload):
        if str(record.get("state", "")) != "Alive":
            continue
        gpu_count = _coerce_non_negative_int(record.get("gpu_count"))
        gpu_memory = _coerce_non_negative_int(record.get("gpu_mem"))
        capability_major = _coerce_non_negative_int(
            record.get("gpu_compute_capability_major")
        )
        jobs = _coerce_jobs(record.get("jobs", 0))
        if (
            gpu_count is None
            or gpu_count <= 0
            or gpu_memory is None
            or capability_major is None
            or jobs is None
        ):
            continue
        capability_minor = _coerce_capability_minor(record)
        if capability_minor is None:
            continue
        resources.append(
            GpuResource(
                gpu_memory_mb=gpu_memory,
                cuda_capability=(capability_major, capability_minor),
                jobs_assigned=jobs,
                production=_coerce_flag(record.get("production", "NO")),
                exotic=_coerce_flag(record.get("exotic", "NO")),
            )
        )
    return tuple(resources)


def choose_allocation(
    resources: Sequence[GpuResource],
    *,
    requirements: SiteRequirements | None = None,
) -> dict[str, str] | None:
    """Derive safe OAR queue/type/property values from live GPU facts."""

    effective = requirements or SiteRequirements()
    compatible = [
        resource
        for resource in resources
        if resource.gpu_memory_mb >= effective.gpu_memory_mb
        and resource.cuda_capability >= effective.cuda_capability
    ]
    if not compatible:
        return None
    production = all(resource.production for resource in compatible)
    matching = [
        resource for resource in compatible if resource.production == production
    ]
    exotic = all(resource.exotic for resource in matching)
    production_value = "YES" if production else "NO"
    capabilities = ", ".join(
        f"'{major}.{minor}'"
        for major, minor in sorted({resource.cuda_capability for resource in matching})
    )
    return {
        "queue": "production" if production else "default",
        "resource_type": "exotic" if exotic else "standard",
        "resource_property": (
            f"gpu_mem>={effective.gpu_memory_mb} "
            f"AND production='{production_value}' "
            f"AND gpu_compute_capability IN ({capabilities})"
        ),
    }


def _parse_single_site(
    site: str,
    *,
    runner: CommandRunner,
    requirements: SiteRequirements,
) -> SiteProbe:
    def call(command: str) -> CommandResult:
        result = runner(
            _ssh_argv(site, command),
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"remote command failed with exit code {result.returncode}"
            )
        return result

    resources = parse_oarnodes_stdout(call(_OAR_NODES_COMMAND).stdout)
    quota = parse_quota_output(call(_HOME_QUOTA_COMMAND).stdout)
    free_kib = call(_FREE_HOME_COMMAND).stdout.strip()
    if not free_kib.isdigit():
        raise ValueError("home free-space output is invalid")
    queued_raw = call(_QUEUE_DEPTH_COMMAND).stdout.strip()
    queued_jobs = int(queued_raw) if queued_raw.isdigit() else 0
    return SiteProbe(
        name=site,
        reachable=True,
        resources=resources,
        persistent_free_bytes=min(
            int(free_kib) * 1024,
            quota.soft_headroom_bytes,
        ),
        queued_jobs=queued_jobs,
    )


def probe_site(
    site: str,
    *,
    runner: CommandRunner | None = None,
    requirements: SiteRequirements | None = None,
) -> SiteProbe:
    """Probe one frontend and turn failures into explicit unreachable facts."""

    effective_requirements = requirements or SiteRequirements()
    try:
        return _parse_single_site(
            site,
            runner=runner or SubprocessCommandRunner(),
            requirements=effective_requirements,
        )
    except Exception as error:
        return SiteProbe(
            name=site,
            reachable=False,
            resources=(),
            persistent_free_bytes=0,
            queued_jobs=0,
            error=str(error),
        )


def probe_all_sites(
    *,
    sites: Sequence[str] = DEFAULT_SITES,
    runner: CommandRunner | None = None,
    requirements: SiteRequirements | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> tuple[SiteProbe, ...]:
    """Probe configured frontends concurrently while preserving site order."""

    if not sites:
        raise Grid5000ConfigurationError("at least one Grid'5000 site is required")
    if max_workers <= 0:
        raise Grid5000ConfigurationError("max_workers must be positive")
    effective_runner = runner or SubprocessCommandRunner()
    effective_requirements = requirements or SiteRequirements()
    unique_sites = tuple(dict.fromkeys(sites))
    with ThreadPoolExecutor(max_workers=min(max_workers, len(unique_sites))) as pool:
        futures = tuple(
            pool.submit(
                probe_site,
                site,
                runner=effective_runner,
                requirements=effective_requirements,
            )
            for site in unique_sites
        )
        return tuple(future.result() for future in futures)


def select_site(
    probes: Sequence[SiteProbe],
    *,
    requirements: SiteRequirements | None = None,
) -> SiteProbe:
    """Choose one compatible site from factual observations only."""

    effective_requirements = requirements or SiteRequirements()
    compatible = [
        probe
        for probe in probes
        if probe.reachable
        and probe.has_compatible(effective_requirements)
        and probe.persistent_free_bytes >= effective_requirements.persistent_free_bytes
    ]
    if not compatible:
        raise RuntimeError("no compatible Grid'5000 site is available")
    return min(
        compatible,
        key=lambda probe: (
            not probe.idle_compatible(effective_requirements),
            probe.name,
        ),
    )


__all__ = [
    "DEFAULT_CUDA_CAPABILITY",
    "DEFAULT_GPU_MEMORY_MB",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_SITES",
    "GpuResource",
    "SiteProbe",
    "SiteRequirements",
    "choose_allocation",
    "parse_oarnodes_stdout",
    "probe_all_sites",
    "probe_site",
    "select_site",
]
