"""Factual multi-site Grid'5000 discovery and deterministic selection."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
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
DEFAULT_PERSISTENT_FREE_BYTES: Final[int] = 8 * 1024**3
DEFAULT_RESUME_PERSISTENT_FREE_BYTES: Final[int] = 512 * 1024**2
SUPPORTED_CPU_ARCHITECTURE: Final[str] = "x86_64"
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
    persistent_free_bytes: int = DEFAULT_PERSISTENT_FREE_BYTES
    resume_persistent_free_bytes: int = DEFAULT_RESUME_PERSISTENT_FREE_BYTES

    def __post_init__(self) -> None:
        _validate_site_requirements(self)

    def for_checkpoint_continuation(self) -> SiteRequirements:
        """Relax only storage headroom for a run with a verified checkpoint."""

        return replace(
            self,
            persistent_free_bytes=self.resume_persistent_free_bytes,
        )


def _validate_site_requirements(requirements: SiteRequirements) -> None:
    _require_positive_gpu_memory(requirements.gpu_memory_mb)
    _require_non_negative_capability(requirements.cuda_capability)
    _require_non_negative_storage(
        requirements.persistent_free_bytes,
        "persistent_free_bytes",
    )
    _require_non_negative_storage(
        requirements.resume_persistent_free_bytes,
        "resume_persistent_free_bytes",
    )
    if requirements.resume_persistent_free_bytes > requirements.persistent_free_bytes:
        raise Grid5000ConfigurationError(
            "resume_persistent_free_bytes cannot exceed persistent_free_bytes"
        )


def _require_positive_gpu_memory(value: int) -> None:
    if value <= 0:
        raise Grid5000ConfigurationError("gpu_memory_mb must be positive")


def _require_non_negative_capability(value: tuple[int, int]) -> None:
    if value < (0, 0):
        raise Grid5000ConfigurationError("cuda_capability must be non-negative")


def _require_non_negative_storage(value: int, name: str) -> None:
    if value < 0:
        raise Grid5000ConfigurationError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class GpuResource:
    """One complete Alive GPU resource record from ``oarnodes -J``."""

    gpu_memory_mb: int
    cuda_capability: tuple[int, int]
    jobs_assigned: int
    production: bool
    exotic: bool
    cpu_architecture: str = SUPPORTED_CPU_ARCHITECTURE


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
            and resource.cpu_architecture == SUPPORTED_CPU_ARCHITECTURE
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
                    "cpu_architecture": resource.cpu_architecture,
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
    return _coerce_non_negative_string(value) if isinstance(value, str) else None


def _coerce_non_negative_string(value: str) -> int | None:
    return int(value) if value.isdigit() else None


def _coerce_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).upper() == "YES"


def _coerce_cpu_architecture(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            return "x86_64" if normalized == "amd64" else normalized
    return "unknown"


def _coerce_jobs(value: object) -> int | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    if value is None:
        return 0
    return (
        _coerce_job_string(value)
        if isinstance(value, str)
        else _coerce_non_negative_int(value)
    )


def _coerce_job_string(value: str) -> int | None:
    normalized = value.strip()
    if not normalized or normalized.casefold() in {"no", "none"}:
        return 0
    if normalized == "0":
        return 0
    if normalized.isdigit():
        return 1
    return None


def _raw_records(payload: object) -> tuple[Mapping[str, object], ...]:
    records = _records_from_payload(payload)
    parsed: list[Mapping[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("oarnodes record must be a mapping")
        parsed.append(cast(Mapping[str, object], record))
    return tuple(parsed)


def _records_from_payload(payload: object) -> Sequence[object]:
    if isinstance(payload, Mapping):
        return (payload,) if "state" in payload else tuple(payload.values())
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return payload
    raise ValueError("oarnodes payload must be a mapping or sequence")


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
    return tuple(
        resource
        for record in _raw_records(payload)
        if (resource := _resource_from_record(record)) is not None
    )


def _resource_from_record(record: Mapping[str, object]) -> GpuResource | None:
    if str(record.get("state", "")) != "Alive":
        return None
    facts = _gpu_facts(record)
    if facts is None:
        return None
    gpu_memory, capability_major, jobs = facts
    capability_minor = _coerce_capability_minor(record)
    if capability_minor is None:
        return None
    return GpuResource(
        gpu_memory_mb=gpu_memory,
        cuda_capability=(capability_major, capability_minor),
        jobs_assigned=jobs,
        production=_coerce_flag(record.get("production", "NO")),
        exotic=_coerce_flag(record.get("exotic", "NO")),
        cpu_architecture=_coerce_cpu_architecture(record.get("cpuarch")),
    )


def _gpu_facts(record: Mapping[str, object]) -> tuple[int, int, int] | None:
    gpu_count = _coerce_non_negative_int(record.get("gpu_count"))
    gpu_memory = _coerce_non_negative_int(record.get("gpu_mem"))
    capability_major = _coerce_non_negative_int(
        record.get("gpu_compute_capability_major")
    )
    jobs_value = record.get("jobs")
    jobs = _coerce_jobs(jobs_value)
    if not _valid_gpu_count(gpu_count):
        return None
    if not _present_gpu_facts(gpu_memory, capability_major, jobs):
        return None
    return cast(tuple[int, int, int], (gpu_memory, capability_major, jobs))


def _valid_gpu_count(value: int | None) -> bool:
    return value is not None and value > 0


def _present_gpu_facts(
    gpu_memory: int | None,
    capability_major: int | None,
    jobs: int | None,
) -> bool:
    return gpu_memory is not None and capability_major is not None and jobs is not None


def choose_allocation(
    resources: Sequence[GpuResource],
    *,
    requirements: SiteRequirements | None = None,
) -> dict[str, str] | None:
    """Derive safe OAR queue/type/property values from live GPU facts."""

    effective = requirements or SiteRequirements()
    compatible = _compatible_resources(resources, effective)
    if not compatible:
        return None
    return _allocation_for_compatible(compatible, effective)


def _compatible_resources(
    resources: Sequence[GpuResource], requirements: SiteRequirements
) -> list[GpuResource]:
    return [
        resource
        for resource in resources
        if resource.gpu_memory_mb >= requirements.gpu_memory_mb
        and resource.cuda_capability >= requirements.cuda_capability
        and resource.cpu_architecture == SUPPORTED_CPU_ARCHITECTURE
    ]


def _allocation_for_compatible(
    compatible: list[GpuResource], requirements: SiteRequirements
) -> dict[str, str]:
    production = all(resource.production for resource in compatible)
    matching = _matching_resources(compatible, production)
    return _build_allocation(matching, requirements, production)


def _build_allocation(
    matching: list[GpuResource], requirements: SiteRequirements, production: bool
) -> dict[str, str]:
    exotic = all(resource.exotic for resource in matching)
    production_value = "YES" if production else "NO"
    capabilities = _capability_expression(matching)
    return {
        "queue": "production" if production else "default",
        "resource_type": "exotic" if exotic else "standard",
        "resource_property": (
            f"gpu_mem>={requirements.gpu_memory_mb} "
            f"AND production='{production_value}' "
            f"AND cpuarch='{SUPPORTED_CPU_ARCHITECTURE}' "
            f"AND gpu_compute_capability IN ({capabilities})"
        ),
    }


def _matching_resources(
    resources: list[GpuResource], production: bool
) -> list[GpuResource]:
    return [resource for resource in resources if resource.production == production]


def _capability_expression(resources: list[GpuResource]) -> str:
    return ", ".join(
        f"'{major}.{minor}'"
        for major, minor in sorted({resource.cuda_capability for resource in resources})
    )


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

    _validate_probe_arguments(sites, max_workers)
    effective_runner = runner or SubprocessCommandRunner()
    effective_requirements = requirements or SiteRequirements()
    unique_sites = tuple(dict.fromkeys(sites))
    return _probe_sites_concurrently(
        unique_sites,
        runner=effective_runner,
        requirements=effective_requirements,
        max_workers=max_workers,
    )


def _validate_probe_arguments(sites: Sequence[str], max_workers: int) -> None:
    if not sites:
        raise Grid5000ConfigurationError("at least one Grid'5000 site is required")
    if max_workers <= 0:
        raise Grid5000ConfigurationError("max_workers must be positive")


def _probe_sites_concurrently(
    sites: tuple[str, ...],
    *,
    runner: CommandRunner,
    requirements: SiteRequirements,
    max_workers: int,
) -> tuple[SiteProbe, ...]:
    with ThreadPoolExecutor(max_workers=min(max_workers, len(sites))) as pool:
        futures = tuple(
            pool.submit(
                probe_site,
                site,
                runner=runner,
                requirements=requirements,
            )
            for site in sites
        )
        return tuple(future.result() for future in futures)


def select_site(
    probes: Sequence[SiteProbe],
    *,
    requirements: SiteRequirements | None = None,
) -> SiteProbe:
    """Choose one compatible site from factual observations only."""

    effective_requirements = requirements or SiteRequirements()
    compatible = _compatible_probes(probes, effective_requirements)
    if not compatible:
        raise RuntimeError("no compatible Grid'5000 site is available")
    return min(
        compatible,
        key=lambda probe: (
            not probe.idle_compatible(effective_requirements),
            probe.name,
        ),
    )


def _compatible_probes(
    probes: Sequence[SiteProbe], requirements: SiteRequirements
) -> list[SiteProbe]:
    return [
        probe
        for probe in probes
        if probe.reachable
        and probe.has_compatible(requirements)
        and probe.persistent_free_bytes >= requirements.persistent_free_bytes
    ]


__all__ = [
    "DEFAULT_CUDA_CAPABILITY",
    "DEFAULT_GPU_MEMORY_MB",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_PERSISTENT_FREE_BYTES",
    "DEFAULT_RESUME_PERSISTENT_FREE_BYTES",
    "DEFAULT_SITES",
    "GpuResource",
    "SiteProbe",
    "SiteRequirements",
    "SUPPORTED_CPU_ARCHITECTURE",
    "choose_allocation",
    "parse_oarnodes_stdout",
    "probe_all_sites",
    "probe_site",
    "select_site",
]
