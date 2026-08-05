import json
from collections.abc import Sequence

import pytest

from osm_polygon_sentence_classifier.grid5000 import CommandResult
from osm_polygon_sentence_classifier.grid5000_sites import (
    DEFAULT_SITES,
    SiteProbe,
    SiteRequirements,
    choose_allocation,
    parse_oarnodes_stdout,
    probe_all_sites,
    select_site,
)


def _inventory(*records: dict[str, object]) -> str:
    return json.dumps(list(records))


def _record(
    *,
    jobs: int = 0,
    gpu_mem: object = 16_000,
    capability: int = 8,
    production: str = "NO",
    exotic: str = "NO",
    state: str = "Alive",
) -> dict[str, object]:
    return {
        "state": state,
        "gpu_count": 1,
        "jobs": jobs,
        "gpu_mem": gpu_mem,
        "gpu_compute_capability_major": capability,
        "production": production,
        "exotic": exotic,
    }


def test_oarnodes_parser_keeps_only_complete_alive_gpu_records() -> None:
    resources = parse_oarnodes_stdout(
        _inventory(
            _record(),
            _record(state="Dead"),
            {"state": "Alive", "gpu_count": 0},
            _record(gpu_mem="unknown"),
        )
    )

    assert len(resources) == 1
    assert resources[0].gpu_memory_mb == 16_000
    assert resources[0].cuda_capability == (8, 0)
    assert resources[0].jobs_assigned == 0
    assert not resources[0].production
    assert not resources[0].exotic


def test_oarnodes_parser_accepts_the_live_keyed_payload_shape() -> None:
    payload = {
        "7584": {
            "state": "Alive",
            "gpu_count": 8,
            "jobs": "2976576",
            "gpu_mem": 23_034,
            "gpu_compute_capability_major": 8,
            "gpu_compute_capability": "8.9",
            "production": "YES",
            "exotic": "NO",
        },
        "6804": {
            "state": "Alive",
            "gpu_count": 0,
            "jobs": "",
            "gpu_mem": 0,
            "gpu_compute_capability_major": 0,
        },
    }

    resources = parse_oarnodes_stdout(json.dumps(payload))

    assert len(resources) == 1
    assert resources[0].jobs_assigned == 1
    assert resources[0].cuda_capability == (8, 9)
    assert resources[0].gpu_memory_mb == 23_034


def test_choose_allocation_matches_reference_production_resource_rules() -> None:
    allocation = choose_allocation(
        parse_oarnodes_stdout(
            _inventory(
                _record(production="YES", exotic="NO"),
                _record(production="YES", exotic="NO", jobs=2),
            )
        ),
        requirements=SiteRequirements(gpu_memory_mb=8_000),
    )

    assert allocation == {
        "queue": "production",
        "resource_type": "standard",
        "resource_property": (
            "gpu_mem>=8000 AND production='YES' AND gpu_compute_capability IN ('8.0')"
        ),
    }


def test_choose_allocation_falls_back_to_default_exotic_resources() -> None:
    allocation = choose_allocation(
        parse_oarnodes_stdout(_inventory(_record(production="NO", exotic="YES"))),
        requirements=SiteRequirements(gpu_memory_mb=8_000),
    )

    assert allocation == {
        "queue": "default",
        "resource_type": "exotic",
        "resource_property": (
            "gpu_mem>=8000 AND production='NO' AND gpu_compute_capability IN ('8.0')"
        ),
    }


def test_choose_allocation_excludes_observed_incompatible_gpu_capabilities() -> None:
    resources = parse_oarnodes_stdout(
        _inventory(
            _record(capability=6),
            _record(capability=8),
        )
    )

    allocation = choose_allocation(resources)

    assert allocation is not None
    assert allocation["resource_property"].endswith("gpu_compute_capability IN ('8.0')")


def test_site_selection_prefers_idle_then_name_without_using_queue_depth() -> None:
    requirements = SiteRequirements(gpu_memory_mb=8_000)
    busy = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=tuple(parse_oarnodes_stdout(_inventory(_record(jobs=1)))),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    idle = SiteProbe(
        name="nancy",
        reachable=True,
        resources=tuple(parse_oarnodes_stdout(_inventory(_record()))),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=999,
    )

    selected = select_site((busy, idle), requirements=requirements)

    assert selected.name == "nancy"


def test_probe_all_sites_is_bounded_and_returns_unreachable_facts() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(argv: Sequence[str], *, timeout: float) -> CommandResult:
        del timeout
        calls.append(tuple(argv))
        command = argv[-1]
        if command == "oarnodes -J":
            return CommandResult(
                returncode=0,
                stdout=_inventory(_record(production="YES")),
            )
        if "quota" in command:
            return CommandResult(returncode=0, stdout="0 20000000 25000000\n")
        if "df -Pk" in command:
            return CommandResult(returncode=0, stdout="100000\n")
        if "oarstat" in command:
            return CommandResult(returncode=0, stdout="3\n")
        raise AssertionError(command)

    probes = probe_all_sites(
        sites=DEFAULT_SITES[:2],
        runner=runner,
        requirements=SiteRequirements(gpu_memory_mb=8_000),
        max_workers=2,
    )

    assert [probe.name for probe in probes] == list(DEFAULT_SITES[:2])
    assert all(probe.reachable for probe in probes)
    assert all(probe.queued_jobs == 3 for probe in probes)
    assert len(calls) == 8


def test_select_site_rejects_when_no_hard_compatible_site_exists() -> None:
    probe = SiteProbe(
        name="nancy",
        reachable=True,
        resources=tuple(parse_oarnodes_stdout(_inventory(_record(gpu_mem=4_000)))),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )

    with pytest.raises(RuntimeError, match="compatible"):
        select_site((probe,), requirements=SiteRequirements(gpu_memory_mb=8_000))


def test_default_requirements_reject_a_gpu_below_the_torch_capability_floor() -> None:
    probe = SiteProbe(
        name="lille",
        reachable=True,
        resources=tuple(parse_oarnodes_stdout(_inventory(_record(capability=7)))),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )

    with pytest.raises(RuntimeError, match="compatible"):
        select_site((probe,), requirements=SiteRequirements())


def test_site_probe_evidence_uses_the_requested_requirements() -> None:
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=tuple(parse_oarnodes_stdout(_inventory(_record(gpu_mem=16_000)))),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )

    evidence = probe.to_dict(SiteRequirements(gpu_memory_mb=32_000))

    assert evidence["has_compatible"] is False
    assert evidence["idle_compatible"] is False
    assert evidence["allocation"] is None


def test_default_requirements_reserve_space_for_multiple_checkpoints() -> None:
    requirements = SiteRequirements()

    assert requirements.persistent_free_bytes >= 8 * 1024**3
