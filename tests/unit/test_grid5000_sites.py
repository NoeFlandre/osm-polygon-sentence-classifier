import json
from collections.abc import Callable, Sequence
from typing import cast

import pytest

import osm_polygon_sentence_classifier.grid5000_sites as grid5000_sites
from osm_polygon_sentence_classifier.grid5000 import (
    CommandResult,
    Grid5000ConfigurationError,
)
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
    jobs: object = 0,
    gpu_mem: object = 16_000,
    capability: int = 8,
    cpuarch: str = "x86_64",
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
        "cpuarch": cpuarch,
        "production": production,
        "exotic": exotic,
    }


def test_capability_minor_prefers_explicit_value_then_combined_value() -> None:
    assert (
        grid5000_sites._coerce_capability_minor(
            {"gpu_compute_capability_minor": 9, "gpu_compute_capability": "8.1"}
        )
        == 9
    )
    assert (
        grid5000_sites._coerce_capability_minor({"gpu_compute_capability_minor": "7"})
        == 7
    )
    assert (
        grid5000_sites._coerce_capability_minor({"gpu_compute_capability": "8.9"}) == 9
    )
    assert grid5000_sites._coerce_capability_minor({"gpu_compute_capability": "8"}) == 0
    assert grid5000_sites._coerce_capability_minor({}) == 0


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
    assert resources[0].cpu_architecture == "x86_64"
    assert not resources[0].production
    assert not resources[0].exotic


def test_site_selection_rejects_aarch64_until_the_remote_runtime_supports_it() -> None:
    probe = SiteProbe(
        name="lyon",
        reachable=True,
        resources=tuple(parse_oarnodes_stdout(_inventory(_record(cpuarch="aarch64")))),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )

    with pytest.raises(RuntimeError, match="compatible"):
        select_site((probe,), requirements=SiteRequirements())


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


def test_resource_parser_uses_stable_defaults_for_optional_record_fields() -> None:
    observed_defaults: dict[str, object] = {}

    class RecordingRecord(dict[str, object]):
        def get(  # ty: ignore[invalid-method-override]
            self, key: str, default: object | None = None
        ) -> object | None:
            if key in {"state", "production", "exotic"}:
                observed_defaults[key] = default
            return super().get(key, default)

    record = RecordingRecord(_record())
    resource = grid5000_sites._resource_from_record(record)

    assert resource is not None
    assert observed_defaults == {
        "state": "",
        "production": "NO",
        "exotic": "NO",
    }


@pytest.mark.parametrize("jobs", ["", " ", "no", "NO", "none", "NONE"])
def test_oarnodes_parser_treats_empty_and_no_job_markers_as_idle(
    jobs: str,
) -> None:
    resources = parse_oarnodes_stdout(_inventory(_record(jobs=jobs)))

    assert len(resources) == 1
    assert resources[0].jobs_assigned == 0


@pytest.mark.parametrize("stdout", ["not-json", ""])
def test_oarnodes_parser_rejects_non_json_responses(stdout: str) -> None:
    with pytest.raises(ValueError, match="^invalid oarnodes JSON$") as error:
        parse_oarnodes_stdout(stdout)
    assert str(error.value) == "invalid oarnodes JSON"


def test_oarnodes_parser_rejects_non_text_responses() -> None:
    with pytest.raises(ValueError, match="^oarnodes output must be text$") as error:
        parse_oarnodes_stdout(cast(str, 123))
    assert str(error.value) == "oarnodes output must be text"


def test_oarnodes_parser_accepts_a_single_record_mapping() -> None:
    resources = parse_oarnodes_stdout(json.dumps(_record()))

    assert len(resources) == 1
    assert resources[0].gpu_memory_mb == 16_000


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-a-sequence", "oarnodes payload must be a mapping or sequence"),
        ("[1]", "oarnodes record must be a mapping"),
    ],
)
def test_oarnodes_parser_rejects_invalid_record_shapes_exactly(
    payload: str, message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$") as error:
        parse_oarnodes_stdout(
            json.dumps(payload) if payload.startswith("n") else payload
        )

    assert str(error.value) == message


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" amd64 ", "x86_64"),
        ("X86_64", "x86_64"),
        ("aarch64", "aarch64"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_cpu_architecture_coercion_is_normalized_and_fails_closed(
    value: object, expected: str
) -> None:
    assert grid5000_sites._coerce_cpu_architecture(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        ([], 0),
        (["job-1", "job-2"], 2),
        (0, 0),
        (-1, None),
        (" 0 ", 0),
        ("1", 1),
        ("no", 0),
        ("NONE", 0),
        ("-1", None),
    ],
)
def test_job_count_coercion_preserves_scheduler_markers(
    value: object, expected: int | None
) -> None:
    assert grid5000_sites._coerce_jobs(value) == expected


def test_capability_minor_uses_the_first_separator_only() -> None:
    assert (
        grid5000_sites._coerce_capability_minor({"gpu_compute_capability": "8.9.10"})
        is None
    )


def test_gpu_facts_default_missing_jobs_to_idle() -> None:
    record = _record()
    del record["jobs"]

    assert grid5000_sites._gpu_facts(record) == (16_000, 8, 0)


def test_capability_expression_separates_distinct_capabilities() -> None:
    resources = [
        grid5000_sites.GpuResource(16_000, (8, 0), 0, False, False),
        grid5000_sites.GpuResource(16_000, (8, 9), 0, False, False),
    ]

    assert grid5000_sites._capability_expression(resources) == "'8.0', '8.9'"


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
            "gpu_mem>=8000 AND production='YES' AND cpuarch='x86_64' "
            "AND gpu_compute_capability IN ('8.0')"
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
            "gpu_mem>=8000 AND production='NO' AND cpuarch='x86_64' "
            "AND gpu_compute_capability IN ('8.0')"
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


def test_probe_all_sites_accepts_one_worker_without_probing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = (
        SiteProbe(
            name="nancy",
            reachable=False,
            resources=(),
            persistent_free_bytes=0,
            queued_jobs=0,
            error="stub",
        ),
    )
    observed: dict[str, object] = {}

    def runner(argv: Sequence[str], *, timeout: float) -> CommandResult:
        del argv, timeout
        return CommandResult(returncode=0)

    def probe_sites(
        sites: tuple[str, ...],
        *,
        runner: object,
        requirements: SiteRequirements,
        max_workers: int,
    ) -> tuple[SiteProbe, ...]:
        observed.update(
            sites=sites,
            runner=cast(grid5000_sites.CommandRunner, runner),
            requirements=requirements,
            max_workers=max_workers,
        )
        return expected

    monkeypatch.setattr(grid5000_sites, "_probe_sites_concurrently", probe_sites)

    requested_requirements = SiteRequirements(gpu_memory_mb=32_000)
    assert (
        probe_all_sites(
            sites=("nancy",),
            runner=cast(grid5000_sites.CommandRunner, runner),
            requirements=requested_requirements,
            max_workers=1,
        )
        == expected
    )
    assert observed["sites"] == ("nancy",)
    assert observed["runner"] is runner
    assert observed["requirements"] is requested_requirements
    assert observed["max_workers"] == 1


def test_probe_sites_concurrently_preserves_pool_bound_and_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    calls: list[tuple[str, object, SiteRequirements]] = []

    class ImmediateFuture:
        def __init__(self, value: SiteProbe) -> None:
            self.value = value

        def result(self) -> SiteProbe:
            return self.value

    class RecordingPool:
        def __init__(self, *, max_workers: int) -> None:
            observed["max_workers"] = max_workers

        def __enter__(self) -> "RecordingPool":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(
            self, function: object, *args: object, **kwargs: object
        ) -> ImmediateFuture:
            callable_function = cast(Callable[..., SiteProbe], function)
            probe = callable_function(*args, **kwargs)
            return ImmediateFuture(probe)

    requested_requirements = SiteRequirements(gpu_memory_mb=32_000)

    def probe_site(
        site: str,
        *,
        runner: object,
        requirements: SiteRequirements,
    ) -> SiteProbe:
        calls.append((site, runner, requirements))
        return SiteProbe(site, True, (), 0, 0)

    monkeypatch.setattr(grid5000_sites, "ThreadPoolExecutor", RecordingPool)
    monkeypatch.setattr(grid5000_sites, "probe_site", probe_site)
    runner = object()

    result = grid5000_sites._probe_sites_concurrently(
        ("nancy", "lille"),
        runner=cast(grid5000_sites.CommandRunner, runner),
        requirements=requested_requirements,
        max_workers=1,
    )

    assert observed["max_workers"] == 1
    assert calls == [
        ("nancy", runner, requested_requirements),
        ("lille", runner, requested_requirements),
    ]
    assert [probe.name for probe in result] == ["nancy", "lille"]


def test_probe_all_sites_reports_invalid_arguments_exactly() -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        probe_all_sites(sites=(), max_workers=1)
    assert str(error.value) == "at least one Grid'5000 site is required"

    with pytest.raises(Grid5000ConfigurationError) as error:
        probe_all_sites(sites=("nancy",), max_workers=0)
    assert str(error.value) == "max_workers must be positive"


def test_parse_single_site_builds_complete_facts_with_bounded_calls() -> None:
    outputs = iter(
        [
            CommandResult(returncode=0, stdout=_inventory(_record())),
            CommandResult(returncode=0, stdout="0 20000000 25000000\n"),
            CommandResult(returncode=0, stdout="1000\n"),
            CommandResult(returncode=0, stdout="not-a-number\n"),
        ]
    )
    calls: list[tuple[tuple[str, ...], float | None]] = []

    def runner(argv: Sequence[str], *, timeout: float) -> CommandResult:
        calls.append((tuple(argv), timeout))
        return next(outputs)

    probe = grid5000_sites._parse_single_site(
        "nancy",
        runner=runner,
        requirements=SiteRequirements(gpu_memory_mb=8_000),
    )

    assert probe.name == "nancy"
    assert probe.reachable is True
    assert probe.resources == tuple(parse_oarnodes_stdout(_inventory(_record())))
    assert probe.persistent_free_bytes == 1_024_000
    assert probe.queued_jobs == 0
    assert [argv[-1] for argv, _timeout in calls] == [
        grid5000_sites._OAR_NODES_COMMAND,
        grid5000_sites._HOME_QUOTA_COMMAND,
        grid5000_sites._FREE_HOME_COMMAND,
        grid5000_sites._QUEUE_DEPTH_COMMAND,
    ]
    assert [timeout for _argv, timeout in calls] == [
        grid5000_sites._PROBE_TIMEOUT_SECONDS
    ] * 4


def test_parse_single_site_preserves_remote_command_failure_message() -> None:
    def runner(_argv: Sequence[str], *, timeout: float) -> CommandResult:
        del timeout
        return CommandResult(returncode=7)

    with pytest.raises(RuntimeError) as error:
        grid5000_sites._parse_single_site(
            "nancy",
            runner=cast(grid5000_sites.CommandRunner, runner),
            requirements=SiteRequirements(),
        )

    assert str(error.value) == "remote command failed with exit code 7"


def test_parse_single_site_rejects_invalid_home_free_space_exactly() -> None:
    outputs = iter(
        [
            CommandResult(returncode=0, stdout=_inventory(_record())),
            CommandResult(returncode=0, stdout="0 20000000 25000000\n"),
            CommandResult(returncode=0, stdout="not-a-number\n"),
        ]
    )

    def runner(_argv: Sequence[str], *, timeout: float) -> CommandResult:
        del timeout
        return next(outputs)

    with pytest.raises(
        ValueError, match="^home free-space output is invalid$"
    ) as error:
        grid5000_sites._parse_single_site(
            "nancy",
            runner=cast(grid5000_sites.CommandRunner, runner),
            requirements=SiteRequirements(),
        )

    assert str(error.value) == "home free-space output is invalid"


def test_probe_site_forwards_explicit_runner_and_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_requirements = SiteRequirements(gpu_memory_mb=32_000)
    calls: list[tuple[str, object, SiteRequirements]] = []

    def runner(argv: Sequence[str], *, timeout: float) -> CommandResult:
        del argv, timeout
        return CommandResult(returncode=0)

    expected_probe = SiteProbe(
        name="nancy",
        reachable=True,
        resources=(),
        persistent_free_bytes=0,
        queued_jobs=0,
    )

    def parse_single_site(
        site: str, *, runner: object, requirements: SiteRequirements
    ) -> SiteProbe:
        calls.append((site, runner, requirements))
        return expected_probe

    monkeypatch.setattr(grid5000_sites, "_parse_single_site", parse_single_site)

    result = grid5000_sites.probe_site(
        "nancy", runner=runner, requirements=requested_requirements
    )

    assert result is expected_probe
    assert calls == [("nancy", runner, requested_requirements)]


def test_probe_site_returns_the_complete_unreachable_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def parse_single_site(
        site: str, *, runner: object, requirements: SiteRequirements
    ) -> SiteProbe:
        del site, runner, requirements
        raise RuntimeError("probe failed")

    monkeypatch.setattr(grid5000_sites, "_parse_single_site", parse_single_site)

    result = grid5000_sites.probe_site(
        "nancy", requirements=SiteRequirements(gpu_memory_mb=32_000)
    )

    assert result == SiteProbe(
        name="nancy",
        reachable=False,
        resources=(),
        persistent_free_bytes=0,
        queued_jobs=0,
        error="probe failed",
    )


def test_select_site_rejects_when_no_hard_compatible_site_exists() -> None:
    probe = SiteProbe(
        name="nancy",
        reachable=True,
        resources=tuple(parse_oarnodes_stdout(_inventory(_record(gpu_mem=4_000)))),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )

    with pytest.raises(RuntimeError) as error:
        select_site((probe,), requirements=SiteRequirements(gpu_memory_mb=8_000))
    assert str(error.value) == "no compatible Grid'5000 site is available"


def test_select_site_accepts_exact_storage_headroom() -> None:
    probe = SiteProbe(
        name="nancy",
        reachable=True,
        resources=tuple(parse_oarnodes_stdout(_inventory(_record()))),
        persistent_free_bytes=8 * 1024**3,
        queued_jobs=0,
    )

    assert select_site((probe,), requirements=SiteRequirements()).name == "nancy"


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


def test_site_requirements_report_storage_contract_errors_exactly() -> None:
    assert SiteRequirements(gpu_memory_mb=1).gpu_memory_mb == 1
    assert SiteRequirements(cuda_capability=(0, 0)).cuda_capability == (0, 0)
    assert SiteRequirements(cuda_capability=(0, 1)).cuda_capability == (0, 1)

    with pytest.raises(Grid5000ConfigurationError) as error:
        SiteRequirements(gpu_memory_mb=0)
    assert str(error.value) == "gpu_memory_mb must be positive"

    with pytest.raises(Grid5000ConfigurationError) as error:
        SiteRequirements(cuda_capability=(-1, 0))
    assert str(error.value) == "cuda_capability must be non-negative"

    with pytest.raises(Grid5000ConfigurationError) as error:
        SiteRequirements(cuda_capability=(0, -1))
    assert str(error.value) == "cuda_capability must be non-negative"

    with pytest.raises(Grid5000ConfigurationError) as error:
        SiteRequirements(persistent_free_bytes=-1)
    assert str(error.value) == "persistent_free_bytes must be non-negative"

    with pytest.raises(Grid5000ConfigurationError) as error:
        SiteRequirements(resume_persistent_free_bytes=-1)
    assert str(error.value) == "resume_persistent_free_bytes must be non-negative"

    with pytest.raises(Grid5000ConfigurationError) as error:
        SiteRequirements(persistent_free_bytes=10, resume_persistent_free_bytes=11)
    assert str(error.value) == (
        "resume_persistent_free_bytes cannot exceed persistent_free_bytes"
    )
