import json
from typing import cast

import osm_polygon_sentence_classifier.grid5000_replacement as replacement_module
from osm_polygon_sentence_classifier.grid5000 import CommandResult, Grid5000Plan
from osm_polygon_sentence_classifier.grid5000_policy import ReplacementOutcome
from osm_polygon_sentence_classifier.grid5000_replacement import (
    ReplacementContext,
    ReplacementCoordinator,
    replacement_candidates,
)
from osm_polygon_sentence_classifier.grid5000_sites import (
    GpuResource,
    SiteProbe,
    SiteRequirements,
)


class _Remote:
    def __init__(self, state_by_job: dict[int, str]) -> None:
        self.state_by_job = state_by_job
        self.cancelled: list[int] = []
        self.prepared: list[dict[str, object]] = []
        self.tokens: list[str] = []
        self.marked: list[tuple[str, str]] = []
        self.cleaned: list[str] = []

    def prepare(self, **kwargs: object) -> None:
        self.prepared.append(kwargs)
        return None

    def install_hugging_face_token(self, token: str) -> None:
        self.tokens.append(token)
        return None

    def raw(self, command: str) -> CommandResult:
        job_id = int(command.split()[2])
        return CommandResult(
            0,
            json.dumps({str(job_id): {"state": self.state_by_job[job_id]}}),
        )

    def run(self, command: str) -> CommandResult:
        self.cancelled.append(int(command.split()[-1]))
        return CommandResult(0)

    def mark_status(self, run_id: str, status: str) -> None:
        self.marked.append((run_id, status))

    def cleanup(self, run_id: str) -> None:
        self.cleaned.append(run_id)


def test_replacement_coordinator_adopts_a_trial_that_starts_first() -> None:
    fallback = _Remote({10: "Waiting"})
    candidate_remote = _Remote({11: "Running"})
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    events: list[str] = []
    context = ReplacementContext(
        run_id="a" * 20,
        source_commit="b" * 40,
        sites=("nancy", "grenoble"),
        requirements=SiteRequirements(),
        max_workers=1,
        cleanup=False,
        poll_seconds=0.01,
        probe_sites=lambda **_: (probe,),
        remote_factory=lambda _: candidate_remote,
        build_plan=lambda *_: cast(Grid5000Plan, object()),
        submit_plan=lambda *_: 11,
        token_provider=lambda: "",
        emit=events.append,
        sleep=lambda _: None,
    )

    result = ReplacementCoordinator(context).run(
        fallback_site="nancy",
        fallback_job_id=10,
        fallback_remote=fallback,
    )

    assert result == ("grenoble", 11, candidate_remote)
    assert fallback.cancelled == [10]
    assert any("replacement candidates: grenoble" in event for event in events)


def test_replacement_candidates_keep_only_idle_compatible_sites() -> None:
    requirements = SiteRequirements()
    compatible = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    busy = SiteProbe(
        name="lille",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=1,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    low_headroom = SiteProbe(
        name="nancy",
        reachable=True,
        resources=compatible.resources,
        persistent_free_bytes=1 * 1024**3,
        queued_jobs=0,
    )

    candidates = replacement_candidates(
        (compatible, busy, low_headroom),
        fallback_site="bordeaux",
        requirements=requirements,
    )

    assert tuple(candidate.site for candidate in candidates) == ("grenoble",)


def _probe(
    name: str,
    *,
    memory: int = 16_000,
    capability: tuple[int, int] = (8, 0),
    jobs: int = 0,
    production: bool = True,
    exotic: bool = False,
    free_bytes: int = 10 * 1024**3,
) -> SiteProbe:
    return SiteProbe(
        name=name,
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=memory,
                cuda_capability=capability,
                jobs_assigned=jobs,
                production=production,
                exotic=exotic,
            ),
        ),
        persistent_free_bytes=free_bytes,
        queued_jobs=0,
    )


def test_replacement_candidates_are_sorted_and_use_the_requested_resources() -> None:
    requirements = SiteRequirements(
        gpu_memory_mb=20_000,
        cuda_capability=(9, 0),
        persistent_free_bytes=0,
        resume_persistent_free_bytes=0,
    )

    candidates = replacement_candidates(
        (
            _probe("toulouse", memory=20_000, capability=(9, 0)),
            _probe("bordeaux", memory=20_000, capability=(9, 0)),
        ),
        fallback_site="nancy",
        requirements=requirements,
    )

    assert tuple(candidate.site for candidate in candidates) == (
        "bordeaux",
        "toulouse",
    )
    assert all(
        "gpu_mem>=20000" in candidate.allocation["resource_property"]
        for candidate in candidates
    )


def test_replacement_candidates_accept_exact_storage_requirement() -> None:
    requirements = SiteRequirements(persistent_free_bytes=10 * 1024**3)

    candidates = replacement_candidates(
        (_probe("grenoble", free_bytes=10 * 1024**3),),
        fallback_site="nancy",
        requirements=requirements,
    )

    assert tuple(candidate.site for candidate in candidates) == ("grenoble",)


def test_replacement_candidates_reject_fallback_and_unusable_probes() -> None:
    requirements = SiteRequirements()
    probes = (
        _probe("nancy"),
        SiteProbe("unreachable", False, (), 10 * 1024**3, 0),
        _probe("busy", jobs=1),
        _probe("small", memory=1_000),
        _probe("old", free_bytes=1),
    )

    assert (
        replacement_candidates(
            probes,
            fallback_site="nancy",
            requirements=requirements,
        )
        == ()
    )


def test_coordinator_reuses_remote_and_client_and_passes_setup_inputs() -> None:
    fallback = _Remote({10: "Waiting"})
    candidate_remote = _Remote({11: "Running"})
    probe = _probe("grenoble")
    factory_calls: list[str] = []
    plans: list[tuple[SiteProbe, int]] = []
    submissions: list[tuple[object, object]] = []
    events: list[str] = []

    def remote_factory(site: str) -> _Remote:
        factory_calls.append(site)
        return candidate_remote

    def build_plan(item: SiteProbe, walltime: int) -> Grid5000Plan:
        plans.append((item, walltime))
        return cast(Grid5000Plan, object())

    def submit_plan(remote: object, plan: Grid5000Plan) -> int:
        submissions.append((remote, plan))
        return 11

    context = ReplacementContext(
        run_id="a" * 20,
        source_commit="b" * 40,
        sites=("nancy", "grenoble"),
        requirements=SiteRequirements(),
        max_workers=1,
        cleanup=False,
        poll_seconds=0.01,
        probe_sites=lambda **_: (probe,),
        remote_factory=remote_factory,
        build_plan=build_plan,
        submit_plan=submit_plan,
        token_provider=lambda: "token",
        emit=events.append,
        sleep=lambda _: None,
        walltime_seconds=123,
    )

    result = ReplacementCoordinator(context).run(
        fallback_site="nancy",
        fallback_job_id=10,
        fallback_remote=fallback,
    )

    assert result == ("grenoble", 11, candidate_remote)
    assert factory_calls == ["grenoble"]
    assert candidate_remote.prepared == [
        {"run_id": "a" * 20, "source_commit": "b" * 40}
    ]
    assert candidate_remote.tokens == ["token"]
    assert plans == [(probe, 123)]
    assert len(submissions) == 1
    assert submissions[0][0] is candidate_remote
    assert any("grenoble replacement job 11: running" in event for event in events)


def test_coordinator_keeps_fallback_when_no_candidate_is_eligible() -> None:
    fallback = _Remote({10: "Waiting"})
    factory_calls: list[str] = []
    probe_kwargs: list[dict[str, object]] = []
    events: list[str] = []

    def probe_sites(**kwargs: object) -> tuple[SiteProbe, ...]:
        probe_kwargs.append(kwargs)
        return ()

    context = ReplacementContext(
        run_id="a" * 20,
        source_commit="b" * 40,
        sites=("nancy",),
        requirements=SiteRequirements(),
        max_workers=1,
        cleanup=False,
        poll_seconds=0.01,
        probe_sites=probe_sites,
        remote_factory=lambda site: factory_calls.append(site) or _Remote({}),
        build_plan=lambda *_: cast(Grid5000Plan, object()),
        submit_plan=lambda *_: 11,
        token_provider=lambda: "",
        emit=events.append,
        sleep=lambda _: None,
    )

    assert ReplacementCoordinator(context).run(
        fallback_site="nancy",
        fallback_job_id=10,
        fallback_remote=fallback,
    ) == ("nancy", 10, fallback)
    assert factory_calls == []
    assert probe_kwargs == [
        {
            "sites": ("nancy",),
            "requirements": context.requirements,
            "max_workers": 1,
        }
    ]
    assert events == [
        "run aaaaaaaaaaaaaaaaaaaa: checking all 1 configured sites for replacement",
        "run aaaaaaaaaaaaaaaaaaaa: replacement candidates: none",
    ]


def test_coordinator_formats_multiple_candidate_sites(monkeypatch) -> None:
    fallback = _Remote({10: "Waiting"})
    events: list[str] = []

    def fake_attempt(**kwargs: object) -> ReplacementOutcome:
        return ReplacementOutcome("nancy", 10, False)

    monkeypatch.setattr(
        replacement_module, "attempt_immediate_replacement", fake_attempt
    )
    context = ReplacementContext(
        run_id="a" * 20,
        source_commit="b" * 40,
        sites=("nancy", "grenoble", "bordeaux"),
        requirements=SiteRequirements(),
        max_workers=1,
        cleanup=False,
        poll_seconds=0.25,
        probe_sites=lambda **_: (
            _probe("grenoble"),
            _probe("bordeaux"),
        ),
        remote_factory=lambda _: _Remote({}),
        build_plan=lambda *_: cast(Grid5000Plan, object()),
        submit_plan=lambda *_: 11,
        token_provider=lambda: "",
        emit=events.append,
        sleep=lambda _: None,
    )

    assert ReplacementCoordinator(context).run(
        fallback_site="nancy",
        fallback_job_id=10,
        fallback_remote=fallback,
    ) == ("nancy", 10, fallback)
    assert events[1] == (
        "run aaaaaaaaaaaaaaaaaaaa: replacement candidates: bordeaux, grenoble"
    )


def test_coordinator_forwards_replacement_timing_and_callbacks(monkeypatch) -> None:
    fallback = _Remote({10: "Waiting"})
    captured: dict[str, object] = {}

    def fake_attempt(**kwargs: object) -> ReplacementOutcome:
        captured.update(kwargs)
        return ReplacementOutcome("nancy", 10, False)

    monkeypatch.setattr(
        replacement_module, "attempt_immediate_replacement", fake_attempt
    )

    def sleep(_: float) -> None:
        return None

    context = ReplacementContext(
        run_id="a" * 20,
        source_commit="b" * 40,
        sites=("nancy",),
        requirements=SiteRequirements(),
        max_workers=1,
        cleanup=False,
        poll_seconds=0.25,
        probe_sites=lambda **_: (),
        remote_factory=lambda _: _Remote({}),
        build_plan=lambda *_: cast(Grid5000Plan, object()),
        submit_plan=lambda *_: 11,
        token_provider=lambda: "",
        emit=lambda _: None,
        sleep=sleep,
    )

    ReplacementCoordinator(context).run(
        fallback_site="nancy",
        fallback_job_id=10,
        fallback_remote=fallback,
    )

    assert captured["sleep"] is sleep
    assert captured["poll_seconds"] == 0.25
    assert callable(captured["submit"])
    assert callable(captured["status"])
    assert callable(captured["cancel"])


def test_coordinator_uses_fallback_remote_for_an_unexpected_outcome_site(
    monkeypatch,
) -> None:
    fallback = _Remote({10: "Waiting"})
    monkeypatch.setattr(
        replacement_module,
        "attempt_immediate_replacement",
        lambda **_: ReplacementOutcome("unexpected", 99, True),
    )
    context = ReplacementContext(
        run_id="a" * 20,
        source_commit="b" * 40,
        sites=("nancy",),
        requirements=SiteRequirements(),
        max_workers=1,
        cleanup=False,
        poll_seconds=0.01,
        probe_sites=lambda **_: (),
        remote_factory=lambda _: _Remote({}),
        build_plan=lambda *_: cast(Grid5000Plan, object()),
        submit_plan=lambda *_: 11,
        token_provider=lambda: "",
        emit=lambda _: None,
        sleep=lambda _: None,
    )

    assert ReplacementCoordinator(context).run(
        fallback_site="nancy",
        fallback_job_id=10,
        fallback_remote=fallback,
    ) == ("unexpected", 99, fallback)


def test_coordinator_reuses_the_cached_oar_client_for_a_site() -> None:
    remote = _Remote({11: "Waiting"})
    context = ReplacementContext(
        run_id="a" * 20,
        source_commit="b" * 40,
        sites=("grenoble",),
        requirements=SiteRequirements(),
        max_workers=1,
        cleanup=False,
        poll_seconds=0.01,
        probe_sites=lambda **_: (),
        remote_factory=lambda _: remote,
        build_plan=lambda *_: cast(Grid5000Plan, object()),
        submit_plan=lambda *_: 11,
        token_provider=lambda: "",
        emit=lambda _: None,
        sleep=lambda _: None,
    )
    coordinator = ReplacementCoordinator(context)
    remotes = {"grenoble": remote}
    clients = {}

    first = coordinator._client("grenoble", remotes=remotes, clients=clients)
    second = coordinator._client("grenoble", remotes=remotes, clients=clients)

    assert first is second
    assert clients["grenoble"] is first


def test_cancel_uses_the_cached_remote_and_swallows_cleanup_status_errors() -> None:
    class _FailingStatusRemote(_Remote):
        def raw(self, command: str) -> CommandResult:
            raise RuntimeError(command)

    remote = _FailingStatusRemote({11: "Terminated"})
    events: list[str] = []
    context = ReplacementContext(
        run_id="a" * 20,
        source_commit="b" * 40,
        sites=("grenoble",),
        requirements=SiteRequirements(),
        max_workers=1,
        cleanup=True,
        poll_seconds=0.01,
        probe_sites=lambda **_: (),
        remote_factory=lambda _: remote,
        build_plan=lambda *_: cast(Grid5000Plan, object()),
        submit_plan=lambda *_: 11,
        token_provider=lambda: "",
        emit=events.append,
        sleep=lambda _: None,
    )
    coordinator = ReplacementCoordinator(context)

    coordinator._cancel(
        "grenoble",
        11,
        remotes={"grenoble": remote},
        clients={},
    )

    assert remote.cancelled == [11]
    assert remote.marked == []
    assert remote.cleaned == []
    assert events == ["cancelled replacement job 11 at grenoble"]


def test_cancel_does_not_clean_a_live_trial() -> None:
    remote = _Remote({11: "Running"})
    context = ReplacementContext(
        run_id="a" * 20,
        source_commit="b" * 40,
        sites=("grenoble",),
        requirements=SiteRequirements(),
        max_workers=1,
        cleanup=True,
        poll_seconds=0.01,
        probe_sites=lambda **_: (),
        remote_factory=lambda _: remote,
        build_plan=lambda *_: cast(Grid5000Plan, object()),
        submit_plan=lambda *_: 11,
        token_provider=lambda: "",
        emit=lambda _: None,
        sleep=lambda _: None,
    )

    ReplacementCoordinator(context)._cancel(
        "grenoble",
        11,
        remotes={"grenoble": remote},
        clients={},
    )

    assert remote.marked == []
    assert remote.cleaned == []


def test_coordinator_cleans_a_cancelled_trial_when_enabled() -> None:
    fallback = _Remote({10: "Waiting"})
    candidate_remote = _Remote({11: "Terminated"})
    context = ReplacementContext(
        run_id="a" * 20,
        source_commit="b" * 40,
        sites=("nancy", "grenoble"),
        requirements=SiteRequirements(),
        max_workers=1,
        cleanup=True,
        poll_seconds=0.01,
        probe_sites=lambda **_: (_probe("grenoble"),),
        remote_factory=lambda _: candidate_remote,
        build_plan=lambda *_: cast(Grid5000Plan, object()),
        submit_plan=lambda *_: 11,
        token_provider=lambda: "",
        emit=lambda _: None,
        sleep=lambda _: None,
    )

    assert ReplacementCoordinator(context).run(
        fallback_site="nancy",
        fallback_job_id=10,
        fallback_remote=fallback,
    ) == ("nancy", 10, fallback)
    assert candidate_remote.cancelled == [11]
    assert candidate_remote.marked == [("a" * 20, "failed")]
    assert candidate_remote.cleaned == ["a" * 20]
