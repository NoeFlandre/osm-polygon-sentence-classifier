import json
from typing import cast

from osm_polygon_sentence_classifier.grid5000 import CommandResult, Grid5000Plan
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

    def prepare(self, **_: object) -> None:
        return None

    def install_hugging_face_token(self, _: str) -> None:
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
