"""Pure site selection for short Grid'5000 replacement trials."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from .grid5000 import Grid5000Plan
from .grid5000_oar import JobStatus, OarClient, format_job_status, is_live_state
from .grid5000_policy import (
    SHORT_TRIAL_WALLTIME_SECONDS,
    ReplacementCandidate,
    attempt_immediate_replacement,
)
from .grid5000_sites import SiteProbe, SiteRequirements, choose_allocation


@dataclass(frozen=True, slots=True)
class ReplacementContext:
    """Dependencies and immutable settings for one replacement round."""

    run_id: str
    source_commit: str
    sites: tuple[str, ...]
    requirements: SiteRequirements
    max_workers: int
    cleanup: bool
    poll_seconds: float
    probe_sites: Callable[..., Sequence[SiteProbe]]
    remote_factory: Callable[[str], Any]
    build_plan: Callable[[SiteProbe, int], Grid5000Plan]
    submit_plan: Callable[[Any, Grid5000Plan], int]
    token_provider: Callable[[], str]
    emit: Callable[[str], None]
    sleep: Callable[[float], None]
    walltime_seconds: int = SHORT_TRIAL_WALLTIME_SECONDS


def replacement_candidates(
    probes: Sequence[SiteProbe],
    *,
    fallback_site: str,
    requirements: SiteRequirements,
) -> tuple[ReplacementCandidate, ...]:
    """Return compatible, idle, well-provisioned replacement candidates."""

    candidates: list[ReplacementCandidate] = []
    for probe in probes:
        candidate = _replacement_candidate(
            probe,
            fallback_site=fallback_site,
            requirements=requirements,
        )
        if candidate is not None:
            candidates.append(candidate)
    return tuple(sorted(candidates, key=lambda item: item.site))


def _replacement_candidate(
    probe: SiteProbe,
    *,
    fallback_site: str,
    requirements: SiteRequirements,
) -> ReplacementCandidate | None:
    if probe.name == fallback_site:
        return None
    if not _probe_can_replace(probe, requirements):
        return None
    allocation = choose_allocation(probe.resources, requirements=requirements)
    return (
        ReplacementCandidate(probe.name, allocation) if allocation is not None else None
    )


def _probe_can_replace(probe: SiteProbe, requirements: SiteRequirements) -> bool:
    return (
        probe.reachable
        and probe.idle_compatible(requirements)
        and probe.persistent_free_bytes >= requirements.persistent_free_bytes
    )


class ReplacementCoordinator:
    """Coordinate remote replacement trials without owning run state."""

    def __init__(self, context: ReplacementContext) -> None:
        self.context = context

    def _remote(self, site: str, remotes: dict[str, Any]) -> Any:
        remote = remotes.get(site)
        if remote is None:
            remote = self.context.remote_factory(site)
            remotes[site] = remote
        return remote

    def _client(
        self,
        site: str,
        *,
        remotes: dict[str, Any],
        clients: dict[str, OarClient],
    ) -> OarClient:
        client = clients.get(site)
        if client is None:
            client = OarClient(self._remote(site, remotes))
            clients[site] = client
        return client

    def _submit_candidate(
        self,
        candidate: ReplacementCandidate,
        *,
        probes: Sequence[SiteProbe],
        remotes: dict[str, Any],
    ) -> int:
        context = self.context
        remote = self._remote(candidate.site, remotes)
        context.emit(f"submitting a short replacement trial at {candidate.site}")
        remote.prepare(run_id=context.run_id, source_commit=context.source_commit)
        token = context.token_provider()
        if token:
            remote.install_hugging_face_token(token)
        probe = next(item for item in probes if item.name == candidate.site)
        plan = context.build_plan(probe, context.walltime_seconds)
        return context.submit_plan(remote, plan)

    def _status(
        self,
        site: str,
        job_id: int,
        *,
        remotes: dict[str, Any],
        clients: dict[str, OarClient],
    ) -> JobStatus:
        status = self._client(site, remotes=remotes, clients=clients).status(job_id)
        self.context.emit(
            f"{site} replacement job {job_id}: {format_job_status(status)}"
        )
        return status

    def _cancel(
        self,
        site: str,
        job_id: int,
        *,
        remotes: dict[str, Any],
        clients: dict[str, OarClient],
    ) -> None:
        remote = self._remote(site, remotes)
        client = self._client(site, remotes=remotes, clients=clients)
        client.cancel(job_id)
        self.context.emit(f"cancelled replacement job {job_id} at {site}")
        if not self.context.cleanup:
            return
        with suppress(Exception):
            if not is_live_state(client.status(job_id).state):
                remote.mark_status(self.context.run_id, "failed")
                remote.cleanup(self.context.run_id)

    def run(
        self,
        *,
        fallback_site: str,
        fallback_job_id: int,
        fallback_remote: Any,
    ) -> tuple[str, int, Any]:
        """Try all currently eligible sites and return the retained allocation."""

        context = self.context
        context.emit(
            f"run {context.run_id}: checking all {len(context.sites)} "
            "configured sites for replacement"
        )
        probes = tuple(
            context.probe_sites(
                sites=context.sites,
                requirements=context.requirements,
                max_workers=context.max_workers,
            )
        )
        candidates = replacement_candidates(
            probes,
            fallback_site=fallback_site,
            requirements=context.requirements,
        )
        candidate_names = (
            ", ".join(candidate.site for candidate in candidates) or "none"
        )
        context.emit(f"run {context.run_id}: replacement candidates: {candidate_names}")
        remotes: dict[str, Any] = {fallback_site: fallback_remote}
        clients: dict[str, OarClient] = {fallback_site: OarClient(fallback_remote)}
        outcome = attempt_immediate_replacement(
            fallback_site=fallback_site,
            fallback_job_id=fallback_job_id,
            candidates=candidates,
            submit=lambda candidate: self._submit_candidate(
                candidate,
                probes=probes,
                remotes=remotes,
            ),
            status=lambda site, job_id: self._status(
                site,
                job_id,
                remotes=remotes,
                clients=clients,
            ),
            cancel=lambda site, job_id: self._cancel(
                site,
                job_id,
                remotes=remotes,
                clients=clients,
            ),
            sleep=context.sleep,
            poll_seconds=context.poll_seconds,
        )
        return outcome.site, outcome.job_id, remotes.get(outcome.site, fallback_remote)


__all__ = [
    "ReplacementContext",
    "ReplacementCoordinator",
    "replacement_candidates",
]
