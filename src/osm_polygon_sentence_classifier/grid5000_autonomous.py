"""One-command autonomous Grid'5000 training lifecycle."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .config import ProjectConfig
from .grid5000 import (
    MINIMUM_HOME_HEADROOM_BYTES,
    CommandRunner,
    Grid5000Allocation,
    Grid5000Plan,
    Grid5000RunIdentity,
    parse_quota_output,
)
from .grid5000_oar import JobState, JobStatus, OarClient, is_live_state
from .grid5000_policy import (
    SHORT_TRIAL_WALLTIME_SECONDS,
    ReplacementCandidate,
    attempt_immediate_replacement,
    policy_type_for,
    should_seek_replacement,
)
from .grid5000_remote import Grid5000Remote
from .grid5000_sites import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_SITES,
    SiteProbe,
    SiteRequirements,
    choose_allocation,
    probe_all_sites,
    select_site,
)
from .grid5000_state import (
    AutonomousRunState,
    AutonomousStateStore,
    LegacyAmbiguousStateError,
    RunPhase,
)
from .publication import ensure_model_repository
from .tracking import ensure_trackio_resources, settings_for
from .training import TrainingConfig

PolicyType = Literal["auto", "day", "night"]


class AutonomousRunError(RuntimeError):
    """The autonomous lifecycle cannot continue safely."""


@dataclass(frozen=True, slots=True)
class AutonomousRunConfig:
    """Immutable settings for one reproducible autonomous run."""

    identity: Grid5000RunIdentity
    training_config: TrainingConfig
    sites: tuple[str, ...] = DEFAULT_SITES
    requirements: SiteRequirements = SiteRequirements()
    walltime_seconds: int = 30 * 60
    policy_type: PolicyType = "auto"
    max_workers: int = DEFAULT_MAX_WORKERS
    cleanup: bool = True

    def __post_init__(self) -> None:
        if not self.sites:
            raise AutonomousRunError("at least one Grid'5000 site is required")
        if self.walltime_seconds <= 0:
            raise AutonomousRunError("walltime_seconds must be positive")
        if self.policy_type not in {"auto", "day", "night"}:
            raise AutonomousRunError("policy_type must be auto, day, or night")
        if self.max_workers <= 0:
            raise AutonomousRunError("max_workers must be positive")
        if self.training_config.model_name_or_path != self.identity.model_name_or_path:
            raise AutonomousRunError("training model does not match run identity")
        if self.training_config.model_revision != self.identity.model_revision:
            raise AutonomousRunError("training revision does not match run identity")


def _local_hugging_face_token(environ: Mapping[str, str] | None = None) -> str:
    environment = os.environ if environ is None else environ
    direct = environment.get("HF_TOKEN", "").strip()
    if direct:
        return direct
    token_path = (
        Path(environment.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        / "token"
    )
    try:
        return token_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _now() -> datetime:
    return datetime.now().astimezone()


class AutonomousRunController:
    """Prepare, submit, monitor, publish, verify, and clean one run."""

    def __init__(
        self,
        config: AutonomousRunConfig,
        *,
        state_root: Path | None = None,
        probe_sites: Callable[..., Sequence[SiteProbe]] = probe_all_sites,
        remote_factory: Callable[[str], Any] = Grid5000Remote,
        hub_api: Any | None = None,
        runner: CommandRunner | None = None,
        poll_seconds: float = 30.0,
        sleeper: Callable[[float], None] = time.sleep,
        emit: Callable[[str], None] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if poll_seconds < 0:
            raise AutonomousRunError("poll_seconds must be non-negative")
        self.config = config
        self.state = AutonomousStateStore(state_root)
        self.probe_sites = probe_sites
        self.remote_factory = remote_factory
        self.hub_api = hub_api
        self.runner = runner
        self.poll_seconds = poll_seconds
        self.sleeper = sleeper
        self.emit = emit or (lambda _message: None)
        self.environ = environ
        self._active_remote: Any | None = None

    def _transition(
        self,
        current: AutonomousRunState,
        phase: RunPhase,
        *,
        site: str | None = None,
        job_id: int | None = None,
        facts: Mapping[str, object] | None = None,
    ) -> AutonomousRunState:
        merged = dict(current.facts or {})
        merged.update(facts or {})
        updated = AutonomousRunState(
            run_id=current.run_id,
            phase=phase,
            identity=current.identity,
            site=site if site is not None else current.site,
            job_id=job_id if job_id is not None else current.job_id,
            facts=merged,
        )
        self.state.save(updated)
        self.state.append_event(current.run_id, phase.value, facts or {})
        return updated

    def _hub(self) -> Any:
        if self.hub_api is not None:
            return self.hub_api
        try:
            import huggingface_hub

            self.hub_api = huggingface_hub.HfApi()
            return self.hub_api
        except Exception as error:
            raise AutonomousRunError(
                "Hugging Face dependencies are required for publication"
            ) from error

    def _provision_hub(self) -> None:
        if self.config.training_config.publish_to_hub:
            ensure_model_repository(
                ProjectConfig().target_model_repository_id,
                hub_api=self._hub(),
            )
        if self.config.training_config.sync_trackio:
            ensure_trackio_resources(
                settings_for(ProjectConfig()),
                hub_api=self._hub(),
            )

    def _local_token_for_publication(self) -> str:
        token = _local_hugging_face_token(self.environ)
        if (
            self.config.training_config.publish_to_hub
            or self.config.training_config.sync_trackio
        ) and not token:
            raise AutonomousRunError(
                "HF authentication is required for requested publication"
            )
        return token

    def _build_plan(
        self,
        probe: SiteProbe,
        *,
        walltime_seconds: int | None = None,
        now: datetime | None = None,
    ) -> Grid5000Plan:
        allocation_facts = choose_allocation(
            probe.resources,
            requirements=self.config.requirements,
        )
        if allocation_facts is None:
            raise AutonomousRunError(
                f"site {probe.name} has no compatible GPU allocation"
            )
        walltime = walltime_seconds or self.config.walltime_seconds
        policy = self.config.policy_type
        if policy == "auto":
            policy = policy_type_for(
                now or _now(),
                walltime_seconds=walltime,
            )
        allocation = Grid5000Allocation(
            site=probe.name,
            walltime_seconds=walltime,
            queue=allocation_facts["queue"],
            resource_type=allocation_facts["resource_type"],
            resource_property=allocation_facts["resource_property"],
            policy_type=policy,
        )
        return Grid5000Plan(
            identity=self.config.identity,
            allocation=allocation,
        )

    def _preflight(self, remote: Any, plan: Grid5000Plan) -> None:
        for command, _label in (
            (plan.remote_checkout_command[-1], "remote checkout"),
            (plan.policy_site_command[-1], "site policy"),
            (plan.policy_total_command[-1], "total policy"),
        ):
            remote.run(command)
        quota_result = remote.run(plan.quota_command[-1])
        quota = parse_quota_output(quota_result.stdout)
        if quota.soft_headroom_bytes < MINIMUM_HOME_HEADROOM_BYTES:
            raise AutonomousRunError(
                "Grid'5000 home soft quota has insufficient safe headroom"
            )

    def _submit_plan(self, remote: Any, plan: Grid5000Plan) -> int:
        self._preflight(remote, plan)
        return OarClient(remote).submit(plan.scheduler_command)

    def _verify_completion(self, remote: Any, manifest: Mapping[str, object]) -> None:
        training = self.config.training_config
        expected_identity = {
            "run_id": self.config.identity.run_id,
            "source_commit": self.config.identity.source_commit,
            "dataset_revision": self.config.identity.dataset_revision,
            "model_name_or_path": self.config.identity.model_name_or_path,
            "model_revision": self.config.identity.model_revision,
        }
        for field, expected in expected_identity.items():
            if manifest.get(field) != expected:
                raise AutonomousRunError(f"completion manifest {field} is invalid")
        publication = manifest.get("model_publication")
        if training.publish_to_hub:
            if not isinstance(publication, Mapping):
                raise AutonomousRunError(
                    "completed worker did not report a model publication"
                )
            repository = publication.get("repository_id")
            commit_id = publication.get("commit_id")
            if (
                not isinstance(repository, str)
                or not isinstance(commit_id, str)
                or len(commit_id) != 40
            ):
                raise AutonomousRunError("model publication facts are invalid")
            if repository != ProjectConfig().target_model_repository_id:
                raise AutonomousRunError("model publication repository is invalid")
            api = self._hub()
            api.model_info(repo_id=repository, revision=commit_id)
        if training.sync_trackio:
            tracking_space = manifest.get("tracking_space_id")
            settings = settings_for(ProjectConfig())
            if tracking_space != settings.space_id:
                raise AutonomousRunError(
                    "completed worker reported the wrong Trackio Space"
                )
            self._hub().space_info(repo_id=settings.space_id)

    def _candidate_list(
        self,
        probes: Sequence[SiteProbe],
        *,
        fallback_site: str,
    ) -> tuple[ReplacementCandidate, ...]:
        candidates: list[ReplacementCandidate] = []
        for probe in probes:
            if probe.name == fallback_site:
                continue
            if (
                not probe.reachable
                or not probe.idle_compatible(self.config.requirements)
                or probe.persistent_free_bytes
                < self.config.requirements.persistent_free_bytes
            ):
                continue
            allocation = choose_allocation(
                probe.resources,
                requirements=self.config.requirements,
            )
            if allocation is not None:
                candidates.append(ReplacementCandidate(probe.name, allocation))
        return tuple(sorted(candidates, key=lambda item: item.site))

    def _replacement_raw_remote(
        self,
        site: str,
        raw_remotes: dict[str, Any],
    ) -> Any:
        remote = raw_remotes.get(site)
        if remote is None:
            remote = self.remote_factory(site)
            raw_remotes[site] = remote
        return remote

    def _replacement_client(
        self,
        site: str,
        *,
        raw_remotes: dict[str, Any],
        clients: dict[str, Any],
    ) -> Any:
        client = clients.get(site)
        if client is None:
            client = OarClient(
                self._replacement_raw_remote(site, raw_remotes),
            )
            clients[site] = client
        return client

    def _submit_replacement_candidate(
        self,
        candidate: ReplacementCandidate,
        *,
        probes: Sequence[SiteProbe],
        raw_remotes: dict[str, Any],
    ) -> int:
        remote = self._replacement_raw_remote(candidate.site, raw_remotes)
        remote.prepare(
            run_id=self.config.identity.run_id,
            source_commit=self.config.identity.source_commit,
        )
        token = self._local_token_for_publication()
        if token:
            remote.install_hugging_face_token(token)
        probe = next(item for item in probes if item.name == candidate.site)
        plan = self._build_plan(
            probe,
            walltime_seconds=SHORT_TRIAL_WALLTIME_SECONDS,
        )
        return self._submit_plan(remote, plan)

    def _replacement_status(
        self,
        site: str,
        job_id: int,
        *,
        raw_remotes: dict[str, Any],
        clients: dict[str, Any],
    ) -> JobStatus:
        return self._replacement_client(
            site,
            raw_remotes=raw_remotes,
            clients=clients,
        ).status(job_id)

    def _cancel_replacement(
        self,
        site: str,
        job_id: int,
        *,
        raw_remotes: dict[str, Any],
        clients: dict[str, Any],
    ) -> None:
        remote = self._replacement_raw_remote(site, raw_remotes)
        client = self._replacement_client(
            site,
            raw_remotes=raw_remotes,
            clients=clients,
        )
        client.cancel(job_id)
        if not self.config.cleanup:
            return
        with suppress(Exception):
            if not is_live_state(client.status(job_id).state):
                remote.mark_status(self.config.identity.run_id, "failed")
                remote.cleanup(self.config.identity.run_id)

    def _try_replacement(
        self,
        *,
        fallback_site: str,
        fallback_job_id: int,
        fallback_remote: Any,
    ) -> tuple[str, int, Any]:
        probes = self.probe_sites(
            sites=self.config.sites,
            requirements=self.config.requirements,
            max_workers=self.config.max_workers,
        )
        candidates = self._candidate_list(probes, fallback_site=fallback_site)
        raw_remotes: dict[str, Any] = {fallback_site: fallback_remote}
        clients: dict[str, Any] = {fallback_site: OarClient(fallback_remote)}

        outcome = attempt_immediate_replacement(
            fallback_site=fallback_site,
            fallback_job_id=fallback_job_id,
            candidates=candidates,
            submit=lambda candidate: self._submit_replacement_candidate(
                candidate,
                probes=probes,
                raw_remotes=raw_remotes,
            ),
            status=lambda site, job_id: self._replacement_status(
                site,
                job_id,
                raw_remotes=raw_remotes,
                clients=clients,
            ),
            cancel=lambda site, job_id: self._cancel_replacement(
                site,
                job_id,
                raw_remotes=raw_remotes,
                clients=clients,
            ),
            sleep=self.sleeper,
            poll_seconds=self.poll_seconds or 1.0,
        )
        raw_remote = raw_remotes.get(outcome.site, fallback_remote)
        return outcome.site, outcome.job_id, raw_remote

    def _handle_queued_status(
        self,
        current: AutonomousRunState,
        *,
        status: JobStatus,
        site: str,
        job_id: int,
        remote: Any,
        replacement_attempted: bool,
    ) -> tuple[AutonomousRunState, str, int, Any, bool]:
        if current.phase is RunPhase.SUBMITTED:
            current = self._transition(current, RunPhase.QUEUED)
        if replacement_attempted or not should_seek_replacement(status, now=_now()):
            return current, site, job_id, remote, replacement_attempted
        current = self._transition(
            current,
            RunPhase.QUEUED,
            facts={"replacement_attempted": True},
        )
        replacement_site, replacement_job, replacement_remote = self._try_replacement(
            fallback_site=site,
            fallback_job_id=job_id,
            fallback_remote=remote,
        )
        current = self._transition(
            current,
            RunPhase.QUEUED,
            site=replacement_site,
            job_id=replacement_job,
            facts={"replacement_attempted": True},
        )
        return current, replacement_site, replacement_job, replacement_remote, True

    def _handle_terminal_status(
        self,
        current: AutonomousRunState,
        *,
        status: JobStatus,
        site: str,
        job_id: int,
        remote: Any,
    ) -> AutonomousRunState:
        if status.state is not JobState.TERMINATED or status.exit_code not in {None, 0}:
            current = self._transition(
                current,
                RunPhase.FAILED,
                site=site,
                job_id=job_id,
                facts={
                    "error": (
                        f"job ended as {status.state.value}; "
                        f"exit_code={status.exit_code}"
                    )
                },
            )
            remote.mark_status(self.config.identity.run_id, "failed")
            raise AutonomousRunError(
                str(dict(current.facts or {}).get("error", "job failed"))
            )
        try:
            manifest = remote.read_completion(self.config.identity.run_id)
            self._verify_completion(remote, manifest)
        except Exception as error:
            current = self._transition(
                current,
                RunPhase.FAILED,
                site=site,
                job_id=job_id,
                facts={"error": str(error)},
            )
            with suppress(Exception):
                remote.mark_status(self.config.identity.run_id, "failed")
            if isinstance(error, AutonomousRunError):
                raise
            raise AutonomousRunError(
                "completed Grid'5000 job could not be verified"
            ) from error
        remote.mark_status(self.config.identity.run_id, "complete")
        if self.config.cleanup:
            remote.cleanup(self.config.identity.run_id)
        return self._transition(
            current,
            RunPhase.COMPLETED,
            site=site,
            job_id=job_id,
            facts={"completion": manifest, "cleanup": self.config.cleanup},
        )

    def _monitor(
        self,
        state: AutonomousRunState,
        *,
        remote: Any,
    ) -> AutonomousRunState:
        if state.site is None or state.job_id is None:
            raise AutonomousRunError("submitted state lacks site or job ID")
        site = state.site
        job_id = state.job_id
        current_remote = remote
        oar = OarClient(current_remote)
        replacement_attempted = bool(
            dict(state.facts or {}).get("replacement_attempted", False)
        )
        current = state
        while True:
            status = oar.status(job_id)
            self.emit(f"{site} job {job_id}: {status.state.value}")
            if status.state is JobState.QUEUED:
                (
                    current,
                    site,
                    job_id,
                    current_remote,
                    replacement_attempted,
                ) = self._handle_queued_status(
                    current,
                    status=status,
                    site=site,
                    job_id=job_id,
                    remote=current_remote,
                    replacement_attempted=replacement_attempted,
                )
                oar = OarClient(current_remote)
            elif status.state is JobState.RUNNING:
                if current.phase in {RunPhase.SUBMITTED, RunPhase.QUEUED}:
                    current = self._transition(
                        current,
                        RunPhase.RUNNING,
                        site=site,
                        job_id=job_id,
                        facts={"node": status.node or ""},
                    )
            elif status.state in {
                JobState.TERMINATED,
                JobState.ERROR,
                JobState.MISSING,
            }:
                return self._handle_terminal_status(
                    current,
                    status=status,
                    site=site,
                    job_id=job_id,
                    remote=current_remote,
                )
            if self.poll_seconds:
                self.sleeper(self.poll_seconds)

    def _fresh_run(self) -> AutonomousRunState:
        state = AutonomousRunState(
            run_id=self.config.identity.run_id,
            phase=RunPhase.CREATED,
            identity=self.config.identity.canonical_payload,
        )
        self.state.create(state)
        state = self._transition(
            state,
            RunPhase.PROBING,
            facts={
                "cleanup": self.config.cleanup,
                "sites": list(self.config.sites),
                "requirements": {
                    "gpu_memory_mb": self.config.requirements.gpu_memory_mb,
                    "cuda_capability": list(self.config.requirements.cuda_capability),
                    "persistent_free_bytes": self.config.requirements.persistent_free_bytes,
                },
            },
        )
        probes = tuple(
            self.probe_sites(
                sites=self.config.sites,
                requirements=self.config.requirements,
                max_workers=self.config.max_workers,
            )
        )
        selected = select_site(probes, requirements=self.config.requirements)
        state = self._transition(
            state,
            RunPhase.PROBING,
            site=selected.name,
            facts={
                "probes": [probe.to_dict(self.config.requirements) for probe in probes],
                "selected_site": selected.name,
            },
        )
        token = self._local_token_for_publication()
        self._provision_hub()
        remote = self.remote_factory(selected.name)
        self._active_remote = remote
        remote.prepare(
            run_id=self.config.identity.run_id,
            source_commit=self.config.identity.source_commit,
        )
        if token:
            remote.install_hugging_face_token(token)
        plan = self._build_plan(selected)
        state = self._transition(
            state,
            RunPhase.PREPARED,
            site=selected.name,
            facts={
                "allocation": plan.to_dict()["allocation"],
                "scheduler_command": list(plan.scheduler_command),
            },
        )
        state = self._transition(state, RunPhase.SUBMITTING)
        job_id = self._submit_plan(remote, plan)
        state = self._transition(
            state,
            RunPhase.SUBMITTED,
            site=selected.name,
            job_id=job_id,
        )
        return self._monitor(state, remote=remote)

    def _load_or_reconcile(self) -> AutonomousRunState | None:
        try:
            return self.state.load(self.config.identity.run_id)
        except LegacyAmbiguousStateError:
            active_job_ids: list[int] = []
            for site in self.config.sites:
                active_job_ids.extend(
                    OarClient(self.remote_factory(site)).user_job_ids()
                )
            archived = self.state.reconcile_legacy(
                self.config.identity.run_id,
                active_job_ids=tuple(active_job_ids),
            )
            self.emit(f"archived legacy ambiguous state at {archived}")
            return None

    def _run_loaded(self, current: AutonomousRunState | None) -> AutonomousRunState:
        if current is None:
            return self._fresh_run()
        if current.phase is RunPhase.COMPLETED:
            return current
        if current.phase is RunPhase.SUBMITTING:
            raise AutonomousRunError(
                "submission is ambiguous; inspect scheduler state before retrying"
            )
        if current.phase in {
            RunPhase.SUBMITTED,
            RunPhase.QUEUED,
            RunPhase.RUNNING,
        }:
            remote = self.remote_factory(current.site or "")
            self._active_remote = remote
            return self._monitor(current, remote=remote)
        raise AutonomousRunError(
            f"run is not resumable from phase {RunPhase(current.phase).value}"
        )

    def _record_unexpected_failure(self, error: Exception) -> None:
        current = self.state.load(self.config.identity.run_id)
        if current is None or current.phase in {
            RunPhase.SUBMITTING,
            RunPhase.SUBMITTED,
            RunPhase.QUEUED,
            RunPhase.RUNNING,
        }:
            if current is not None:
                self.state.append_event(
                    current.run_id,
                    "controller_error",
                    {"message": str(error)},
                )
            return
        failed = self._transition(
            current,
            RunPhase.FAILED,
            facts={"error": str(error)},
        )
        if self._active_remote is not None:
            with suppress(Exception):
                self._active_remote.mark_status(failed.run_id, "failed")

    def run(self) -> AutonomousRunState:
        """Resume a safe state or execute one complete autonomous lifecycle."""

        try:
            return self._run_loaded(self._load_or_reconcile())
        except LegacyAmbiguousStateError:
            raise
        except AutonomousRunError:
            raise
        except Exception as error:
            self._record_unexpected_failure(error)
            raise AutonomousRunError("autonomous Grid'5000 run failed") from error


__all__ = [
    "AutonomousRunConfig",
    "AutonomousRunController",
    "AutonomousRunError",
    "PolicyType",
]
