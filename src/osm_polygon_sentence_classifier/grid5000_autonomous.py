"""One-command autonomous Grid'5000 training lifecycle."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from .checkpoint_hub import HubCheckpointError, latest_published_checkpoint
from .config import ProjectConfig
from .grid5000 import (
    MINIMUM_HOME_HEADROOM_BYTES,
    CommandRunner,
    ContainerRuntime,
    Grid5000Allocation,
    Grid5000ConfigurationError,
    Grid5000Plan,
    Grid5000RunIdentity,
    _validate_container_settings,
    parse_quota_output,
)
from .grid5000_checkpointing import (
    CHECKPOINT_PROBE_ATTEMPTS,
    CHECKPOINT_PROBE_RETRY_SECONDS,
    CheckpointProbeError,
    probe_complete_checkpoint,
)
from .grid5000_oar import (
    JobState,
    JobStatus,
    OarClient,
    format_job_status,
    is_live_state,
)
from .grid5000_policy import (
    SHORT_TRIAL_WALLTIME_SECONDS,
    QueuedReplacementDecision,
    ReplacementCandidate,
    decide_queued_replacement,
    policy_type_for,
    should_seek_replacement,
)
from .grid5000_remote import Grid5000Remote
from .grid5000_replacement import (
    ReplacementContext,
    ReplacementCoordinator,
    replacement_candidates,
)
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
from .huggingface_http import configure_huggingface_http
from .publication import ensure_model_repository
from .tracking import ensure_trackio_resources, settings_for
from .training import TrainingConfig

PolicyType = Literal["auto", "day", "night"]
_UNSET = object()
DEFAULT_AUTONOMOUS_WALLTIME_SECONDS = SHORT_TRIAL_WALLTIME_SECONDS
MAX_REPLACEMENT_ATTEMPTS = 3
REPLACEMENT_RETRY_INTERVAL = timedelta(minutes=10)
_SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_STALE_QUEUE_ERROR_PATTERN = re.compile(
    r"^[A-Za-z0-9_.-]+ job [1-9][0-9]* remained queued with "
    r"(?:no start-time prediction|scheduled start "
    r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) "
    r"after [1-9][0-9]* replacement rounds$"
)


class AutonomousRunError(RuntimeError):
    """The autonomous lifecycle cannot continue safely."""


@dataclass(frozen=True, slots=True)
class AutonomousRunConfig:
    """Immutable settings for one reproducible autonomous run."""

    identity: Grid5000RunIdentity
    training_config: TrainingConfig
    sites: tuple[str, ...] = DEFAULT_SITES
    requirements: SiteRequirements = SiteRequirements()
    walltime_seconds: int = DEFAULT_AUTONOMOUS_WALLTIME_SECONDS
    policy_type: PolicyType = "auto"
    max_workers: int = DEFAULT_MAX_WORKERS
    max_continuations: int = 3
    worker_source_commit: str | None = None
    container_image: str | None = None
    container_runtime: ContainerRuntime = "auto"
    cleanup: bool = True

    def __post_init__(self) -> None:
        _validate_autonomous_limits(self)
        _validate_worker_source_commit(self.worker_source_commit)
        _validate_autonomous_container(self.container_image, self.container_runtime)
        _validate_autonomous_identity(self.training_config, self.identity)


def _validate_autonomous_limits(config: AutonomousRunConfig) -> None:
    if not config.sites:
        raise AutonomousRunError("at least one Grid'5000 site is required")
    if config.walltime_seconds <= 0:
        raise AutonomousRunError("walltime_seconds must be positive")
    if config.policy_type not in {"auto", "day", "night"}:
        raise AutonomousRunError("policy_type must be auto, day, or night")
    if config.max_workers <= 0:
        raise AutonomousRunError("max_workers must be positive")
    _validate_max_continuations(config.max_continuations)


def _validate_max_continuations(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AutonomousRunError("max_continuations must be positive")


def _validate_worker_source_commit(value: str | None) -> None:
    if value is not None and _SOURCE_COMMIT_PATTERN.fullmatch(value) is None:
        raise AutonomousRunError("worker_source_commit must be a pinned revision")


def _validate_autonomous_container(
    container_image: str | None, container_runtime: ContainerRuntime
) -> None:
    try:
        _validate_container_settings(container_image, container_runtime)
    except Grid5000ConfigurationError as error:
        raise AutonomousRunError(str(error)) from error


def _validate_autonomous_identity(
    training_config: TrainingConfig, identity: Grid5000RunIdentity
) -> None:
    if training_config.model_name_or_path != identity.model_name_or_path:
        raise AutonomousRunError("training model does not match run identity")
    if training_config.model_revision != identity.model_revision:
        raise AutonomousRunError("training revision does not match run identity")


def _validate_persisted_container_image(value: object) -> None:
    if value is not None and not isinstance(value, str):
        raise AutonomousRunError("persisted container image is invalid")


def _validate_persisted_container_runtime(value: object) -> None:
    if value not in {"auto", "docker", "podman"}:
        raise AutonomousRunError("persisted container runtime is invalid")


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


def _continuation_facts(
    *,
    continuation_count: int,
    max_continuations: int,
    worker_source_commit: str | None,
    continuation_pending: bool,
    continuation_reason: str | None = None,
    last_terminal_job_id: int | None = None,
    resume_from_checkpoint: bool | None = None,
    scheduler_command: Sequence[str] | None = None,
    requested_policy_type: PolicyType | None = None,
) -> dict[str, object]:
    facts: dict[str, object] = {
        "continuation_count": continuation_count,
        "max_continuations": max_continuations,
        "worker_source_commit": worker_source_commit,
        "continuation_pending": continuation_pending,
    }
    facts.update(
        {
            "replacement_attempted": False,
            "replacement_attempted_job_id": None,
            "replacement_attempt_count": 0,
            "replacement_last_attempt_at": None,
        }
    )
    _add_optional_continuation_facts(
        facts,
        continuation_reason=continuation_reason,
        last_terminal_job_id=last_terminal_job_id,
        resume_from_checkpoint=resume_from_checkpoint,
        scheduler_command=scheduler_command,
        requested_policy_type=requested_policy_type,
    )
    return facts


def _add_optional_continuation_facts(
    facts: dict[str, object],
    *,
    continuation_reason: str | None,
    last_terminal_job_id: int | None,
    resume_from_checkpoint: bool | None,
    scheduler_command: Sequence[str] | None,
    requested_policy_type: PolicyType | None,
) -> None:
    _set_optional_fact(facts, "continuation_reason", continuation_reason)
    _set_optional_fact(facts, "last_terminal_job_id", last_terminal_job_id)
    _set_optional_fact(facts, "resume_from_checkpoint", resume_from_checkpoint)
    _set_optional_fact(
        facts,
        "scheduler_command",
        list(scheduler_command) if scheduler_command is not None else None,
    )
    _set_optional_fact(facts, "requested_policy_type", requested_policy_type)


def _set_optional_fact(
    facts: dict[str, object], key: str, value: object | None
) -> None:
    if value is not None:
        facts[key] = value


def _replacement_attempt_count_for_job(
    facts: Mapping[str, object],
    *,
    job_id: int,
) -> int:
    if (
        facts.get("replacement_attempted") is not True
        or facts.get("replacement_attempted_job_id") != job_id
    ):
        return 0
    return _validated_replacement_attempt_count(
        facts.get("replacement_attempt_count", 0)
    )


def _validated_replacement_attempt_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutonomousRunError("durable replacement attempt count is invalid")
    return value


def _replacement_retry_due_for_job(
    facts: Mapping[str, object],
    *,
    job_id: int,
    now: datetime,
) -> bool:
    """Return whether another bounded replacement probe may start."""

    _validate_timezone(now)
    attempt_count = _replacement_attempt_count_for_job(facts, job_id=job_id)
    if attempt_count == 0:
        return True
    last_attempt = _last_replacement_attempt(facts)
    if last_attempt is None:
        return True

    return now >= last_attempt + REPLACEMENT_RETRY_INTERVAL


def _validate_timezone(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")


def _last_replacement_attempt(facts: Mapping[str, object]) -> datetime | None:
    raw_timestamp = facts.get("replacement_last_attempt_at")
    if raw_timestamp is None:
        # State written before the retry fields existed is safe to upgrade on
        # the next explicit resume; the new attempt records the durable time.
        return None
    if not isinstance(raw_timestamp, str):
        raise AutonomousRunError("durable replacement timestamp is invalid")
    return _parse_replacement_timestamp(raw_timestamp)


def _parse_replacement_timestamp(raw_timestamp: str) -> datetime:
    try:
        last_attempt = datetime.fromisoformat(raw_timestamp)
    except ValueError as error:
        raise AutonomousRunError("durable replacement timestamp is invalid") from error
    if last_attempt.tzinfo is None:
        raise AutonomousRunError("durable replacement timestamp is invalid")
    return last_attempt


def _validated_failed_run_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutonomousRunError("failed run continuation evidence is invalid")
    return value


def _validated_failed_run_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AutonomousRunError("failed run continuation evidence is invalid")
    return value


def _validated_continuation_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutonomousRunError("durable continuation count is invalid")
    return value


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
        site: str | None | object = _UNSET,
        job_id: int | None | object = _UNSET,
        facts: Mapping[str, object] | None = None,
    ) -> AutonomousRunState:
        updated = self._transition_state(current, phase, site, job_id, facts)
        self.state.save(updated)
        self.state.append_event(current.run_id, phase.value, facts or {})
        location, job = self._transition_location(updated)
        self.emit(f"run {current.run_id}: phase={phase.value}{location}{job}")
        return updated

    @staticmethod
    def _transition_state(
        current: AutonomousRunState,
        phase: RunPhase,
        site: str | None | object,
        job_id: int | None | object,
        facts: Mapping[str, object] | None,
    ) -> AutonomousRunState:
        merged = dict(current.facts or {})
        merged.update(facts or {})
        return AutonomousRunState(
            run_id=current.run_id,
            phase=phase,
            identity=current.identity,
            site=current.site if site is _UNSET else cast(str | None, site),
            job_id=current.job_id if job_id is _UNSET else cast(int | None, job_id),
            facts=merged,
        )

    @staticmethod
    def _transition_location(state: AutonomousRunState) -> tuple[str, str]:
        location = f" site={state.site}" if state.site is not None else ""
        job = f" job={state.job_id}" if state.job_id is not None else ""
        return location, job

    def _hub(self) -> Any:
        if self.hub_api is not None:
            return self.hub_api
        try:
            configure_huggingface_http()
            import huggingface_hub

            self.hub_api = huggingface_hub.HfApi()
            return self.hub_api
        except Exception as error:
            raise AutonomousRunError(
                "Hugging Face dependencies are required for publication"
            ) from error

    def _provision_hub(self) -> None:
        if self.config.training_config.publish_to_hub:
            self.emit("provisioning the Hugging Face model repository")
            ensure_model_repository(
                ProjectConfig().target_model_repository_id,
                hub_api=self._hub(),
            )
        if self.config.training_config.sync_trackio:
            self.emit("provisioning the Trackio Space and bucket")
            ensure_trackio_resources(
                settings_for(
                    ProjectConfig(),
                    project=self.config.training_config.tracking_project,
                ),
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

    def _worker_source_commit(self) -> str:
        return self.config.worker_source_commit or self.config.identity.source_commit

    def _build_plan(
        self,
        probe: SiteProbe,
        *,
        walltime_seconds: int | None = None,
        now: datetime | None = None,
        resume_from_checkpoint: bool = False,
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
            resume_from_checkpoint=resume_from_checkpoint,
            checkout_commit=self.config.worker_source_commit,
            container_image=self.config.container_image,
            container_runtime=self.config.container_runtime,
        )

    def _preflight(self, remote: Any, plan: Grid5000Plan) -> None:
        self.emit(
            f"{plan.allocation.site}: running checkout, policy, and quota preflight"
        )
        for command in (
            plan.remote_checkout_command[-1],
            plan.policy_site_command[-1],
            plan.policy_total_command[-1],
        ):
            remote.run(command)
        quota_result = remote.run(plan.quota_command[-1])
        quota = parse_quota_output(quota_result.stdout)
        minimum_headroom = (
            self.config.requirements.resume_persistent_free_bytes
            if plan.resume_from_checkpoint
            else MINIMUM_HOME_HEADROOM_BYTES
        )
        if quota.soft_headroom_bytes < minimum_headroom:
            raise AutonomousRunError(
                "Grid'5000 home soft quota has insufficient safe headroom"
            )

    def _submit_plan(self, remote: Any, plan: Grid5000Plan) -> int:
        self._preflight(remote, plan)
        return OarClient(remote).submit(plan.scheduler_command)

    def _verify_completion(self, remote: Any, manifest: Mapping[str, object]) -> None:
        training = self.config.training_config
        self._verify_completion_identity(manifest)
        if training.publish_to_hub:
            self._verify_model_publication(manifest)
        if training.sync_trackio:
            self._verify_tracking_space(manifest)

    def _verify_completion_identity(self, manifest: Mapping[str, object]) -> None:
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

    def _verify_model_publication(self, manifest: Mapping[str, object]) -> None:
        repository, commit_id = self._validated_model_publication(
            manifest.get("model_publication")
        )
        self._hub().model_info(repo_id=repository, revision=commit_id)

    def _validated_model_publication(self, value: object) -> tuple[str, str]:
        if not isinstance(value, Mapping):
            raise AutonomousRunError(
                "completed worker did not report a model publication"
            )
        publication = cast(Mapping[str, object], value)
        repository, commit_id = self._publication_values(publication)
        if repository != ProjectConfig().target_model_repository_id:
            raise AutonomousRunError("model publication repository is invalid")
        return repository, commit_id

    @staticmethod
    def _publication_values(value: Mapping[str, object]) -> tuple[str, str]:
        repository = value.get("repository_id")
        commit_id = value.get("commit_id")
        if (
            not isinstance(repository, str)
            or not isinstance(commit_id, str)
            or len(commit_id) != 40
        ):
            raise AutonomousRunError("model publication facts are invalid")
        return repository, commit_id

    def _verify_tracking_space(self, manifest: Mapping[str, object]) -> None:
        tracking_space = manifest.get("tracking_space_id")
        settings = settings_for(
            ProjectConfig(),
            project=self.config.training_config.tracking_project,
        )
        if tracking_space != settings.static_space_id:
            raise AutonomousRunError(
                "completed worker reported the wrong Trackio Space"
            )
        self._hub().space_info(repo_id=settings.static_space_id)

    def _candidate_list(
        self,
        probes: Sequence[SiteProbe],
        *,
        fallback_site: str,
    ) -> tuple[ReplacementCandidate, ...]:
        return replacement_candidates(
            probes,
            fallback_site=fallback_site,
            requirements=self.config.requirements,
        )

    def _try_replacement(
        self,
        *,
        fallback_site: str,
        fallback_job_id: int,
        fallback_remote: Any,
        resume_from_checkpoint: bool = False,
    ) -> tuple[str, int, Any]:
        requirements = (
            self.config.requirements.for_checkpoint_continuation()
            if resume_from_checkpoint
            else self.config.requirements
        )
        coordinator = ReplacementCoordinator(
            ReplacementContext(
                run_id=self.config.identity.run_id,
                source_commit=self._worker_source_commit(),
                sites=self.config.sites,
                requirements=requirements,
                max_workers=self.config.max_workers,
                cleanup=self.config.cleanup,
                poll_seconds=self.poll_seconds or 1.0,
                walltime_seconds=self.config.walltime_seconds,
                probe_sites=self.probe_sites,
                remote_factory=self.remote_factory,
                build_plan=lambda probe, walltime: self._build_plan(
                    probe,
                    walltime_seconds=walltime,
                    resume_from_checkpoint=resume_from_checkpoint,
                ),
                submit_plan=self._submit_plan,
                token_provider=self._local_token_for_publication,
                emit=self.emit,
                sleep=self.sleeper,
            )
        )
        return coordinator.run(
            fallback_site=fallback_site,
            fallback_job_id=fallback_job_id,
            fallback_remote=fallback_remote,
        )

    def _handle_queued_status(
        self,
        current: AutonomousRunState,
        *,
        status: JobStatus,
        site: str,
        job_id: int,
        remote: Any,
    ) -> tuple[AutonomousRunState, str, int, Any] | AutonomousRunState:
        if current.phase is RunPhase.SUBMITTED:
            current = self._transition(current, RunPhase.QUEUED)
        now = _now()
        facts = dict(current.facts or {})
        attempt_count = _replacement_attempt_count_for_job(facts, job_id=job_id)
        decision = self._queued_replacement_decision(
            status,
            site=site,
            job_id=job_id,
            now=now,
            facts=facts,
            attempt_count=attempt_count,
        )
        if decision.action == "fail":
            return self._fail_queued_job(
                current,
                site=site,
                job_id=job_id,
                remote=remote,
                decision=decision,
            )
        if decision.action == "wait":
            return current, site, job_id, remote
        return self._replace_queued_job(
            current,
            site=site,
            job_id=job_id,
            remote=remote,
            facts=facts,
            decision=decision,
        )

    def _queued_replacement_decision(
        self,
        status: JobStatus,
        *,
        site: str,
        job_id: int,
        now: datetime,
        facts: Mapping[str, object],
        attempt_count: int,
    ) -> QueuedReplacementDecision:
        retry_due = False
        if attempt_count < MAX_REPLACEMENT_ATTEMPTS and should_seek_replacement(
            status, now=now
        ):
            retry_due = _replacement_retry_due_for_job(facts, job_id=job_id, now=now)
        return decide_queued_replacement(
            status,
            site=site,
            job_id=job_id,
            now=now,
            attempt_count=attempt_count,
            retry_due=retry_due,
            max_attempts=MAX_REPLACEMENT_ATTEMPTS,
        )

    def _fail_queued_job(
        self,
        current: AutonomousRunState,
        *,
        site: str,
        job_id: int,
        remote: Any,
        decision: QueuedReplacementDecision,
    ) -> AutonomousRunState:
        message = cast(str, decision.message)
        self.emit(f"{message}; canceling stale fallback")
        OarClient(remote).cancel(job_id)
        return self._continue_after_incomplete(
            current,
            site=site,
            job_id=job_id,
            remote=remote,
            reason=message,
            failure_message=message,
        )

    def _replace_queued_job(
        self,
        current: AutonomousRunState,
        *,
        site: str,
        job_id: int,
        remote: Any,
        facts: Mapping[str, object],
        decision: QueuedReplacementDecision,
    ) -> tuple[AutonomousRunState, str, int, Any]:
        if decision.attempt_count is None or decision.attempt_timestamp is None:
            raise AutonomousRunError("replacement decision is missing attempt metadata")
        attempt_count = decision.attempt_count
        attempt_timestamp = decision.attempt_timestamp
        self.emit(
            f"{site} job {job_id}: replacement round "
            f"{attempt_count}/{MAX_REPLACEMENT_ATTEMPTS}"
        )
        current = self._transition(
            current,
            RunPhase.QUEUED,
            facts={
                "replacement_attempted": True,
                "replacement_attempted_job_id": job_id,
                "replacement_attempt_count": attempt_count,
                "replacement_last_attempt_at": attempt_timestamp,
            },
        )
        replacement_site, replacement_job, replacement_remote = self._try_replacement(
            fallback_site=site,
            fallback_job_id=job_id,
            fallback_remote=remote,
            resume_from_checkpoint=facts.get("resume_from_checkpoint") is True,
        )
        current = self._transition(
            current,
            RunPhase.QUEUED,
            site=replacement_site,
            job_id=replacement_job,
            facts=self._replacement_state_facts(
                replacement_site,
                replacement_job,
                site=site,
                job_id=job_id,
                attempt_count=attempt_count,
                attempt_timestamp=attempt_timestamp,
            ),
        )
        return current, replacement_site, replacement_job, replacement_remote

    @staticmethod
    def _replacement_state_facts(
        replacement_site: str,
        replacement_job: int,
        *,
        site: str,
        job_id: int,
        attempt_count: int,
        attempt_timestamp: str,
    ) -> dict[str, object]:
        if replacement_site == site and replacement_job == job_id:
            return {
                "replacement_attempted": True,
                "replacement_attempted_job_id": job_id,
                "replacement_attempt_count": attempt_count,
                "replacement_last_attempt_at": attempt_timestamp,
            }
        return {
            "replacement_attempted": False,
            "replacement_attempted_job_id": None,
            "replacement_attempt_count": 0,
            "replacement_last_attempt_at": None,
        }

    def _fail_terminal(
        self,
        current: AutonomousRunState,
        *,
        site: str,
        job_id: int,
        remote: Any,
        message: str,
    ) -> NoReturn:
        failed = self._transition(
            current,
            RunPhase.FAILED,
            site=site,
            job_id=job_id,
            facts={"error": message},
        )
        remote.mark_status(self.config.identity.run_id, "failed")
        raise AutonomousRunError(
            str(dict(failed.facts or {}).get("error", "job failed"))
        )

    def _continuation_probe(self, site: str) -> SiteProbe:
        del site
        requirements = self.config.requirements.for_checkpoint_continuation()
        probes = tuple(
            self.probe_sites(
                sites=self.config.sites,
                requirements=requirements,
                max_workers=self.config.max_workers,
            )
        )
        try:
            return select_site(probes, requirements=requirements)
        except RuntimeError as error:
            raise AutonomousRunError(
                "no compatible Grid'5000 site is available for checkpoint continuation"
            ) from error

    def _has_complete_checkpoint(
        self,
        remote: Any,
        *,
        site: str,
        job_id: int,
        allow_failed_status: bool,
    ) -> bool:
        """Probe checkpoint evidence with bounded retries for SSH outages."""
        local_ready = self._probe_local_checkpoint(
            remote,
            site=site,
            job_id=job_id,
            allow_failed_status=allow_failed_status,
        )
        if local_ready:
            return True
        return self._probe_published_checkpoint(site=site, job_id=job_id)

    def _probe_local_checkpoint(
        self,
        remote: Any,
        *,
        site: str,
        job_id: int,
        allow_failed_status: bool,
    ) -> bool:
        try:
            return probe_complete_checkpoint(
                remote,
                run_id=self.config.identity.run_id,
                output_subdirectory=self.config.training_config.output_subdirectory,
                identity=self.config.identity.canonical_payload,
                allow_failed_status=allow_failed_status,
                site=site,
                job_id=job_id,
                poll_seconds=self.poll_seconds,
                emit=self.emit,
                sleep=self.sleeper,
                attempts=CHECKPOINT_PROBE_ATTEMPTS,
                retry_seconds=CHECKPOINT_PROBE_RETRY_SECONDS,
            )
        except CheckpointProbeError as error:
            if error.__cause__ is not None:
                raise AutonomousRunError(str(error)) from error.__cause__
            raise AutonomousRunError(str(error)) from error

    def _probe_published_checkpoint(self, *, site: str, job_id: int) -> bool:
        if not self.config.training_config.publish_to_hub:
            return False
        try:
            published = latest_published_checkpoint(
                self.config.identity.canonical_payload,
                repository_id=ProjectConfig().target_model_repository_id,
                hub_api=self._hub(),
            )
        except HubCheckpointError as error:
            raise AutonomousRunError(
                "published checkpoint availability could not be verified"
            ) from error
        if published is None:
            return False
        self.emit(
            f"{site} job {job_id}: using published checkpoint at step {published.step}"
        )
        return True

    def _continue_after_incomplete(
        self,
        current: AutonomousRunState,
        *,
        site: str,
        job_id: int,
        remote: Any,
        reason: str,
        failure_message: str | None = None,
    ) -> AutonomousRunState:
        facts = dict(current.facts or {})
        raw_count = _validated_continuation_count(facts.get("continuation_count", 0))
        if raw_count >= self.config.max_continuations:
            return self._fail_terminal(
                current,
                site=site,
                job_id=job_id,
                remote=remote,
                message=(
                    f"job ended without completion after {raw_count} checkpoint "
                    "continuations"
                ),
            )
        self.emit(
            f"{site} job {job_id}: checking checkpoint evidence before "
            f"continuation {raw_count + 1}/{self.config.max_continuations}"
        )
        self._require_continuation_checkpoint(
            current,
            site=site,
            job_id=job_id,
            remote=remote,
            failure_message=failure_message,
        )
        self.emit(f"{site} job {job_id}: complete checkpoint found")
        probe, remote, plan = self._prepare_continuation(
            current, site=site, remote=remote
        )
        return self._submit_continuation(
            current,
            probe=probe,
            remote=remote,
            plan=plan,
            raw_count=raw_count,
            reason=reason,
            last_terminal_job_id=job_id,
        )

    def _require_continuation_checkpoint(
        self,
        current: AutonomousRunState,
        *,
        site: str,
        job_id: int,
        remote: Any,
        failure_message: str | None,
    ) -> None:
        try:
            checkpoint_ready = self._has_complete_checkpoint(
                remote,
                site=site,
                job_id=job_id,
                allow_failed_status=current.phase is RunPhase.FAILED,
            )
        except AutonomousRunError as error:
            self._fail_terminal(
                current,
                site=site,
                job_id=job_id,
                remote=remote,
                message=str(error),
            )
        if not checkpoint_ready:
            self._fail_terminal(
                current,
                site=site,
                job_id=job_id,
                remote=remote,
                message=failure_message or "job ended without a complete checkpoint",
            )

    def _prepare_continuation(
        self,
        current: AutonomousRunState,
        *,
        site: str,
        remote: Any,
    ) -> tuple[SiteProbe, Any, Grid5000Plan]:
        probe = self._continuation_probe(site)
        if probe.name != site:
            self.emit(
                f"run {self.config.identity.run_id}: continuing on {probe.name} "
                f"after {site} became unavailable"
            )
            remote = self.remote_factory(probe.name)
            self._active_remote = remote
        token = self._local_token_for_publication()
        remote.prepare(
            run_id=self.config.identity.run_id,
            source_commit=self._worker_source_commit(),
            allow_failed_run=current.phase is RunPhase.FAILED,
        )
        if token:
            remote.install_hugging_face_token(token)
        plan = self._build_plan(probe, resume_from_checkpoint=True)
        self._preflight(remote, plan)
        return probe, remote, plan

    def _submit_continuation(
        self,
        current: AutonomousRunState,
        *,
        probe: SiteProbe,
        remote: Any,
        plan: Grid5000Plan,
        raw_count: int,
        reason: str,
        last_terminal_job_id: int,
    ) -> AutonomousRunState:
        pending = self._transition(
            current,
            RunPhase.SUBMITTING,
            site=probe.name,
            job_id=None,
            facts=_continuation_facts(
                continuation_count=raw_count + 1,
                max_continuations=self.config.max_continuations,
                worker_source_commit=self.config.worker_source_commit,
                continuation_pending=True,
                continuation_reason=reason,
                last_terminal_job_id=last_terminal_job_id,
                requested_policy_type=self.config.policy_type,
            ),
        )
        successor_job_id = OarClient(remote).submit(plan.scheduler_command)
        submitted = self._transition(
            pending,
            RunPhase.SUBMITTED,
            site=probe.name,
            job_id=successor_job_id,
            facts=_continuation_facts(
                continuation_count=raw_count + 1,
                max_continuations=self.config.max_continuations,
                worker_source_commit=self.config.worker_source_commit,
                continuation_pending=False,
                resume_from_checkpoint=True,
                scheduler_command=plan.scheduler_command,
                requested_policy_type=self.config.policy_type,
            ),
        )
        return self._monitor(submitted, remote=remote)

    def _handle_terminal_status(
        self,
        current: AutonomousRunState,
        *,
        status: JobStatus,
        site: str,
        job_id: int,
        remote: Any,
    ) -> AutonomousRunState:
        reason = f"job ended as {status.state.value}; exit_code={status.exit_code}"
        if self._terminal_job_failed(status):
            return self._continue_after_incomplete(
                current,
                site=site,
                job_id=job_id,
                remote=remote,
                reason=reason,
            )
        return self._complete_terminal_job(
            current,
            site=site,
            job_id=job_id,
            remote=remote,
        )

    @staticmethod
    def _terminal_job_failed(status: JobStatus) -> bool:
        return status.state is not JobState.TERMINATED or status.exit_code not in {
            None,
            0,
        }

    def _complete_terminal_job(
        self,
        current: AutonomousRunState,
        *,
        site: str,
        job_id: int,
        remote: Any,
    ) -> AutonomousRunState:
        try:
            self.emit(f"{site} job {job_id}: verifying completion manifest")
            manifest = remote.read_completion(self.config.identity.run_id)
            self._verify_completion(remote, manifest)
        except AutonomousRunError as error:
            return self._fail_terminal(
                current,
                site=site,
                job_id=job_id,
                remote=remote,
                message=str(error),
            )
        except Exception as error:
            return self._continue_after_incomplete(
                current,
                site=site,
                job_id=job_id,
                remote=remote,
                reason=str(error),
            )
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

    def _resume_failed_run(self, current: AutonomousRunState) -> AutonomousRunState:
        """Extend only a failed run whose checkpoint limit was exhausted."""

        facts = dict(current.facts or {})
        raw_count, raw_limit = self._failed_run_counts(facts)
        self._validate_failed_run_resume(facts, raw_count, raw_limit)
        site, job_id = self._failed_run_location(current)

        remote = self.remote_factory(site)
        self._active_remote = remote
        status = self._previous_job_status(remote, job_id)
        if is_live_state(status.state):
            raise AutonomousRunError(
                "failed run still has an active Grid'5000 job; refusing a duplicate"
            )
        self.emit(
            f"run {current.run_id}: extending from the retained checkpoint "
            f"after job {job_id}"
        )
        return self._continue_after_incomplete(
            current,
            site=site,
            job_id=job_id,
            remote=remote,
            reason="explicit continuation extension",
        )

    @staticmethod
    def _failed_run_counts(facts: Mapping[str, object]) -> tuple[int, int]:
        return (
            _validated_failed_run_count(facts.get("continuation_count")),
            _validated_failed_run_limit(facts.get("max_continuations")),
        )

    def _validate_failed_run_resume(
        self,
        facts: Mapping[str, object],
        raw_count: int,
        raw_limit: int,
    ) -> None:
        exhausted = self._failed_run_is_exhausted(facts, raw_count, raw_limit)
        if not self._failed_run_is_resumable(facts, raw_count, exhausted):
            raise AutonomousRunError("run is not resumable from phase failed")
        if exhausted and self.config.max_continuations <= raw_limit:
            raise AutonomousRunError(
                f"failed run exhausted {raw_limit} checkpoint continuations; "
                f"resume with --max-continuations greater than {raw_limit}"
            )

    def _failed_run_is_resumable(
        self,
        facts: Mapping[str, object],
        raw_count: int,
        exhausted: bool,
    ) -> bool:
        return (
            self._recoverable_checkpoint_failure(facts, raw_count)
            or self._stale_queued_failure(facts, raw_count)
            or exhausted
        )

    @staticmethod
    def _failed_run_is_exhausted(
        facts: Mapping[str, object], raw_count: int, raw_limit: int
    ) -> bool:
        expected_error = (
            f"job ended without completion after {raw_count} checkpoint continuations"
        )
        return facts.get("error") == expected_error and raw_count == raw_limit

    def _recoverable_checkpoint_failure(
        self, facts: Mapping[str, object], raw_count: int
    ) -> bool:
        return (
            facts.get("error") == "job ended without a complete checkpoint"
            and raw_count < self.config.max_continuations
        )

    def _stale_queued_failure(
        self, facts: Mapping[str, object], raw_count: int
    ) -> bool:
        error = facts.get("error")
        return (
            isinstance(error, str)
            and _STALE_QUEUE_ERROR_PATTERN.fullmatch(error) is not None
            and raw_count < self.config.max_continuations
        )

    @staticmethod
    def _failed_run_location(current: AutonomousRunState) -> tuple[str, int]:
        if current.site is None or current.job_id is None:
            raise AutonomousRunError("failed run lacks its last Grid'5000 job")
        return current.site, current.job_id

    @staticmethod
    def _previous_job_status(remote: Any, job_id: int) -> JobStatus:
        try:
            return OarClient(remote).status(job_id)
        except Exception as error:
            raise AutonomousRunError(
                "previous Grid'5000 job status could not be verified"
            ) from error

    def _monitor(
        self,
        state: AutonomousRunState,
        *,
        remote: Any,
    ) -> AutonomousRunState:
        site, job_id = self._monitor_location(state)
        current_remote = remote
        oar = OarClient(current_remote)
        current = state
        self.emit(
            f"run {state.run_id}: monitoring {site} job {job_id} "
            f"(poll every {self.poll_seconds:g}s)"
        )
        while True:
            status = oar.status(job_id)
            self.emit(f"{site} job {job_id}: {format_job_status(status)}")
            result = self._monitor_status(
                current,
                status=status,
                site=site,
                job_id=job_id,
                remote=current_remote,
            )
            if isinstance(result, AutonomousRunState):
                return result
            if result is not None:
                current, site, job_id, current_remote = result
                oar = OarClient(current_remote)
            if self.poll_seconds:
                self.sleeper(self.poll_seconds)

    @staticmethod
    def _monitor_location(state: AutonomousRunState) -> tuple[str, int]:
        if state.site is None or state.job_id is None:
            raise AutonomousRunError("submitted state lacks site or job ID")
        return state.site, state.job_id

    def _monitor_status(
        self,
        current: AutonomousRunState,
        *,
        status: JobStatus,
        site: str,
        job_id: int,
        remote: Any,
    ) -> tuple[AutonomousRunState, str, int, Any] | AutonomousRunState | None:
        if status.state is JobState.QUEUED:
            return self._handle_queued_status(
                current,
                status=status,
                site=site,
                job_id=job_id,
                remote=remote,
            )
        if status.state is JobState.RUNNING:
            updated = self._running_state(current, status, site=site, job_id=job_id)
            return None if updated is None else (updated, site, job_id, remote)
        if status.state in {
            JobState.TERMINATED,
            JobState.ERROR,
            JobState.MISSING,
        }:
            return self._handle_terminal_status(
                current,
                status=status,
                site=site,
                job_id=job_id,
                remote=remote,
            )
        return None

    def _running_state(
        self,
        current: AutonomousRunState,
        status: JobStatus,
        *,
        site: str,
        job_id: int,
    ) -> AutonomousRunState | None:
        if current.phase in {RunPhase.SUBMITTED, RunPhase.QUEUED}:
            return self._transition(
                current,
                RunPhase.RUNNING,
                site=site,
                job_id=job_id,
                facts={"node": status.node or ""},
            )
        return None

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
                "max_continuations": self.config.max_continuations,
                "worker_source_commit": self.config.worker_source_commit,
                "container_image": self.config.container_image,
                "container_runtime": self.config.container_runtime,
                "requested_policy_type": self.config.policy_type,
                "sites": list(self.config.sites),
                "requirements": {
                    "gpu_memory_mb": self.config.requirements.gpu_memory_mb,
                    "cuda_capability": list(self.config.requirements.cuda_capability),
                    "persistent_free_bytes": self.config.requirements.persistent_free_bytes,
                    "resume_persistent_free_bytes": (
                        self.config.requirements.resume_persistent_free_bytes
                    ),
                },
            },
        )
        self.emit(
            f"run {self.config.identity.run_id}: probing "
            f"{len(self.config.sites)} configured Grid'5000 sites"
        )
        probes = tuple(
            self.probe_sites(
                sites=self.config.sites,
                requirements=self.config.requirements,
                max_workers=self.config.max_workers,
            )
        )
        selected = select_site(probes, requirements=self.config.requirements)
        self.emit(f"run {self.config.identity.run_id}: selected site {selected.name}")
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
            source_commit=self._worker_source_commit(),
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
        if current.phase is not RunPhase.COMPLETED:
            self._validate_persisted_container_settings(current)
        if current.phase is RunPhase.COMPLETED:
            return current
        return self._resume_loaded_phase(current)

    def _resume_loaded_phase(self, current: AutonomousRunState) -> AutonomousRunState:
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
        if current.phase is RunPhase.FAILED:
            return self._resume_failed_run(current)
        raise AutonomousRunError(
            f"run is not resumable from phase {RunPhase(current.phase).value}"
        )

    def _validate_persisted_container_settings(
        self, current: AutonomousRunState
    ) -> None:
        facts = dict(current.facts or {})
        persisted_image = facts.get("container_image")
        persisted_runtime = facts.get("container_runtime", "auto")
        _validate_persisted_container_image(persisted_image)
        _validate_persisted_container_runtime(persisted_runtime)
        if self._container_settings_differ(persisted_image, persisted_runtime):
            raise AutonomousRunError(
                "container settings differ from the persisted run; "
                "start a new run instead"
            )

    def _container_settings_differ(
        self: AutonomousRunController,
        persisted_image: object,
        persisted_runtime: object,
    ) -> bool:
        return (
            persisted_image != self.config.container_image
            or persisted_runtime != self.config.container_runtime
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
    "DEFAULT_AUTONOMOUS_WALLTIME_SECONDS",
    "MAX_REPLACEMENT_ATTEMPTS",
    "PolicyType",
    "REPLACEMENT_RETRY_INTERVAL",
]
