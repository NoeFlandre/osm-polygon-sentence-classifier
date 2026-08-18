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
from .publication import ensure_model_repository
from .tracking import ensure_trackio_resources, settings_for
from .training import TrainingConfig

PolicyType = Literal["auto", "day", "night"]
_UNSET = object()
DEFAULT_AUTONOMOUS_WALLTIME_SECONDS = SHORT_TRIAL_WALLTIME_SECONDS
MAX_REPLACEMENT_ATTEMPTS = 3
REPLACEMENT_RETRY_INTERVAL = timedelta(minutes=10)
_SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


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
        if not self.sites:
            raise AutonomousRunError("at least one Grid'5000 site is required")
        if self.walltime_seconds <= 0:
            raise AutonomousRunError("walltime_seconds must be positive")
        if self.policy_type not in {"auto", "day", "night"}:
            raise AutonomousRunError("policy_type must be auto, day, or night")
        if self.max_workers <= 0:
            raise AutonomousRunError("max_workers must be positive")
        if (
            isinstance(self.max_continuations, bool)
            or not isinstance(self.max_continuations, int)
            or self.max_continuations <= 0
        ):
            raise AutonomousRunError("max_continuations must be positive")
        if (
            self.worker_source_commit is not None
            and _SOURCE_COMMIT_PATTERN.fullmatch(self.worker_source_commit) is None
        ):
            raise AutonomousRunError("worker_source_commit must be a pinned revision")
        try:
            _validate_container_settings(self.container_image, self.container_runtime)
        except Grid5000ConfigurationError as error:
            raise AutonomousRunError(str(error)) from error
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
) -> dict[str, object]:
    facts: dict[str, object] = {
        "continuation_count": continuation_count,
        "max_continuations": max_continuations,
        "worker_source_commit": worker_source_commit,
        "continuation_pending": continuation_pending,
    }
    if continuation_reason is not None:
        facts["continuation_reason"] = continuation_reason
    if last_terminal_job_id is not None:
        facts["last_terminal_job_id"] = last_terminal_job_id
    facts.update(
        {
            "replacement_attempted": False,
            "replacement_attempted_job_id": None,
            "replacement_attempt_count": 0,
            "replacement_last_attempt_at": None,
        }
    )
    if resume_from_checkpoint is not None:
        facts["resume_from_checkpoint"] = resume_from_checkpoint
    if scheduler_command is not None:
        facts["scheduler_command"] = list(scheduler_command)
    return facts


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
    raw_count = facts.get("replacement_attempt_count", 0)
    if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
        raise AutonomousRunError("durable replacement attempt count is invalid")
    return raw_count


def _replacement_retry_due_for_job(
    facts: Mapping[str, object],
    *,
    job_id: int,
    now: datetime,
) -> bool:
    """Return whether another bounded replacement probe may start."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    attempt_count = _replacement_attempt_count_for_job(facts, job_id=job_id)
    if attempt_count == 0:
        return True
    raw_timestamp = facts.get("replacement_last_attempt_at")
    if raw_timestamp is None:
        # State written before the retry fields existed is safe to upgrade on
        # the next explicit resume; the new attempt records the durable time.
        return True
    if not isinstance(raw_timestamp, str):
        raise AutonomousRunError("durable replacement timestamp is invalid")
    try:
        last_attempt = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise AutonomousRunError("durable replacement timestamp is invalid") from error
    if last_attempt.tzinfo is None or last_attempt.utcoffset() is None:
        raise AutonomousRunError("durable replacement timestamp is invalid")
    return now >= last_attempt + REPLACEMENT_RETRY_INTERVAL


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
        merged = dict(current.facts or {})
        merged.update(facts or {})
        updated = AutonomousRunState(
            run_id=current.run_id,
            phase=phase,
            identity=current.identity,
            site=current.site if site is _UNSET else cast(str | None, site),
            job_id=current.job_id if job_id is _UNSET else cast(int | None, job_id),
            facts=merged,
        )
        self.state.save(updated)
        self.state.append_event(current.run_id, phase.value, facts or {})
        location = f" site={updated.site}" if updated.site is not None else ""
        job = f" job={updated.job_id}" if updated.job_id is not None else ""
        self.emit(f"run {current.run_id}: phase={phase.value}{location}{job}")
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
            settings = settings_for(
                ProjectConfig(),
                project=training.tracking_project,
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
    ) -> tuple[str, int, Any]:
        coordinator = ReplacementCoordinator(
            ReplacementContext(
                run_id=self.config.identity.run_id,
                source_commit=self._worker_source_commit(),
                sites=self.config.sites,
                requirements=self.config.requirements,
                max_workers=self.config.max_workers,
                cleanup=self.config.cleanup,
                poll_seconds=self.poll_seconds or 1.0,
                probe_sites=self.probe_sites,
                remote_factory=self.remote_factory,
                build_plan=lambda probe, walltime: self._build_plan(
                    probe,
                    walltime_seconds=walltime,
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
    ) -> tuple[AutonomousRunState, str, int, Any]:
        if current.phase is RunPhase.SUBMITTED:
            current = self._transition(current, RunPhase.QUEUED)
        now = _now()
        facts = dict(current.facts or {})
        attempt_count = _replacement_attempt_count_for_job(facts, job_id=job_id)
        retry_due = False
        if attempt_count < MAX_REPLACEMENT_ATTEMPTS and should_seek_replacement(
            status, now=now
        ):
            retry_due = _replacement_retry_due_for_job(facts, job_id=job_id, now=now)
        decision = decide_queued_replacement(
            status,
            site=site,
            job_id=job_id,
            now=now,
            attempt_count=attempt_count,
            retry_due=retry_due,
            max_attempts=MAX_REPLACEMENT_ATTEMPTS,
        )
        if decision.action == "fail":
            message = cast(str, decision.message)
            self.emit(f"{message}; canceling stale fallback")
            OarClient(remote).cancel(job_id)
            return self._fail_terminal(
                current,
                site=site,
                job_id=job_id,
                remote=remote,
                message=message,
            )
        if decision.action == "wait":
            return current, site, job_id, remote
        attempt_count = cast(int, decision.attempt_count)
        attempt_timestamp = cast(str, decision.attempt_timestamp)
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
        )
        current = self._transition(
            current,
            RunPhase.QUEUED,
            site=replacement_site,
            job_id=replacement_job,
            facts=(
                {
                    "replacement_attempted": True,
                    "replacement_attempted_job_id": job_id,
                    "replacement_attempt_count": attempt_count,
                    "replacement_last_attempt_at": attempt_timestamp,
                }
                if replacement_site == site and replacement_job == job_id
                else {
                    "replacement_attempted": False,
                    "replacement_attempted_job_id": None,
                    "replacement_attempt_count": 0,
                    "replacement_last_attempt_at": None,
                }
            ),
        )
        return current, replacement_site, replacement_job, replacement_remote

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
        probes = tuple(
            self.probe_sites(
                sites=(site,),
                requirements=self.config.requirements,
                max_workers=1,
            )
        )
        try:
            return select_site(probes, requirements=self.config.requirements)
        except RuntimeError as error:
            raise AutonomousRunError(
                f"site {site} is no longer compatible for checkpoint continuation"
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

    def _continue_after_incomplete(
        self,
        current: AutonomousRunState,
        *,
        site: str,
        job_id: int,
        remote: Any,
        reason: str,
    ) -> AutonomousRunState:
        facts = dict(current.facts or {})
        raw_count = facts.get("continuation_count", 0)
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 0
        ):
            raise AutonomousRunError("durable continuation count is invalid")
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
        checkpoint_ready = self._has_complete_checkpoint(
            remote,
            site=site,
            job_id=job_id,
            allow_failed_status=current.phase is RunPhase.FAILED,
        )
        if not checkpoint_ready:
            return self._fail_terminal(
                current,
                site=site,
                job_id=job_id,
                remote=remote,
                message="job ended without a complete checkpoint",
            )
        self.emit(f"{site} job {job_id}: complete checkpoint found")
        probe = self._continuation_probe(site)
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
        pending = self._transition(
            current,
            RunPhase.SUBMITTING,
            site=site,
            job_id=None,
            facts=_continuation_facts(
                continuation_count=raw_count + 1,
                max_continuations=self.config.max_continuations,
                worker_source_commit=self.config.worker_source_commit,
                continuation_pending=True,
                continuation_reason=reason,
                last_terminal_job_id=job_id,
            ),
        )
        successor_job_id = OarClient(remote).submit(plan.scheduler_command)
        submitted = self._transition(
            pending,
            RunPhase.SUBMITTED,
            site=site,
            job_id=successor_job_id,
            facts=_continuation_facts(
                continuation_count=raw_count + 1,
                max_continuations=self.config.max_continuations,
                worker_source_commit=self.config.worker_source_commit,
                continuation_pending=False,
                resume_from_checkpoint=True,
                scheduler_command=plan.scheduler_command,
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
        if status.state is not JobState.TERMINATED or status.exit_code not in {None, 0}:
            return self._continue_after_incomplete(
                current,
                site=site,
                job_id=job_id,
                remote=remote,
                reason=reason,
            )
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
        raw_count = facts.get("continuation_count")
        raw_limit = facts.get("max_continuations")
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 0
            or isinstance(raw_limit, bool)
            or not isinstance(raw_limit, int)
            or raw_limit <= 0
        ):
            raise AutonomousRunError("failed run continuation evidence is invalid")
        expected_error = (
            f"job ended without completion after {raw_count} checkpoint continuations"
        )
        if raw_count != raw_limit or facts.get("error") != expected_error:
            raise AutonomousRunError("run is not resumable from phase failed")
        if self.config.max_continuations <= raw_limit:
            raise AutonomousRunError(
                f"failed run exhausted {raw_limit} checkpoint continuations; "
                f"resume with --max-continuations greater than {raw_limit}"
            )
        if current.site is None or current.job_id is None:
            raise AutonomousRunError("failed run lacks its last Grid'5000 job")

        remote = self.remote_factory(current.site)
        self._active_remote = remote
        try:
            status = OarClient(remote).status(current.job_id)
        except Exception as error:
            raise AutonomousRunError(
                "previous Grid'5000 job status could not be verified"
            ) from error
        if is_live_state(status.state):
            raise AutonomousRunError(
                "failed run still has an active Grid'5000 job; refusing a duplicate"
            )
        self.emit(
            f"run {current.run_id}: extending from the retained checkpoint "
            f"after job {current.job_id}"
        )
        return self._continue_after_incomplete(
            current,
            site=current.site,
            job_id=current.job_id,
            remote=remote,
            reason="explicit continuation extension",
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
        current = state
        self.emit(
            f"run {state.run_id}: monitoring {site} job {job_id} "
            f"(poll every {self.poll_seconds:g}s)"
        )
        while True:
            status = oar.status(job_id)
            self.emit(f"{site} job {job_id}: {format_job_status(status)}")
            if status.state is JobState.QUEUED:
                (
                    current,
                    site,
                    job_id,
                    current_remote,
                ) = self._handle_queued_status(
                    current,
                    status=status,
                    site=site,
                    job_id=job_id,
                    remote=current_remote,
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
                "max_continuations": self.config.max_continuations,
                "worker_source_commit": self.config.worker_source_commit,
                "container_image": self.config.container_image,
                "container_runtime": self.config.container_runtime,
                "sites": list(self.config.sites),
                "requirements": {
                    "gpu_memory_mb": self.config.requirements.gpu_memory_mb,
                    "cuda_capability": list(self.config.requirements.cuda_capability),
                    "persistent_free_bytes": self.config.requirements.persistent_free_bytes,
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
        if persisted_image is not None and not isinstance(persisted_image, str):
            raise AutonomousRunError("persisted container image is invalid")
        if persisted_runtime not in {"auto", "docker", "podman"}:
            raise AutonomousRunError("persisted container runtime is invalid")
        if (
            persisted_image != self.config.container_image
            or persisted_runtime != self.config.container_runtime
        ):
            raise AutonomousRunError(
                "container settings differ from the persisted run; "
                "start a new run instead"
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
