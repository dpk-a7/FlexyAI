"""Redis-backed runtime state, artifacts, and the single isolated Flexy runner."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import hashlib
import hmac
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import re
import secrets
import shutil
import time
from typing import Any, Callable
from urllib.parse import quote, urlsplit
from uuid import uuid4

from apps._registry import AppDefinition, get_app
from flexy_worker import run_worker
from resource_manager import ResourceManager, ResourceSettings, ResourceState, RunnerInfo, shutdown_gpu_monitor


PROJECT_ROOT = Path(__file__).resolve().parent
ACTIVE_RUNNER_KEY = "flexy:active_runner"
EXPIRY_KEY = "flexy:artifact_expiry"
TERMINAL_JOB_STATES = {"completed", "failed", "expired"}


class PlatformError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def parse_duration(value: str, name: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([smhd])", value)
    if match is None:
        raise ValueError(f"{name} must be a positive duration such as 30m, 2h, or 1d")
    return int(match[1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[match[2]]


@dataclass(frozen=True)
class RuntimeSettings:
    redis_url: str
    artifact_root: Path
    artifact_ttl_seconds: int
    download_url_ttl_seconds: int
    max_upload_bytes: int

    @classmethod
    def from_environment(cls) -> "RuntimeSettings":
        artifact_root_value = os.getenv("ARTIFACT_ROOT", "").strip()
        if not artifact_root_value:
            raise ValueError("ARTIFACT_ROOT must point to a dedicated temporary media directory")
        artifact_root = Path(artifact_root_value).resolve()
        redis_url = os.getenv("REDIS_URL", "").strip()
        if redis_url:
            parsed_redis_url = urlsplit(redis_url)
            if (
                parsed_redis_url.scheme not in {"redis", "rediss"}
                or not parsed_redis_url.hostname
                or not parsed_redis_url.password
            ):
                raise ValueError("REDIS_URL must use an authenticated redis:// or rediss:// URL")
        else:
            try:
                redis_port = int(os.getenv("REDIS_PORT", "6379"))
            except ValueError as exc:
                raise ValueError("REDIS_PORT must be an integer") from exc
            if not 1 <= redis_port <= 65535:
                raise ValueError("REDIS_PORT must be between 1 and 65535")
            redis_password = os.getenv("REDIS_PASSWORD", "")
            if not redis_password:
                raise ValueError("Set REDIS_PASSWORD or an authenticated REDIS_URL")
            credentials = f":{quote(redis_password, safe='')}@"
            redis_url = f"redis://{credentials}127.0.0.1:{redis_port}/0"
        try:
            max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(1024**3)))
        except ValueError as exc:
            raise ValueError("MAX_UPLOAD_BYTES must be a positive integer") from exc
        if max_upload_bytes < 1:
            raise ValueError("MAX_UPLOAD_BYTES must be a positive integer")
        return cls(
            redis_url=redis_url,
            artifact_root=artifact_root,
            artifact_ttl_seconds=parse_duration(os.getenv("AUTO_DELETE_MEDIA", "30m"), "AUTO_DELETE_MEDIA"),
            download_url_ttl_seconds=parse_duration(
                os.getenv("DOWNLOAD_URL_TTL", "10m"), "DOWNLOAD_URL_TTL"),
            max_upload_bytes=max_upload_bytes,
        )


class RedisState:
    """Small async wrapper that keeps Redis keys and values deliberate."""

    def __init__(self, client: Any):
        self.client = client

    async def start(self) -> None:
        await self.client.ping()

    async def close(self) -> None:
        await self.client.aclose()

    async def acquire_active_runner(self, runner_id: str) -> bool:
        return bool(await self.client.set(ACTIVE_RUNNER_KEY, runner_id, nx=True))

    async def release_active_runner(self, runner_id: str) -> None:
        if await self.client.get(ACTIVE_RUNNER_KEY) == runner_id:
            await self.client.delete(ACTIVE_RUNNER_KEY)

    async def active_runner_id(self) -> str | None:
        return await self.client.get(ACTIVE_RUNNER_KEY)

    async def save_runner(self, runner_id: str, fields: dict[str, Any]) -> None:
        await self.client.hset(self._runner_key(runner_id), mapping=_redis_mapping(fields))

    async def update_runner(self, runner_id: str, **fields: Any) -> None:
        await self.save_runner(runner_id, fields)

    async def get_runner(self, runner_id: str) -> dict[str, str] | None:
        result = await self.client.hgetall(self._runner_key(runner_id))
        return result or None

    async def enqueue(self, runner_id: str, job_id: str) -> None:
        await self.client.rpush(self._queue_key(runner_id), job_id)

    async def dequeue(self, runner_id: str) -> str | None:
        return await self.client.lpop(self._queue_key(runner_id))

    async def remove_queued_job(self, runner_id: str, job_id: str) -> bool:
        return bool(await self.client.lrem(self._queue_key(runner_id), 1, job_id))

    async def queue_depth(self, runner_id: str) -> int:
        return int(await self.client.llen(self._queue_key(runner_id)))

    async def clear_queue(self, runner_id: str) -> None:
        await self.client.delete(self._queue_key(runner_id))

    async def save_job(self, job_id: str, fields: dict[str, Any]) -> None:
        await self.client.hset(self._job_key(job_id), mapping=_redis_mapping(fields))

    async def update_job(self, job_id: str, **fields: Any) -> None:
        await self.save_job(job_id, fields)

    async def get_job(self, job_id: str) -> dict[str, str] | None:
        result = await self.client.hgetall(self._job_key(job_id))
        return result or None

    async def add_expiry(self, job_id: str, expires_at: int) -> None:
        await self.client.zadd(EXPIRY_KEY, {job_id: expires_at})

    async def due_expiries(self, now: int) -> list[str]:
        return list(await self.client.zrangebyscore(EXPIRY_KEY, "-inf", now))

    async def remove_expiry(self, job_id: str) -> None:
        await self.client.zrem(EXPIRY_KEY, job_id)

    async def delete_job(self, job_id: str) -> None:
        await self.client.delete(self._job_key(job_id))

    async def expire_runner(self, runner_id: str, seconds: int) -> None:
        await self.client.expire(self._runner_key(runner_id), seconds)

    async def recover_nonterminal_jobs(self) -> list[tuple[str, dict[str, str]]]:
        stale: list[tuple[str, dict[str, str]]] = []
        async for key in self.client.scan_iter(match="flexy:job:*"):
            job_id = str(key).rsplit(":", 1)[-1]
            record = await self.get_job(job_id)
            if record and record.get("state") not in TERMINAL_JOB_STATES:
                stale.append((job_id, record))
        return stale

    @staticmethod
    def _runner_key(runner_id: str) -> str:
        return f"flexy:runner:{runner_id}"

    @staticmethod
    def _queue_key(runner_id: str) -> str:
        return f"flexy:runner:{runner_id}:queue"

    @staticmethod
    def _job_key(job_id: str) -> str:
        return f"flexy:job:{job_id}"


class ArtifactStore:
    """Owner-scoped temporary media; Redis stores IDs, never these paths."""

    _suffixes = {
        "image": {".jpg", ".jpeg", ".png", ".webp"},
        "video": {".mp4", ".mov", ".mkv", ".webm"},
        "audio": {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"},
    }

    def __init__(self, root: Path, max_upload_bytes: int):
        self.root = root.resolve()
        self.max_upload_bytes = max_upload_bytes

    async def save_uploads(
        self,
        owner_id: str,
        job_id: str,
        definition: AppDefinition,
        uploads: dict[str, list[Any]],
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, list[Path]]:
        try:
            if cancelled is not None and cancelled():
                raise PlatformError("RUNNER_NOT_READY", "Runner is not ready for inference.", 409)
            unknown = set(uploads) - set(definition.uploads)
            if unknown:
                raise PlatformError("INVALID_REQUEST", "Unsupported upload field.", 422)
            directory = self.input_directory(owner_id, job_id)
            directory.mkdir(parents=True, exist_ok=False)
            saved: dict[str, list[Path]] = {}
            total = 0
            for field, files in uploads.items():
                rule = definition.uploads[field]
                if not rule.multiple and len(files) > 1:
                    raise PlatformError("INVALID_REQUEST", "Too many files for an upload field.", 422)
                for index, upload in enumerate(files):
                    if cancelled is not None and cancelled():
                        raise PlatformError("RUNNER_NOT_READY", "Runner is not ready for inference.", 409)
                    suffix = self._safe_suffix(upload, rule.media_type)
                    target = directory / f"{field}-{index}{suffix}"
                    with target.open("xb") as destination:
                        while chunk := await upload.read(64 * 1024):
                            if cancelled is not None and cancelled():
                                raise PlatformError("RUNNER_NOT_READY", "Runner is not ready for inference.", 409)
                            total += len(chunk)
                            if total > self.max_upload_bytes:
                                raise PlatformError("UPLOAD_TOO_LARGE", "Upload exceeds the configured size limit.", 413)
                            destination.write(chunk)
                    saved.setdefault(field, []).append(target)
            return saved
        except Exception:
            self.remove_job(owner_id, job_id)
            raise
        finally:
            for files in uploads.values():
                for upload in files:
                    try:
                        await upload.close()
                    except Exception:
                        pass

    def output_path(self, owner_id: str, job_id: str) -> Path:
        directory = self.job_directory(owner_id, job_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "output.mp4"

    def input_directory(self, owner_id: str, job_id: str) -> Path:
        return self.job_directory(owner_id, job_id) / "input"

    def job_directory(self, owner_id: str, job_id: str) -> Path:
        path = (self.root / owner_id / job_id).resolve()
        if not path.is_relative_to(self.root):
            raise PlatformError("INVALID_REQUEST", "Invalid artifact identifier.", 422)
        return path

    def output_exists(self, owner_id: str, job_id: str) -> bool:
        return self.output_path(owner_id, job_id).is_file()

    def remove_inputs(self, owner_id: str, job_id: str) -> None:
        shutil.rmtree(self.input_directory(owner_id, job_id), ignore_errors=True)

    def remove_job(self, owner_id: str, job_id: str) -> bool:
        directory = self.job_directory(owner_id, job_id)
        if directory.exists():
            try:
                shutil.rmtree(directory)
            except OSError:
                return False
        try:
            directory.parent.rmdir()
        except OSError:
            pass
        return not directory.exists()

    def _safe_suffix(self, upload: Any, media_type: str) -> str:
        content_type = str(getattr(upload, "content_type", "") or "").lower()
        if content_type and content_type != "application/octet-stream" and not content_type.startswith(f"{media_type}/"):
            raise PlatformError("INVALID_REQUEST", "Upload media type does not match the input field.", 422)
        suffix = Path(str(getattr(upload, "filename", "") or "")).suffix.lower()
        if suffix not in self._suffixes[media_type]:
            raise PlatformError("INVALID_REQUEST", "Upload file type is not supported.", 422)
        return suffix


@dataclass
class LocalJob:
    id: str
    owner_id: str
    payload: dict[str, Any] | None


@dataclass
class LocalRunner:
    id: str
    owner_id: str
    definition: AppDefinition
    gpu_id: int
    process: Any
    commands: Any
    events: Any
    state: str = "loading"
    queued: deque[LocalJob] = field(default_factory=deque)
    active: LocalJob | None = None
    pending_jobs: dict[str, str] = field(default_factory=dict)
    pending_event: dict[str, Any] | None = None
    failure_code: str | None = None


class FlexyRuntime:
    """Owns the only permitted child runner and its in-memory command payloads."""

    def __init__(
        self,
        state: RedisState,
        secret: str,
        settings: RuntimeSettings | None = None,
        resource_settings: ResourceSettings | None = None,
    ) -> None:
        self.state = state
        self.secret = secret.encode("utf-8")
        self.settings = settings or RuntimeSettings.from_environment()
        self.artifacts = ArtifactStore(self.settings.artifact_root, self.settings.max_upload_bytes)
        self.resources = ResourceManager(resource_settings, stop_runner=self._resource_stop_requested)
        self.runner: LocalRunner | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ticker: asyncio.Task[None] | None = None
        self._runner_lock = asyncio.Lock()
        self._stopping = False
        self._resource_stop_pending = False
        self._ticker_error: str | None = None

    async def start(self) -> None:
        self.settings.artifact_root.mkdir(parents=True, exist_ok=True)
        await self.state.start()
        await self._recover_after_restart()
        self._loop = asyncio.get_running_loop()
        self._ticker = asyncio.create_task(self._run_ticker())

    async def stop(self) -> None:
        self._stopping = True
        if self._ticker is not None:
            self._ticker.cancel()
            with _suppress_cancelled():
                await self._ticker
        if self.runner is not None:
            await self._terminate_runner("SERVER_STOPPING")
        shutdown_gpu_monitor()
        await self.state.close()

    async def load_runner(self, owner_id: str, app_name: str, config: dict[str, Any], gpu_id: int) -> dict[str, str]:
        async with self._runner_lock:
            return await self._load_runner(owner_id, app_name, config, gpu_id)

    async def _load_runner(self, owner_id: str, app_name: str, config: dict[str, Any], gpu_id: int) -> dict[str, str]:
        try:
            definition = get_app(app_name)
            safe_config = definition.validate_config(config)
        except ValueError as exc:
            raise PlatformError("INVALID_REQUEST", str(exc), 422) from None
        if not isinstance(gpu_id, int) or isinstance(gpu_id, bool) or gpu_id < 0:
            raise PlatformError("INVALID_REQUEST", "GPU device must be a non-negative integer.", 422)
        decision = self.resources.admission_decision(gpu_id=gpu_id, cpu_offload=definition.uses_cpu_offload)
        if not decision.allowed:
            raise PlatformError(decision.code or "RESOURCE_UNAVAILABLE", "Resources are not safe for a new runner.", 503)

        runner_id = str(uuid4())
        if not await self.state.acquire_active_runner(runner_id):
            raise PlatformError("RUNNER_ALREADY_ACTIVE", "Only one runner may be loaded at a time.", 409)
        now = _now()
        try:
            await self.state.save_runner(runner_id, {
                "owner_id": owner_id,
                "app": definition.name,
                "app_version": definition.version,
                "state": "loading",
                "gpu_id": gpu_id,
                "created_at": now,
                "updated_at": now,
                "status_code": "LOADING",
            })
        except Exception:
            await _release_runner_slot_quietly(self.state, runner_id)
            raise PlatformError("RUNTIME_UNAVAILABLE", "The runner state store is unavailable.", 503) from None
        process = None
        try:
            context = multiprocessing.get_context("spawn")
            commands = context.Queue()
            events = context.Queue()
            process = context.Process(
                target=run_worker,
                args=(definition.name, safe_config, gpu_id, commands, events),
                daemon=True,
            )
            process.start()
            await self.state.update_runner(
                runner_id,
                worker_pid=process.pid,
                worker_started_at=_process_started_at(process.pid),
                updated_at=_now(),
            )
        except Exception:
            if process is not None:
                await _terminate_process(process)
            try:
                await self.state.update_runner(
                    runner_id, state="failed", status_code="RUNNER_START_FAILED", updated_at=_now())
            except Exception:
                pass
            await _release_runner_slot_quietly(self.state, runner_id)
            raise PlatformError("RUNNER_START_FAILED", "The runner process could not start.", 503) from None
        self.runner = LocalRunner(runner_id, owner_id, definition, gpu_id, process, commands, events)
        self.resources.set_runner(RunnerInfo(runner_id, process.pid, gpu_id, definition.uses_cpu_offload, False))
        return {"runner_uuid": runner_id, "status": "loading"}

    async def submit_inference(
        self,
        owner_id: str,
        runner_id: str,
        raw_payload: dict[str, Any],
        uploads: dict[str, list[Any]],
    ) -> dict[str, Any]:
        job_id = str(uuid4())
        async with self._runner_lock:
            runner = await self._owned_local_runner(owner_id, runner_id)
            if runner.state not in {"ready", "busy"}:
                raise PlatformError("RUNNER_NOT_READY", "Runner is not ready for inference.", 409)
            saved_job = False
            try:
                now = _now()
                await self.state.save_job(job_id, {
                    "owner_id": owner_id,
                    "runner_uuid": runner_id,
                    "state": "queued",
                    "created_at": now,
                    "updated_at": now,
                    "status_code": "QUEUED",
                })
                saved_job = True
                await self.state.enqueue(runner_id, job_id)
                # Reserve the FIFO position before awaiting media writes. A later,
                # smaller upload must not overtake this request.
                runner.queued.append(LocalJob(job_id, owner_id, None))
                runner.pending_jobs[job_id] = owner_id
            except Exception:
                if saved_job:
                    try:
                        await self.state.remove_queued_job(runner_id, job_id)
                        await self.state.update_job(
                            job_id,
                            state="failed",
                            status_code="QUEUE_UNAVAILABLE",
                            updated_at=_now(),
                        )
                        await self.state.add_expiry(job_id, _now() + self.settings.artifact_ttl_seconds)
                    except Exception:
                        pass
                raise PlatformError("QUEUE_UNAVAILABLE", "The job could not be queued.", 503) from None
        try:
            try:
                files = await self.artifacts.save_uploads(
                    owner_id,
                    job_id,
                    runner.definition,
                    uploads,
                    cancelled=lambda: job_id not in runner.pending_jobs,
                )
                output_path = self.artifacts.output_path(owner_id, job_id)
                payload = runner.definition.build_payload(raw_payload, files, output_path)
            except (PlatformError, ValueError) as exc:
                self.artifacts.remove_job(owner_id, job_id)
                if isinstance(exc, PlatformError):
                    raise
                raise PlatformError("INVALID_REQUEST", "Invalid inference request.", 422) from None
            async with self._runner_lock:
                if self.runner is not runner or runner.state not in {"ready", "busy"}:
                    self.artifacts.remove_job(owner_id, job_id)
                    if self.runner is runner and job_id in runner.pending_jobs:
                        await self._discard_pending_submission(runner, job_id)
                    raise PlatformError("RUNNER_NOT_READY", "Runner is not ready for inference.", 409)
                job = next((item for item in runner.queued if item.id == job_id), None)
                if job is None or job_id not in runner.pending_jobs:
                    self.artifacts.remove_job(owner_id, job_id)
                    raise PlatformError("RUNNER_NOT_READY", "Runner is not ready for inference.", 409)
                job.payload = payload
                runner.pending_jobs.pop(job_id, None)
                return {"success": True, "job_uuid": job_id, "status": "queued"}
        except BaseException:
            async with self._runner_lock:
                if self.runner is runner and job_id in runner.pending_jobs:
                    await self._discard_pending_submission(runner, job_id)
            raise

    async def _discard_pending_submission(self, runner: LocalRunner, job_id: str) -> None:
        """Remove an unmaterialized FIFO reservation after an upload failure."""
        if self.runner is not runner or job_id not in runner.pending_jobs:
            return
        try:
            if not await self.state.remove_queued_job(runner.id, job_id):
                raise RuntimeError("Redis queue no longer contains the pending job")
            await self.state.delete_job(job_id)
        except Exception:
            try:
                await self._fail_runner("QUEUE_UNAVAILABLE")
            except Exception:
                pass
            raise PlatformError("QUEUE_UNAVAILABLE", "The job queue is unavailable.", 503) from None
        runner.pending_jobs.pop(job_id, None)
        queued_job = next((item for item in runner.queued if item.id == job_id), None)
        if queued_job is not None:
            runner.queued.remove(queued_job)

    async def unload_runner(self, owner_id: str, runner_id: str) -> dict[str, str]:
        async with self._runner_lock:
            runner = await self._owned_local_runner(owner_id, runner_id)
            if runner.state == "busy" or runner.queued or runner.pending_jobs:
                raise PlatformError("RUNNER_BUSY", "Queued or active work must finish before unloading.", 409)
            if runner.state != "ready":
                raise PlatformError("RUNNER_NOT_READY", "Runner cannot be unloaded in its current state.", 409)
            try:
                await self.state.update_runner(
                    runner_id, state="stopping", status_code="UNLOADING", updated_at=_now())
            except Exception:
                runner.state = "failed"
                runner.failure_code = "RUNTIME_STATE_UNAVAILABLE"
                await _terminate_process(runner.process)
                raise PlatformError("RUNTIME_UNAVAILABLE", "The runner state store is unavailable.", 503) from None
            runner.state = "stopping"
            try:
                runner.commands.put({"type": "shutdown"})
            except Exception:
                runner.state = "failed"
                runner.failure_code = "RUNNER_SHUTDOWN_FAILED"
                await _terminate_process(runner.process)
                raise PlatformError("RUNNER_SHUTDOWN_FAILED", "The runner could not be stopped.", 503) from None
            return {"runner_uuid": runner_id, "status": "unloading"}

    async def get_owned_runner(self, owner_id: str, runner_id: str) -> dict[str, str]:
        runner = await self.state.get_runner(runner_id)
        if runner is None or runner.get("owner_id") != owner_id:
            raise PlatformError("RESOURCE_NOT_FOUND", "Requested resource was not found.", 404)
        return runner

    async def get_owned_job(self, owner_id: str, runner_id: str, job_id: str) -> dict[str, str]:
        job = await self.state.get_job(job_id)
        if job is None or job.get("owner_id") != owner_id or job.get("runner_uuid") != runner_id:
            raise PlatformError("RESOURCE_NOT_FOUND", "Requested resource was not found.", 404)
        return job

    async def runner_status(self, owner_id: str, runner_id: str) -> dict[str, Any]:
        runner = await self.get_owned_runner(owner_id, runner_id)
        resource = self.resources.poll().to_dict()
        selected = next((gpu for gpu in resource["gpus"] if gpu["id"] == int(runner["gpu_id"])), None)
        result: dict[str, Any] = {
            "runner_uuid": runner_id,
            "app": runner["app"],
            "status": runner["state"],
            "status_code": runner.get("status_code"),
            "metadata": {
                "app_version": runner.get("app_version"),
                "gpu_id": int(runner["gpu_id"]),
                "queue_depth": await self.state.queue_depth(runner_id),
                "resource": {"host": resource["host"], "gpu": selected},
            },
        }
        if self.runner is not None and self.runner.id == runner_id:
            rss = _process_rss_bytes(self.runner.process.pid)
            if rss is not None:
                result["metadata"]["estimated_runner_ram_gb"] = round(rss / 1024**3, 2)
        if self._ticker_error is not None:
            result["metadata"]["runtime_status_code"] = self._ticker_error
        return result

    async def job_status(self, owner_id: str, runner_id: str, job_id: str) -> dict[str, Any]:
        job = await self.get_owned_job(owner_id, runner_id, job_id)
        result: dict[str, Any] = {"job_uuid": job_id, "status": job["state"]}
        if job.get("status_code"):
            result["status_code"] = job["status_code"]
        if job.get("state") == "completed":
            artifact_expires_at = int(job["expires_at"])
            download_expires_at = min(
                artifact_expires_at,
                _now() + self.settings.download_url_ttl_seconds,
            )
            result["download_expires_at"] = download_expires_at
            result["download_signature"] = self.download_signature(owner_id, job_id, download_expires_at)
        return result

    async def resource_status(self) -> dict[str, Any]:
        return self.resources.poll().to_dict()

    def download_signature(self, owner_id: str, job_id: str, expires_at: int) -> str:
        payload = f"{owner_id}:{job_id}:{expires_at}".encode("utf-8")
        return hmac.new(self.secret, payload, hashlib.sha256).hexdigest()

    async def download_path(self, owner_id: str, job_id: str, expires_at: int, signature: str) -> Path:
        job = await self.state.get_job(job_id)
        if (
            job is None
            or job.get("owner_id") != owner_id
            or job.get("state") != "completed"
            or expires_at < _now()
            or expires_at > int(job.get("expires_at", 0))
        ):
            raise PlatformError("RESOURCE_NOT_FOUND", "Requested resource was not found.", 404)
        expected = self.download_signature(owner_id, job_id, expires_at)
        if not secrets.compare_digest(signature, expected):
            raise PlatformError("RESOURCE_NOT_FOUND", "Requested resource was not found.", 404)
        path = self.artifacts.output_path(owner_id, job_id)
        if not path.is_file():
            raise PlatformError("RESOURCE_NOT_FOUND", "Requested resource was not found.", 404)
        return path

    async def _owned_local_runner(self, owner_id: str, runner_id: str) -> LocalRunner:
        await self.get_owned_runner(owner_id, runner_id)
        if self.runner is None or self.runner.id != runner_id:
            raise PlatformError("RUNNER_NOT_READY", "Runner is not available in this server process.", 409)
        return self.runner

    async def _run_ticker(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Redis reconnects automatically. Keep polling resource safety rather
                # than allowing a transient state-store outage to kill the monitor.
                self._ticker_error = "RUNTIME_STATE_UNAVAILABLE"
            else:
                self._ticker_error = None
            await asyncio.sleep(self.resources.settings.resource_poll_seconds)

    async def tick(self) -> None:
        async with self._runner_lock:
            self.resources.poll()
            await self._drain_worker_events()
            await self._check_worker_process()
            await self._dispatch_next_job()
            await self._cleanup_expired_artifacts()

    async def _drain_worker_events(self) -> None:
        runner = self.runner
        if runner is None:
            return
        while self.runner is runner:
            event = runner.pending_event
            if event is None:
                try:
                    event = runner.events.get_nowait()
                except Empty:
                    return
                runner.pending_event = event
            kind = event.get("type")
            stop_draining = False
            if kind == "ready":
                runner.state = "ready"
                await self.state.update_runner(runner.id, state="ready", status_code="READY", updated_at=_now())
                self.resources.set_runner(RunnerInfo(runner.id, runner.process.pid, runner.gpu_id,
                                                     runner.definition.uses_cpu_offload, False))
            elif kind == "load_failed":
                await self._fail_runner(event.get("code", "RUNNER_LOAD_FAILED"))
                stop_draining = True
            elif kind == "job_completed" and runner.active and event.get("job_id") == runner.active.id:
                await self._complete_active_job()
            elif kind == "job_failed" and runner.active and event.get("job_id") == runner.active.id:
                await self._fail_active_job(event.get("code", "INFERENCE_FAILED"))
            elif kind == "stopped":
                if runner.state == "stopping":
                    await _terminate_process(runner.process)
                    await self._finish_runner("stopped", "STOPPED")
                elif runner.state != "failed":
                    await self._fail_runner("RUNNER_STOPPED")
                stop_draining = True
            # Do not lose a terminal worker event during a temporary Redis outage.
            # Clearing occurs only after its state/artifact transition has completed.
            runner.pending_event = None
            if stop_draining or self.runner is not runner:
                return

    async def _check_worker_process(self) -> None:
        runner = self.runner
        if runner is None:
            return
        if runner.state == "failed":
            await self._fail_runner(runner.failure_code or "RUNNER_FAILED")
            return
        if runner.process.is_alive():
            return
        if runner.state == "stopping":
            await self._finish_runner("stopped", "STOPPED")
        else:
            await self._fail_runner("RUNNER_CRASHED")

    async def _dispatch_next_job(self) -> None:
        runner = self.runner
        if runner is None or runner.state != "ready" or runner.active is not None or not runner.queued:
            return
        snapshot = self.resources.last_snapshot or self.resources.poll()
        if not _safe_to_schedule(snapshot, runner.gpu_id, runner.definition.uses_cpu_offload):
            return
        job = runner.queued[0]
        if job.payload is None:
            return
        runner.queued.popleft()
        runner.active = job
        runner.state = "busy"
        self.resources.set_runner(RunnerInfo(runner.id, runner.process.pid, runner.gpu_id,
                                             runner.definition.uses_cpu_offload, True))
        try:
            dequeued_job_id = await self.state.dequeue(runner.id)
            if dequeued_job_id != job.id:
                raise RuntimeError("Redis queue no longer matches the local FIFO")
            await self.state.update_runner(runner.id, state="busy", status_code="BUSY", updated_at=_now())
            await self.state.update_job(job.id, state="running", status_code="RUNNING", updated_at=_now())
        except Exception:
            await self._fail_runner("QUEUE_UNAVAILABLE")
            return
        runner.commands.put({"type": "infer", "job_id": job.id, "payload": job.payload})

    async def _complete_active_job(self) -> None:
        runner = self.runner
        if runner is None or runner.active is None:
            return
        job = runner.active
        if not self.artifacts.output_exists(job.owner_id, job.id):
            await self._fail_active_job("OUTPUT_MISSING")
            return
        expires_at = _now() + self.settings.artifact_ttl_seconds
        self.artifacts.remove_inputs(job.owner_id, job.id)
        await self.state.update_job(job.id, state="completed", status_code="COMPLETED",
                                    updated_at=_now(), expires_at=expires_at, artifact_name="output.mp4")
        await self.state.add_expiry(job.id, expires_at)
        await self.state.update_runner(runner.id, state="ready", status_code="READY", updated_at=_now())
        runner.active = None
        runner.state = "ready"
        self.resources.set_runner(RunnerInfo(runner.id, runner.process.pid, runner.gpu_id,
                                             runner.definition.uses_cpu_offload, False))

    async def _fail_active_job(self, code: str) -> None:
        runner = self.runner
        if runner is None or runner.active is None:
            return
        job = runner.active
        cleanup_at = self._failed_artifact_cleanup_at(job.owner_id, job.id)
        await self.state.update_job(job.id, state="failed", status_code=code, updated_at=_now())
        await self.state.add_expiry(job.id, cleanup_at)
        if runner.state != "failed":
            await self.state.update_runner(runner.id, state="ready", status_code="READY", updated_at=_now())
            runner.active = None
            runner.state = "ready"
            self.resources.set_runner(RunnerInfo(runner.id, runner.process.pid, runner.gpu_id,
                                                 runner.definition.uses_cpu_offload, False))
        else:
            runner.active = None

    async def _fail_runner(self, code: str) -> None:
        runner = self.runner
        if runner is None:
            return
        runner.state = "failed"
        runner.failure_code = code
        runner.pending_jobs.clear()
        await _terminate_process(runner.process)
        if runner.active is not None:
            await self._fail_active_job(code)
        while runner.queued:
            job = runner.queued.popleft()
            cleanup_at = self._failed_artifact_cleanup_at(job.owner_id, job.id)
            await self.state.update_job(job.id, state="failed", status_code=code, updated_at=_now())
            await self.state.add_expiry(job.id, cleanup_at)
        await self.state.clear_queue(runner.id)
        await self._finish_runner("failed", code)

    async def _terminate_runner(self, code: str) -> None:
        async with self._runner_lock:
            if self.runner is not None:
                await self._fail_runner(code)

    async def _finish_runner(self, state: str, status_code: str) -> None:
        runner = self.runner
        if runner is None:
            return
        await self.state.update_runner(runner.id, state=state, status_code=status_code, updated_at=_now())
        await self.state.release_active_runner(runner.id)
        try:
            await self.state.expire_runner(runner.id, self.settings.artifact_ttl_seconds)
        except Exception:
            self._ticker_error = "RUNTIME_STATE_UNAVAILABLE"
        self.resources.set_runner(None)
        self.runner = None

    def _failed_artifact_cleanup_at(self, owner_id: str, job_id: str) -> int:
        """Retry locked failed-artifact directories on the next scheduler pass."""
        if self.artifacts.remove_job(owner_id, job_id):
            return _now() + self.settings.artifact_ttl_seconds
        return _now()

    async def _cleanup_expired_artifacts(self) -> None:
        for job_id in await self.state.due_expiries(_now()):
            job = await self.state.get_job(job_id)
            if job is not None:
                if not self.artifacts.remove_job(job["owner_id"], job_id):
                    continue
                await self.state.delete_job(job_id)
            await self.state.remove_expiry(job_id)

    async def _recover_after_restart(self) -> None:
        active = await self.state.active_runner_id()
        if active:
            record = await self.state.get_runner(active)
            worker_state = await asyncio.to_thread(_recorded_worker_state, record)
            if worker_state == "alive":
                worker_state = await asyncio.to_thread(_terminate_recorded_worker, record)
            if worker_state == "gone":
                await self.state.update_runner(
                    active, state="failed", status_code="SERVER_RESTARTED", updated_at=_now())
                await self.state.clear_queue(active)
                await self.state.release_active_runner(active)
                await self.state.expire_runner(active, self.settings.artifact_ttl_seconds)
            else:
                # Without a matching PID identity, releasing the lock could allow
                # a second unmonitored model process to use the host and GPU.
                await self.state.update_runner(
                    active,
                    state="failed",
                    status_code="ORPHANED_RUNNER_UNCONFIRMED",
                    updated_at=_now(),
                )
        for job_id, job in await self.state.recover_nonterminal_jobs():
            cleanup_at = self._failed_artifact_cleanup_at(job["owner_id"], job_id)
            await self.state.update_job(job_id, state="failed", status_code="SERVER_RESTARTED", updated_at=_now())
            await self.state.add_expiry(job_id, cleanup_at)
        await self._cleanup_expired_artifacts()

    def _resource_stop_requested(self, runner_info: RunnerInfo, reason: str) -> None:
        if self._loop is None or self._stopping or self._resource_stop_pending:
            return
        self._resource_stop_pending = True

        def stop() -> None:
            asyncio.create_task(self._stop_for_resource(runner_info.runner_id, reason))

        self._loop.call_soon_threadsafe(stop)

    async def _stop_for_resource(self, runner_id: str, reason: str) -> None:
        try:
            if self.runner is not None and self.runner.id == runner_id:
                await self._terminate_runner(reason)
        finally:
            self._resource_stop_pending = False


def create_runtime(secret: str) -> FlexyRuntime:
    """Build the production runtime lazily so auth-only tests need no Redis package."""
    from redis.asyncio import Redis

    settings = RuntimeSettings.from_environment()
    client = Redis.from_url(settings.redis_url, decode_responses=True)
    return FlexyRuntime(RedisState(client), secret, settings)


async def _terminate_process(process: Any) -> None:
    if not process.is_alive():
        return
    process.terminate()
    await asyncio.to_thread(process.join, 5)
    if process.is_alive():
        process.kill()
        await asyncio.to_thread(process.join, 2)


async def _release_runner_slot_quietly(state: RedisState, runner_id: str) -> None:
    try:
        await state.release_active_runner(runner_id)
    except Exception:
        pass


def _process_started_at(pid: int) -> float | None:
    try:
        import psutil

        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _recorded_worker_state(record: dict[str, str] | None) -> str:
    if record is None:
        return "unknown"
    try:
        pid = int(record["worker_pid"])
        expected_started_at = float(record["worker_started_at"])
    except (KeyError, TypeError, ValueError):
        return "unknown"
    try:
        import psutil

        process = psutil.Process(pid)
        if abs(process.create_time() - expected_started_at) > 0.01:
            return "gone"
        return "alive"
    except Exception as exc:
        try:
            import psutil

            if isinstance(exc, psutil.NoSuchProcess):
                return "gone"
        except Exception:
            pass
        return "unknown"


def _terminate_recorded_worker(record: dict[str, str] | None) -> str:
    if _recorded_worker_state(record) != "alive":
        return _recorded_worker_state(record)
    try:
        import psutil

        pid = int(record["worker_pid"])
        expected_started_at = float(record["worker_started_at"])
        process = psutil.Process(pid)
        if abs(process.create_time() - expected_started_at) > 0.01:
            return "gone"
        process.terminate()
        try:
            process.wait(5)
        except psutil.TimeoutExpired:
            process.kill()
            process.wait(2)
    except Exception:
        return "unknown"
    return _recorded_worker_state(record)


def _safe_to_schedule(snapshot: Any, gpu_id: int, cpu_offload: bool) -> bool:
    if not snapshot.host_monitor_available or not snapshot.gpu_monitor_available or snapshot.host is None:
        return False
    if snapshot.host.state not in {ResourceState.NORMAL, ResourceState.WARNING}:
        return False
    if snapshot.host.swap_state not in {ResourceState.NORMAL, ResourceState.WARNING}:
        return False
    if cpu_offload and snapshot.host.offload_state not in {ResourceState.NORMAL, ResourceState.WARNING}:
        return False
    gpu = next((item for item in snapshot.gpus if item.id == gpu_id), None)
    return gpu is not None and gpu.state in {ResourceState.NORMAL, ResourceState.WARNING}


def _redis_mapping(fields: dict[str, Any]) -> dict[str, str]:
    return {key: str(value) for key, value in fields.items() if value is not None}


def _now() -> int:
    return int(time.time())


def _process_rss_bytes(pid: int) -> int | None:
    try:
        import psutil

        return int(psutil.Process(pid).memory_info().rss)
    except Exception:
        return None


class _suppress_cancelled:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: Any) -> bool:
        return exc_type is asyncio.CancelledError
