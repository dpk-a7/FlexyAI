import asyncio
import json
import os
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps._registry import get_app
from flexy_runtime import ArtifactStore, FlexyRuntime, LocalJob, LocalRunner, PlatformError, RuntimeSettings
from resource_manager import ResourceState
import main


class MemoryState:
    def __init__(self):
        self.runners = {}
        self.jobs = {}
        self.queues = {}
        self.expiries = {}
        self.runner_expiries = {}
        self.fail_enqueue = False
        self.fail_dequeue = False
        self.fail_update_job_once = False
        self.fail_update_runner_once = False
        self.active_runner = None

    async def get_runner(self, runner_id):
        return self.runners.get(runner_id)

    async def acquire_active_runner(self, runner_id):
        if self.active_runner is not None:
            return False
        self.active_runner = runner_id
        return True

    async def save_runner(self, runner_id, fields):
        self.runners[runner_id] = {**self.runners.get(runner_id, {}), **fields}

    async def update_runner(self, runner_id, **fields):
        if self.fail_update_runner_once:
            self.fail_update_runner_once = False
            raise ConnectionError("test state store is unavailable")
        await self.save_runner(runner_id, fields)

    async def save_job(self, job_id, fields):
        self.jobs[job_id] = {**self.jobs.get(job_id, {}), **fields}

    async def update_job(self, job_id, **fields):
        if self.fail_update_job_once:
            self.fail_update_job_once = False
            raise ConnectionError("test state store is unavailable")
        await self.save_job(job_id, fields)

    async def get_job(self, job_id):
        return self.jobs.get(job_id)

    async def enqueue(self, runner_id, job_id):
        if self.fail_enqueue:
            raise ConnectionError("test queue is unavailable")
        self.queues.setdefault(runner_id, []).append(job_id)

    async def dequeue(self, runner_id):
        if self.fail_dequeue:
            raise ConnectionError("test queue is unavailable")
        queue = self.queues.setdefault(runner_id, [])
        return queue.pop(0) if queue else None

    async def remove_queued_job(self, runner_id, job_id):
        queue = self.queues.setdefault(runner_id, [])
        try:
            queue.remove(job_id)
        except ValueError:
            return False
        return True

    async def queue_depth(self, runner_id):
        return len(self.queues.get(runner_id, []))

    async def clear_queue(self, runner_id):
        self.queues[runner_id] = []

    async def release_active_runner(self, runner_id):
        if self.active_runner == runner_id:
            self.active_runner = None

    async def active_runner_id(self):
        return self.active_runner

    async def add_expiry(self, job_id, expires_at):
        self.expiries[job_id] = expires_at

    async def remove_expiry(self, job_id):
        self.expiries.pop(job_id, None)

    async def delete_job(self, job_id):
        self.jobs.pop(job_id, None)

    async def expire_runner(self, runner_id, seconds):
        self.runner_expiries[runner_id] = seconds

    async def due_expiries(self, now):
        return [job_id for job_id, expires_at in self.expiries.items() if expires_at <= now]

    async def recover_nonterminal_jobs(self):
        return [
            (job_id, job)
            for job_id, job in self.jobs.items()
            if job.get("state") not in {"completed", "failed", "expired"}
        ]


class FakeUpload:
    def __init__(self, filename, content_type, content):
        self.filename = filename
        self.content_type = content_type
        self.content = content
        self.offset = 0
        self.closed = False

    async def read(self, size):
        result = self.content[self.offset:self.offset + size]
        self.offset += len(result)
        return result

    async def close(self):
        self.closed = True


class DelayedUpload(FakeUpload):
    def __init__(self, filename, content_type, content):
        super().__init__(filename, content_type, content)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.wait_once = True

    async def read(self, size):
        if self.wait_once:
            self.wait_once = False
            self.started.set()
            await self.release.wait()
        return await super().read(size)


class FakeQueue:
    def __init__(self):
        self.messages = []

    def put(self, message):
        self.messages.append(message)

    def get_nowait(self):
        if not self.messages:
            raise Empty
        return self.messages.pop(0)


class FakeProcess:
    pid = 12345

    def is_alive(self):
        return True


class DeadProcess(FakeProcess):
    def is_alive(self):
        return False


class TerminableProcess(FakeProcess):
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    def kill(self):
        self.alive = False

    def join(self, timeout):
        pass


class FakeSpawnProcess(TerminableProcess):
    pid = 4242

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.started = False

    def start(self):
        self.started = True


class FakeSpawnContext:
    def __init__(self):
        self.process = None

    def Queue(self):
        return FakeQueue()

    def Process(self, **kwargs):
        self.process = FakeSpawnProcess(**kwargs)
        return self.process


class FakeResources:
    def __init__(self):
        self.settings = SimpleNamespace(resource_poll_seconds=1)
        self.last_snapshot = SimpleNamespace(
            host_monitor_available=True,
            gpu_monitor_available=True,
            host=SimpleNamespace(
                state=ResourceState.NORMAL,
                swap_state=ResourceState.NORMAL,
                offload_state=ResourceState.NORMAL,
            ),
            gpus=(SimpleNamespace(id=0, state=ResourceState.NORMAL),),
        )
        self.runner = None

    def poll(self):
        return self.last_snapshot

    def set_runner(self, runner):
        self.runner = runner


class FakeLoadResources(FakeResources):
    def admission_decision(self, **kwargs):
        return SimpleNamespace(allowed=True, code=None)


class ApiRuntime:
    def __init__(self):
        self.settings = SimpleNamespace(max_upload_bytes=1024)
        self.submissions = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def submit_inference(self, owner_id, runner_id, payload, uploads):
        self.submissions.append((owner_id, runner_id, payload, uploads))
        for files in uploads.values():
            for upload in files:
                await upload.close()
        return {"success": True, "job_uuid": "job-1", "status": "queued"}

    async def job_status(self, owner_id, runner_id, job_id):
        return {
            "job_uuid": job_id,
            "status": "completed",
            "download_expires_at": 123,
            "download_signature": "signature",
        }


class RegistryTests(unittest.TestCase):
    def test_public_app_config_and_payload_cannot_supply_server_paths(self):
        app = get_app("mmh3_fi2v")
        with self.assertRaisesRegex(ValueError, "Invalid app load configuration"):
            app.validate_config({"model_root": "C:/private/models"})
        with self.assertRaisesRegex(ValueError, "Invalid inference payload"):
            app.build_payload(
                {"prompt": "A short animation", "output_path": "C:/private/output.mp4"},
                {"first_frame": [Path("server-input.png")]},
                Path("server-output.mp4"),
            )
        with self.assertRaisesRegex(ValueError, "Unsupported Flexy app"):
            get_app("not-an-app")

    def test_runtime_settings_build_an_authenticated_loopback_redis_url(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {
                "ARTIFACT_ROOT": temporary,
                "REDIS_PASSWORD": "p@ss word",
                "REDIS_PORT": "6380",
            }, clear=True):
                settings = RuntimeSettings.from_environment()
        self.assertEqual(settings.redis_url, "redis://:p%40ss%20word@127.0.0.1:6380/0")

    def test_runtime_settings_require_a_dedicated_artifact_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ARTIFACT_ROOT"):
                RuntimeSettings.from_environment()

    def test_runtime_settings_require_authenticated_redis(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"ARTIFACT_ROOT": temporary}, clear=True):
                with self.assertRaisesRegex(ValueError, "REDIS_PASSWORD"):
                    RuntimeSettings.from_environment()
            with patch.dict(os.environ, {
                "ARTIFACT_ROOT": temporary,
                "REDIS_URL": "redis://127.0.0.1:6379/0",
            }, clear=True):
                with self.assertRaisesRegex(ValueError, "authenticated"):
                    RuntimeSettings.from_environment()


class ApiBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.runtime = ApiRuntime()
        self.enterContext(patch.object(main, "DATABASE_PATH", Path(temporary.name) / "users.sqlite3"))
        self.enterContext(patch.dict(os.environ, {
            "APP_SECRET_KEY": "test-only-signing-secret-at-least-32-bytes",
            "TOKEN_EXPIRATION": "30m",
            "SUPERUSER_USERNAME": "admin",
            "SUPERUSER_PASSWORD": "test-admin-password",
        }))
        self.enterContext(patch.object(main, "create_runtime", return_value=self.runtime))
        self.client = self.enterContext(TestClient(main.app))

    def token(self):
        response = self.client.post("/v1/api/get_token", json={
            "username": "admin", "password": "test-admin-password",
        })
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["token"]

    def inference_request(self, token, **data):
        fields = {"runner_uuid": "runner-1", "payload": json.dumps({"prompt": "private prompt"})}
        fields.update(data)
        return self.client.post(
            "/v1/api/inference",
            headers={"Authorization": f"Bearer {token}"},
            data=fields,
            files={"first_frame": ("frame.png", b"frame", "image/png")},
        )

    def test_inference_requires_jwt_and_parses_only_the_documented_multipart_fields(self):
        response = self.client.post("/v1/api/inference")
        self.assertEqual(response.status_code, 401)

        token = self.token()
        response = self.inference_request(token)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"success": True, "job_uuid": "job-1", "status": "queued"})
        owner_id, runner_id, payload, uploads = self.runtime.submissions[0]
        self.assertEqual(owner_id, "1")
        self.assertEqual(runner_id, "runner-1")
        self.assertEqual(payload, {"prompt": "private prompt"})
        self.assertEqual(set(uploads), {"first_frame"})

        response = self.inference_request(token, server_path="D:/private/output.mp4")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")

    def test_content_length_is_rejected_before_multipart_is_spooled(self):
        self.runtime.settings.max_upload_bytes = 1
        response = self.inference_request(self.token())
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "UPLOAD_TOO_LARGE")
        self.assertEqual(self.runtime.submissions, [])

    def test_media_upload_requires_content_length_before_form_parsing(self):
        request = SimpleNamespace(headers={})
        with self.assertRaises(PlatformError) as raised:
            asyncio.run(main.parse_inference_form(request))
        self.assertEqual(raised.exception.code, "CONTENT_LENGTH_REQUIRED")

    def test_completed_status_returns_a_relative_signed_download_path(self):
        response = self.client.get(
            "/v1/api/inference_status?runner_uuid=runner-1&job_uuid=job-1",
            headers={"Authorization": f"Bearer {self.token()}", "Host": "untrusted.example"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["download_url"],
            "/v1/api/artifacts/job-1/output.mp4?expires=123&signature=signature",
        )


class ArtifactStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_uploads_use_generated_owner_scoped_paths_and_cleanup_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            store = ArtifactStore(root, max_upload_bytes=32)
            app = get_app("mmh3_fi2v")
            upload = FakeUpload("../../private.png", "image/png", b"image-data")
            saved = await store.save_uploads("owner-1", "job-1", app, {"first_frame": [upload]})
            saved_path = saved["first_frame"][0]
            self.assertEqual(saved_path.name, "first_frame-0.png")
            self.assertEqual(saved_path.read_bytes(), b"image-data")
            self.assertTrue(saved_path.is_relative_to(root.resolve()))
            self.assertTrue(upload.closed)

            too_large = FakeUpload("image.png", "image/png", b"x" * 33)
            with self.assertRaises(PlatformError) as raised:
                await store.save_uploads("owner-1", "job-2", app, {"first_frame": [too_large]})
            self.assertEqual(raised.exception.code, "UPLOAD_TOO_LARGE")
            self.assertFalse(store.job_directory("owner-1", "job-2").exists())


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def make_runtime(self, root):
        state = MemoryState()
        settings = RuntimeSettings(
            redis_url="redis://test",
            artifact_root=root,
            artifact_ttl_seconds=300,
            download_url_ttl_seconds=30,
            max_upload_bytes=1024,
        )
        return FlexyRuntime(state, "test-signing-secret", settings), state

    def attach_ready_runner(self, runtime, state, owner="owner-1", runner_id="runner-1"):
        commands = FakeQueue()
        runtime.runner = LocalRunner(
            runner_id,
            owner,
            get_app("mmh3_fi2v"),
            0,
            FakeProcess(),
            commands,
            FakeQueue(),
            state="ready",
        )
        runtime.resources = FakeResources()
        state.runners[runner_id] = {
            "owner_id": owner,
            "app": "mmh3_fi2v",
            "app_version": "1",
            "state": "ready",
            "gpu_id": "0",
        }
        return commands

    async def test_submission_keeps_payload_out_of_state_and_queue_failure_removes_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            runtime, state = self.make_runtime(root)
            self.attach_ready_runner(runtime, state)
            result = await runtime.submit_inference(
                "owner-1",
                "runner-1",
                {"prompt": "private prompt", "seed": 5},
                {"first_frame": [FakeUpload("frame.png", "image/png", b"frame")]},
            )
            self.assertTrue(result["success"])
            self.assertEqual(result["status"], "queued")
            self.assertNotIn("private prompt", str(state.jobs))
            self.assertEqual(state.queues["runner-1"], [result["job_uuid"]])
            self.assertEqual(len(runtime.runner.queued), 1)

            state.fail_enqueue = True
            with self.assertRaises(PlatformError) as raised:
                await runtime.submit_inference(
                    "owner-1",
                    "runner-1",
                    {"prompt": "must be removed"},
                    {"first_frame": [FakeUpload("frame.png", "image/png", b"frame")]},
                )
            self.assertEqual(raised.exception.code, "QUEUE_UNAVAILABLE")
            owner_directory = root / "owner-1"
            self.assertEqual(len(list(owner_directory.iterdir())), 1)

    async def test_load_persists_worker_identity_before_exposing_the_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, state = self.make_runtime(Path(temporary) / "artifacts")
            runtime.resources = FakeLoadResources()
            context = FakeSpawnContext()
            with patch("flexy_runtime.multiprocessing.get_context", return_value=context), patch(
                "flexy_runtime._process_started_at", return_value=123.5
            ):
                result = await runtime.load_runner("owner-1", "mmh3_fi2v", {}, 0)
            runner = state.runners[result["runner_uuid"]]
            self.assertEqual(result["status"], "loading")
            self.assertTrue(context.process.started)
            self.assertEqual(runner["worker_pid"], 4242)
            self.assertEqual(runner["worker_started_at"], 123.5)
            await runtime._terminate_runner("TEST_COMPLETE")

    async def test_fifo_dispatch_completes_jobs_in_submission_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            runtime, state = self.make_runtime(root)
            commands = self.attach_ready_runner(runtime, state)
            runner = runtime.runner
            first_output = runtime.artifacts.output_path("owner-1", "job-1")
            second_output = runtime.artifacts.output_path("owner-1", "job-2")
            runner.queued = deque([
                LocalJob("job-1", "owner-1", {"output_path": str(first_output)}),
                LocalJob("job-2", "owner-1", {"output_path": str(second_output)}),
            ])
            state.queues["runner-1"] = ["job-1", "job-2"]
            for job_id in ("job-1", "job-2"):
                state.jobs[job_id] = {
                    "owner_id": "owner-1",
                    "runner_uuid": "runner-1",
                    "state": "queued",
                }

            await runtime._dispatch_next_job()
            self.assertEqual(commands.messages[0]["job_id"], "job-1")
            first_output.write_bytes(b"video")
            await runtime._complete_active_job()
            await runtime._dispatch_next_job()
            self.assertEqual(commands.messages[1]["job_id"], "job-2")
            second_output.write_bytes(b"video")
            await runtime._complete_active_job()

            self.assertEqual(state.jobs["job-1"]["state"], "completed")
            self.assertEqual(state.jobs["job-2"]["state"], "completed")
            self.assertEqual(list(state.expiries), ["job-1", "job-2"])

    async def test_fifo_uses_submission_order_when_an_earlier_upload_is_slower(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, state = self.make_runtime(Path(temporary) / "artifacts")
            commands = self.attach_ready_runner(runtime, state)
            slow_upload = DelayedUpload("first.png", "image/png", b"slow")
            first = asyncio.create_task(runtime.submit_inference(
                "owner-1",
                "runner-1",
                {"prompt": "first"},
                {"first_frame": [slow_upload]},
            ))
            await slow_upload.started.wait()

            second = await runtime.submit_inference(
                "owner-1",
                "runner-1",
                {"prompt": "second"},
                {"first_frame": [FakeUpload("second.png", "image/png", b"fast")]},
            )
            await runtime._dispatch_next_job()
            self.assertEqual(commands.messages, [])

            slow_upload.release.set()
            first_result = await first
            self.assertEqual(
                state.queues["runner-1"],
                [first_result["job_uuid"], second["job_uuid"]],
            )
            await runtime._dispatch_next_job()
            self.assertEqual(commands.messages[0]["job_id"], first_result["job_uuid"])
            Path(commands.messages[0]["payload"]["output_path"]).write_bytes(b"video")
            await runtime._complete_active_job()
            await runtime._dispatch_next_job()
            self.assertEqual(commands.messages[1]["job_id"], second["job_uuid"])

    async def test_cancelled_upload_releases_its_fifo_position(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, state = self.make_runtime(Path(temporary) / "artifacts")
            commands = self.attach_ready_runner(runtime, state)
            slow_upload = DelayedUpload("first.png", "image/png", b"slow")
            first = asyncio.create_task(runtime.submit_inference(
                "owner-1",
                "runner-1",
                {"prompt": "first"},
                {"first_frame": [slow_upload]},
            ))
            await slow_upload.started.wait()
            second = await runtime.submit_inference(
                "owner-1",
                "runner-1",
                {"prompt": "second"},
                {"first_frame": [FakeUpload("second.png", "image/png", b"fast")]},
            )

            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertEqual(state.queues["runner-1"], [second["job_uuid"]])
            await runtime._dispatch_next_job()
            self.assertEqual(commands.messages[0]["job_id"], second["job_uuid"])

    async def test_queue_dispatch_failure_stops_before_sending_an_inference_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            runtime, state = self.make_runtime(root)
            commands = self.attach_ready_runner(runtime, state)
            runtime.runner.process = TerminableProcess()
            output = runtime.artifacts.output_path("owner-1", "job-1")
            runtime.runner.queued.append(LocalJob("job-1", "owner-1", {"output_path": str(output)}))
            state.queues["runner-1"] = ["job-1"]
            state.jobs["job-1"] = {
                "owner_id": "owner-1",
                "runner_uuid": "runner-1",
                "state": "queued",
            }
            state.active_runner = "runner-1"
            state.fail_dequeue = True
            await runtime._dispatch_next_job()
            self.assertEqual(commands.messages, [])
            self.assertIsNone(runtime.runner)
            self.assertIsNone(state.active_runner)
            self.assertEqual(state.jobs["job-1"]["status_code"], "QUEUE_UNAVAILABLE")

    async def test_pending_upload_prevents_unload_until_it_is_queued(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, state = self.make_runtime(Path(temporary) / "artifacts")
            self.attach_ready_runner(runtime, state)
            upload = DelayedUpload("frame.png", "image/png", b"frame")
            submission = asyncio.create_task(runtime.submit_inference(
                "owner-1",
                "runner-1",
                {"prompt": "private prompt"},
                {"first_frame": [upload]},
            ))
            await upload.started.wait()
            with self.assertRaises(PlatformError) as raised:
                await runtime.unload_runner("owner-1", "runner-1")
            self.assertEqual(raised.exception.code, "RUNNER_BUSY")
            upload.release.set()
            result = await submission
            self.assertEqual(result["status"], "queued")

    async def test_unload_fails_closed_when_its_state_transition_cannot_be_saved(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, state = self.make_runtime(Path(temporary) / "artifacts")
            self.attach_ready_runner(runtime, state)
            process = TerminableProcess()
            runtime.runner.process = process
            state.active_runner = "runner-1"
            state.fail_update_runner_once = True

            with self.assertRaises(PlatformError) as raised:
                await runtime.unload_runner("owner-1", "runner-1")
            self.assertEqual(raised.exception.code, "RUNTIME_UNAVAILABLE")
            self.assertEqual(runtime.runner.state, "failed")
            self.assertFalse(process.is_alive())

            await runtime._check_worker_process()
            self.assertIsNone(runtime.runner)
            self.assertIsNone(state.active_runner)

    async def test_hard_stop_cancels_an_inflight_upload_and_removes_its_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            runtime, state = self.make_runtime(root)
            self.attach_ready_runner(runtime, state)
            runtime.runner.process = TerminableProcess()
            upload = DelayedUpload("frame.png", "image/png", b"frame")
            submission = asyncio.create_task(runtime.submit_inference(
                "owner-1",
                "runner-1",
                {"prompt": "private prompt"},
                {"first_frame": [upload]},
            ))
            await upload.started.wait()
            await runtime._terminate_runner("OFFLOAD_RAM_HARD_STOP")
            upload.release.set()
            with self.assertRaises(PlatformError) as raised:
                await submission
            self.assertEqual(raised.exception.code, "RUNNER_NOT_READY")
            self.assertFalse((root / "owner-1").exists())

    async def test_failed_runner_cleanup_retries_after_a_transient_state_store_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, state = self.make_runtime(Path(temporary) / "artifacts")
            self.attach_ready_runner(runtime, state)
            runtime.runner.process = TerminableProcess()
            runtime.runner.active = LocalJob("job-1", "owner-1", {})
            state.jobs["job-1"] = {
                "owner_id": "owner-1",
                "runner_uuid": "runner-1",
                "state": "running",
            }
            state.active_runner = "runner-1"
            state.fail_update_job_once = True
            with self.assertRaises(ConnectionError):
                await runtime._terminate_runner("HOST_RAM_HARD_STOP")
            self.assertEqual(runtime.runner.state, "failed")

            await runtime._check_worker_process()
            self.assertIsNone(runtime.runner)
            self.assertIsNone(state.active_runner)
            self.assertEqual(state.jobs["job-1"]["status_code"], "HOST_RAM_HARD_STOP")

    async def test_completed_worker_event_retries_after_a_transient_state_store_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            runtime, state = self.make_runtime(root)
            self.attach_ready_runner(runtime, state)
            output = runtime.artifacts.output_path("owner-1", "job-1")
            output.write_bytes(b"video")
            runtime.runner.active = LocalJob("job-1", "owner-1", {"output_path": str(output)})
            runtime.runner.state = "busy"
            runtime.runner.events.put({"type": "job_completed", "job_id": "job-1"})
            state.jobs["job-1"] = {
                "owner_id": "owner-1",
                "runner_uuid": "runner-1",
                "state": "running",
            }
            state.fail_update_job_once = True

            with self.assertRaises(ConnectionError):
                await runtime._drain_worker_events()
            self.assertIsNotNone(runtime.runner.pending_event)
            self.assertEqual(runtime.runner.active.id, "job-1")

            await runtime._drain_worker_events()
            self.assertIsNone(runtime.runner.pending_event)
            self.assertIsNone(runtime.runner.active)
            self.assertEqual(runtime.runner.state, "ready")
            self.assertEqual(state.jobs["job-1"]["state"], "completed")

    async def test_completed_worker_event_retries_if_the_runner_ready_state_cannot_be_saved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            runtime, state = self.make_runtime(root)
            self.attach_ready_runner(runtime, state)
            output = runtime.artifacts.output_path("owner-1", "job-1")
            output.write_bytes(b"video")
            runtime.runner.active = LocalJob("job-1", "owner-1", {"output_path": str(output)})
            runtime.runner.state = "busy"
            runtime.runner.events.put({"type": "job_completed", "job_id": "job-1"})
            state.jobs["job-1"] = {
                "owner_id": "owner-1",
                "runner_uuid": "runner-1",
                "state": "running",
            }
            state.fail_update_runner_once = True

            with self.assertRaises(ConnectionError):
                await runtime._drain_worker_events()
            self.assertIsNotNone(runtime.runner.pending_event)
            self.assertEqual(runtime.runner.active.id, "job-1")
            self.assertEqual(runtime.runner.state, "busy")

            await runtime._drain_worker_events()
            self.assertIsNone(runtime.runner.active)
            self.assertEqual(runtime.runner.state, "ready")
            self.assertEqual(state.jobs["job-1"]["state"], "completed")

    async def test_stopped_worker_without_a_final_event_releases_the_global_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, state = self.make_runtime(Path(temporary) / "artifacts")
            self.attach_ready_runner(runtime, state)
            runtime.runner.state = "stopping"
            runtime.runner.process = DeadProcess()
            state.active_runner = "runner-1"
            await runtime._check_worker_process()
            self.assertIsNone(runtime.runner)
            self.assertIsNone(state.active_runner)
            self.assertEqual(state.runners["runner-1"]["state"], "stopped")

    async def test_stopped_event_reaps_the_child_before_releasing_its_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, state = self.make_runtime(Path(temporary) / "artifacts")
            self.attach_ready_runner(runtime, state)
            process = TerminableProcess()
            runtime.runner.process = process
            runtime.runner.state = "stopping"
            runtime.runner.events.put({"type": "stopped"})
            state.active_runner = "runner-1"
            await runtime._drain_worker_events()
            self.assertFalse(process.is_alive())
            self.assertIsNone(runtime.runner)
            self.assertIsNone(state.active_runner)

    async def test_restart_releases_only_a_confirmed_gone_worker_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, state = self.make_runtime(Path(temporary) / "artifacts")
            state.active_runner = "runner-1"
            state.queues["runner-1"] = ["job-1"]
            state.runners["runner-1"] = {
                "owner_id": "owner-1",
                "state": "ready",
                "worker_pid": "12345",
                "worker_started_at": "1.0",
            }
            with patch("flexy_runtime._recorded_worker_state", return_value="gone"):
                await runtime._recover_after_restart()
            self.assertIsNone(state.active_runner)
            self.assertEqual(state.runners["runner-1"]["status_code"], "SERVER_RESTARTED")
            self.assertEqual(state.queues["runner-1"], [])
            self.assertEqual(
                state.runner_expiries["runner-1"], runtime.settings.artifact_ttl_seconds)

    async def test_restart_keeps_the_slot_when_an_orphan_worker_cannot_be_confirmed(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, state = self.make_runtime(Path(temporary) / "artifacts")
            state.active_runner = "runner-1"
            state.runners["runner-1"] = {"owner_id": "owner-1", "state": "ready"}
            with patch("flexy_runtime._recorded_worker_state", return_value="unknown"):
                await runtime._recover_after_restart()
            self.assertEqual(state.active_runner, "runner-1")
            self.assertEqual(state.runners["runner-1"]["status_code"], "ORPHANED_RUNNER_UNCONFIRMED")

    async def test_offload_ram_pressure_holds_queued_work_without_treating_full_vram_as_a_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, state = self.make_runtime(Path(temporary) / "artifacts")
            commands = self.attach_ready_runner(runtime, state)
            output = runtime.artifacts.output_path("owner-1", "job-1")
            runtime.runner.queued.append(LocalJob("job-1", "owner-1", {"output_path": str(output)}))
            state.queues["runner-1"] = ["job-1"]
            state.jobs["job-1"] = {
                "owner_id": "owner-1",
                "runner_uuid": "runner-1",
                "state": "queued",
            }
            runtime.resources.last_snapshot.host.offload_state = ResourceState.CRITICAL
            await runtime._dispatch_next_job()
            self.assertEqual(commands.messages, [])
            self.assertEqual(runtime.runner.state, "ready")

            runtime.resources.last_snapshot.host.offload_state = ResourceState.NORMAL
            runtime.resources.last_snapshot.host.swap_state = ResourceState.CRITICAL
            await runtime._dispatch_next_job()
            self.assertEqual(commands.messages, [])

            runtime.resources.last_snapshot.host.swap_state = ResourceState.NORMAL
            await runtime._dispatch_next_job()
            self.assertEqual(commands.messages[0]["job_id"], "job-1")

    async def test_completed_downloads_are_short_lived_and_owner_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            runtime, state = self.make_runtime(root)
            output = runtime.artifacts.output_path("owner-1", "job-1")
            output.write_bytes(b"video")
            state.jobs["job-1"] = {
                "owner_id": "owner-1",
                "runner_uuid": "runner-1",
                "state": "completed",
                "expires_at": int(time.time()) + 300,
            }

            result = await runtime.job_status("owner-1", "runner-1", "job-1")
            self.assertLessEqual(result["download_expires_at"], int(time.time()) + 30)
            path = await runtime.download_path(
                "owner-1", "job-1", result["download_expires_at"], result["download_signature"])
            self.assertEqual(path, output)
            with self.assertRaises(PlatformError) as wrong_owner:
                await runtime.download_path(
                    "owner-2", "job-1", result["download_expires_at"], result["download_signature"])
            self.assertEqual(wrong_owner.exception.code, "RESOURCE_NOT_FOUND")
            with self.assertRaises(PlatformError):
                await runtime.download_path(
                    "owner-1", "job-1", result["download_expires_at"], "not-the-signature")

    async def test_expiry_cleanup_removes_artifact_and_minimal_job_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            runtime, state = self.make_runtime(root)
            output = runtime.artifacts.output_path("owner-1", "job-1")
            output.write_bytes(b"video")
            state.jobs["job-1"] = {
                "owner_id": "owner-1",
                "runner_uuid": "runner-1",
                "state": "completed",
            }
            state.expiries["job-1"] = 0
            await runtime._cleanup_expired_artifacts()
            self.assertFalse(output.exists())
            self.assertNotIn("job-1", state.jobs)
            self.assertNotIn("job-1", state.expiries)

    async def test_expiry_cleanup_retries_when_windows_cannot_remove_a_locked_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, state = self.make_runtime(Path(temporary) / "artifacts")
            state.jobs["job-1"] = {"owner_id": "owner-1", "state": "completed"}
            state.expiries["job-1"] = 0
            with patch.object(runtime.artifacts, "remove_job", return_value=False):
                await runtime._cleanup_expired_artifacts()
            self.assertIn("job-1", state.jobs)
            self.assertIn("job-1", state.expiries)

    async def test_failed_artifact_retries_cleanup_without_waiting_for_the_retention_ttl(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, state = self.make_runtime(Path(temporary) / "artifacts")
            self.attach_ready_runner(runtime, state)
            runtime.runner.active = LocalJob("job-1", "owner-1", {})
            runtime.runner.state = "busy"
            state.jobs["job-1"] = {
                "owner_id": "owner-1",
                "runner_uuid": "runner-1",
                "state": "running",
            }
            with patch.object(runtime.artifacts, "remove_job", return_value=False), patch(
                "flexy_runtime._now", return_value=1000
            ):
                await runtime._fail_active_job("INFERENCE_FAILED")
            self.assertEqual(state.expiries["job-1"], 1000)

            with patch("flexy_runtime._now", return_value=1000):
                await runtime._cleanup_expired_artifacts()
            self.assertNotIn("job-1", state.jobs)
            self.assertNotIn("job-1", state.expiries)


if __name__ == "__main__":
    unittest.main()
