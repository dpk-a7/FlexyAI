"""Child-process entry point for one isolated native Flexy runner."""

import importlib
import os
from queue import Empty
from typing import Any

from apps._registry import get_app


def run_worker(
    app_name: str,
    config: dict[str, Any],
    gpu_device: int,
    commands: Any,
    events: Any,
) -> None:
    """Load one app and execute commands serially without exposing model errors."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    app = None
    try:
        definition = get_app(app_name)
        module = importlib.import_module(definition.module)
        app = module.load(config)
        events.put({"type": "ready"})
        while True:
            command = commands.get()
            if command["type"] == "shutdown":
                break
            if command["type"] != "infer":
                continue
            job_id = command["job_id"]
            try:
                app.infer(command["payload"])
            except Exception:
                events.put({"type": "job_failed", "job_id": job_id, "code": "INFERENCE_FAILED"})
            else:
                events.put({"type": "job_completed", "job_id": job_id})
    except Exception:
        events.put({"type": "load_failed", "code": "RUNNER_LOAD_FAILED"})
    finally:
        if app is not None:
            try:
                app.unload()
            except Exception:
                pass
        events.put({"type": "stopped"})
