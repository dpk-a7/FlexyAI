"""Local checkpoint resolution and the pinned native inference dependency."""

import gc
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "vendor" / "Wan2GP"
MODEL_ROOT = Path(os.getenv("MODEL_PATH", "D:/ComfyModels/models"))
BACKEND_REVISION = "b8b18f8114e432eea8f3d7e853a51dd91fa99571"


def prepare_backend(model_root):
    if not (BACKEND_ROOT / "models/minimax_h3/pipeline.py").is_file():
        raise FileNotFoundError("Install the pinned Wan2GP source described in docs/inference.md")
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    import torch
    from mmgp import offload
    from shared.utils import files_locator

    if not torch.cuda.is_available():
        raise RuntimeError("These apps require CUDA-enabled PyTorch and a CUDA GPU")
    files_locator.set_checkpoints_paths([str(model_root), str(BACKEND_ROOT / "assets")])
    offload.shared_state["_attention"] = "sdpa"


def checkpoint_paths(model_root, filenames):
    """Resolve exact filenames; never replace a missing checkpoint with another model."""
    root = Path(model_root).resolve()
    paths = {}
    missing = []
    for name, relative in filenames.items():
        path = root / relative
        if not path.is_file():
            # Comfy installations commonly put models in nested category folders.
            matches = list((root / Path(relative).parts[0]).rglob(Path(relative).name))
            if len(matches) > 1:
                raise ValueError(f"Multiple copies of {relative}; use an unambiguous model root")
            if matches:
                path = matches[0]
        if not path.is_file():
            missing.append(str(path))
        paths[name] = str(path)
    if missing:
        raise FileNotFoundError("Missing exact workflow checkpoints:\n" + "\n".join(missing))
    return paths


def load_loras(model, model_type, paths, strengths):
    from mmgp import offload

    preprocessor = getattr(model, "preprocess_loras", None)
    offload.load_loras_into_model(
        model, paths, strengths, activate_all_loras=True, pinnedLora=False,
        preprocess_sd=(lambda sd: preprocessor(model_type, sd)) if preprocessor else None,
        split_linear_modules_map=getattr(model, "split_linear_modules_map", None),
    )
    if model._loras_errors:
        raise RuntimeError("LoRA loading failed: " + "; ".join(str(e) for e in model._loras_errors))


def release_memory():
    import torch
    gc.collect()
    torch.cuda.empty_cache()
