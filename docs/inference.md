# Standalone workflow apps

The three app packages call local Python model code. They do not start ComfyUI,
submit workflow JSON, import ComfyUI/custom-node packages, or require the ComfyUI
installation at runtime. The native model implementations are bundled under
`vendor/Wan2GP`; Flexy's workflow-specific operations live under `apps/`.

**Development status:** the API/control-plane, queue, artifact, and resource-policy
tests have run, but the native apps have not executed on a GPU in this environment.
The original ComfyUI workflow's results do not verify this port. Output parity and
real memory use remain unverified.

These are Python app entry points behind a FastAPI runner manager. The server starts
one native child process at a time, and that process loads one app and executes one
FIFO job at a time. MMGP and the native backend use process-global state, so this
is intentional isolation rather than hidden concurrency. Each current app uses one
selected CUDA device with CPU offload. Multi-GPU splitting, parallel inference, and
vLLM are not implemented for these video/audio apps.

## Setup

The isolated child worker inherits the FastAPI process's Python environment. Run
the native server from one Python 3.12 CUDA environment containing both the API
and inference dependencies. For example, the pinned backend's Torch 2.10/CUDA 13
option is:

```powershell
py -3.12 -m venv .inference-venv
.inference-venv/Scripts/python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
.inference-venv/Scripts/python -m pip install -r requirements.txt
.inference-venv/Scripts/python -m pip install -r requirements-inference.txt
```

This option requires a driver supporting CUDA 13. `comfy-kitchen` is a standalone
numerical-kernel dependency; it does not require ComfyUI. The apps use PyTorch SDPA
attention and MMGP profile 4 to offload models to CPU. CPU RAM must also accommodate
the loaded models. No checkpoints are downloaded or replaced by `load()`.

`MODEL_PATH` selects the server-owned checkpoint directory and defaults to the
existing `D:/ComfyModels/models` directory. The HTTP API never accepts a model path.
For direct, trusted Python use only, `load({"model_root": ...})` remains available.
Each loader checks the exact filenames, including nested category directories, and lists missing files.
The Qwen configuration/tokenizer and existing T5 tokenizer are bundled independently
of ComfyUI under `vendor/Wan2GP/assets`.

## HTTP runner behavior

All runner routes require a JWT. `POST /v1/api/load_flexy_app` accepts the static
allowlist (`mmh3_fi2v`, `mmh3_fr2v`, `wan_p_iv2v`), the limited `turbo`/`steps`
load configuration, and `{ "gpu": { "device": 0 } }`. It returns a UUID in
`loading` state without waiting for model loading. The only supported GPU mode is
one physical GPU per isolated runner; CUDA visibility is set in the child before
the native app imports the model stack.

`POST /v1/api/inference` uses `multipart/form-data`: `runner_uuid`, a JSON-string
`payload`, and the app's named file parts. I2V requires `first_frame`; R2V accepts
one or more `reference_images` and/or `reference_video`/`reference_audio`; WAN
requires `reference_image` and `driving_video`. Public payloads cannot include
model paths, input paths, or output paths. The server writes generated filenames
under `ARTIFACT_ROOT/<owner>/<job>/`, retains inference payloads only in the
controller process, and queues job UUIDs in Redis FIFO order. Media requests must
include `Content-Length`; chunked uploads are rejected before multipart parsing.

The optional `logging` flag is accepted for compatibility, but the current native
video/audio service treats every request as incognito: it does not persist prompts,
request bodies, original filenames, media paths, or generation history.

Use `GET /v1/api/inference_status?runner_uuid=...&job_uuid=...` to poll. Native
apps do not expose dependable progress callbacks, so Flexy returns honest state
instead of inventing a percentage. A completed job has a short-lived signed URL;
the URL still requires the owner's JWT. Failed jobs delete their whole artifact
directory immediately. Successful `output.mp4` files expire through Redis-backed
cleanup after `AUTO_DELETE_MEDIA`, including recovery after a controller restart.
Terminal job metadata expires with the same temporary window rather than becoming
long-lived generation history.

`GET /v1/api/get_flexy_status?runner_uuid=...` returns the owner's runner state
and safe host/selected-GPU RAM, swap, VRAM, and temperature snapshot. The superuser-only
`GET /v1/api/admin/resources` returns all safe host/GPU telemetry. Resource policy
blocks new work under critical RAM, swap, or offload pressure; it terminates the
sole worker only for host/offload RAM hard stops, a severe swap hard stop, or a
sustained selected-GPU thermal hard stop.
Full VRAM alone does not terminate an offloading video app when host RAM is healthy.

## Python usage

Run from the project root, or place the project root on Python's import path.
Paths below are local files supplied by a trusted caller. They are not the public
HTTP contract: the HTTP runner resolves uploaded files and output directories
against the authenticated job owner and never exposes arbitrary server paths.

```python
from apps import mmh3_fi2v

app = mmh3_fi2v.load({"model_root": "D:/ComfyModels/models"})
try:
    result = app.infer({
        "prompt": "The person turns toward the camera and smiles.",
        "first_frame": "D:/inputs/start.png",
        "output_path": "D:/outputs/i2v.mp4",
        "duration_seconds": 5,
        "seed": 1,
    })
    print(result)  # output_path, frame_count, fps
finally:
    app.unload()
```

For reference-to-video:

```python
from apps import mmh3_fr2v

app = mmh3_fr2v.load()
try:
    result = app.infer({
        "prompt": "The reference character walks through a garden.",
        "reference_images": ["D:/inputs/character.png"],
        "output_path": "D:/outputs/r2v.mp4",
        "duration_seconds": 10,
        "seed": 1,
    })
finally:
    app.unload()
```

R2V also accepts optional `reference_video` and `reference_audio` local filenames.
These branches were bypassed in the saved workflow; they are opt-in. Reference video
is resampled to 24 fps. I2V accepts an optional `last_frame` filename. Both apps
generate video and audio at 24 fps and use the workflow's `17k + 5` frame-count rule.
Actual output duration can therefore slightly exceed the requested duration.

Both accept `megapixels` (default 0.9). R2V uses `aspect` (default `[9, 16]`); I2V
uses the first frame's aspect ratio and nearest resize, as in the root graph.
Dimensions align to multiples of 32. Set `turbo` and optionally `steps` in the load
configuration. R2V defaults to full mode/20 steps; turbo uses four steps and its
turbo LoRA. I2V defaults to turbo/six steps, matching the root subgraph override
even though the LoRA filename says eight steps; full mode uses 20 steps. I2V also
loads Unlocked V2 and VBVR at 0.7 each. Sampling uses `res_multistep`, the workflow's
simple schedule and the native H3 video/audio flow shifts.

For character replacement:

```python
from apps import wan_p_iv2v

app = wan_p_iv2v.load()
try:
    result = app.infer({
        "prompt": "A person follows the motion in the driving video.",
        "reference_image": "D:/inputs/character.png",
        "driving_video": "D:/inputs/motion.mp4",
        "base_detection_prompt": "girl,",
        "extension_detection_prompt": "girl, handkerchief",
        "reference_detection_prompt": "human",
        "output_path": "D:/outputs/replacement.mp4",
        "base_output_path": "D:/outputs/replacement-base.mp4",  # optional
        "seed": 1,
    })
finally:
    app.unload()
```

WAN decodes the source frames and relabels playback to 15 fps, preserving the
workflow's behavior rather than temporal resampling. Original audio is retained
and clipped to the output duration. The output defaults to 9:16, 0.2 megapixels,
with dimensions rounded as the graph does. `megapixels` can override its size.

The graph's linked video-frame count overrides its saved 81-frame widget. Accordingly,
`segment_frames` defaults to the entire driving video's frame count. The base and
extension each generate `4k + 1` frames; the extension uses five history frames,
then drops those five from its output and matches colors to the base's final frame.
For example, a source with 81 frames produces 81 + 76 output frames. Its extension
has only five remaining driving frames and extrapolates the rest, matching the
saved graph's layout. Set `segment_frames` explicitly when using a longer driving
video with shorter chunks. This implementation performs the graph's two segments.

WAN defaults to turbo/six steps/CFG 1, with DPO strength 1 and LightX2V strength 0.8.
`load({"turbo": False})` uses 40 steps/CFG 5 without LightX2V. `steps` can override
the step count. The sampler is Euler/simple with shift 5. Pose strength is 1 for
the full denoising range. The negative prompt defaults to empty; `negative_prompt`
can override it.

## Custom-node processing included

| Workflow operation | Python implementation |
| --- | --- |
| Resolution selector, megapixel resize, frame arithmetic | `apps/_media.py` |
| MiniMax reference/keyframe conditioning, audio/video VAEs, ConvRot INT8 and NVFP4 text loading, LoRA transforms | `apps/_minimax.py` and bundled native MiniMax code |
| SAM3.1 text segmentation/tracking, threshold 0.5, four objects | `wan_p_iv2v/processing.py` and bundled SAM3 code |
| Colored object masks, stable identities, first appearance then left-to-right order | `wan_p_iv2v/processing.py` |
| SCAIL-2 reference matting, pose/mask latents and replacement conditioning | `wan_p_iv2v/__init__.py` and bundled SCAIL-2 code |
| Vision-only CLIP-H using the existing checkpoint, no crop | `VisionEncoder` in `wan_p_iv2v/processing.py` |
| Frame slicing/count, five-frame continuation, concatenation | `wan_p_iv2v/__init__.py` |
| Reinhard Lab color matching, per frame | `wan_p_iv2v/processing.py` |
| Video/audio decode and MP4 encode | `apps/_media.py` |
| Easy prompt text, switch and primitive nodes | Direct payload values and Python branches |

UI previews, note nodes and disconnected/bypassed caption-generation branches do
not run. No Gemini service is required. Prompts are not written into MP4 metadata;
text-embedding caches are disabled and SAM sessions are closed after processing.
Media is saved only to the caller's requested output paths.

## Exact WAN files still needed

These files were absent from the configured model folders during source inspection:

- `diffusion_models/wan2.1_14B_SCAIL_2_int8_convrot.safetensors`
- `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- `vae/Wan2_1_VAE_bf16.safetensors`
- `loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors`
- `loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors`

The existing `clip_vision/clip_vision_h.safetensors` and
`checkpoints/sam3.1_multiplex_fp16.safetensors` are used unchanged. MiniMax's named
checkpoints and LoRAs were present. Neither checkpoint conversion nor replacement
with official unquantized weights is performed.

## Dependency provenance

See [vendor/Wan2GP/FLEXY-NOTES.md](../vendor/Wan2GP/FLEXY-NOTES.md) for the pinned
source revision, the limited inference changes, tokenizer origins and licenses.
