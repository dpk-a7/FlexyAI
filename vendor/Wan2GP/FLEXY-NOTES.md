# Flexy native inference snapshot

Source: https://github.com/deepbeepmeep/Wan2GP

Revision: `b8b18f8114e432eea8f3d7e853a51dd91fa99571`.

Credit: WanGP / DeepBeepMeep, https://github.com/deepbeepmeep and
https://twitter.com/deepbeepmeep, and the upstream model authors named in each file.

This is a source dependency, not a pip package. The snapshot contains the
`models/minimax_h3`, `models/wan`, `models/ideogram4`, `shared`, and `preprocessing`
trees and upstream root files. Flexy imports the native inference modules directly;
it does not launch `wgp.py`, its Gradio UI, or ComfyUI. The models require substantial
architecture and quantized-loading code, which is kept here with its attribution
instead of copying it into the small app packages.

Flexy modifications:

- Remove eager UI/family-handler imports from the MiniMax and WAN package initializers,
  and the unrelated preprocessing entry point from SAM3's initializer.
- Add opt-in `flexy_simple_schedule` to H3 and WAN Euler scheduling to select the
  workflow's discrete 1000-entry flow schedule.
- SCAIL-2 accepts `flexy_frame_count`, separating target output length from the
  shorter driving clip available during extension. Half-resolution pose and mask
  resizing uses area interpolation, as in the source workflow.

Bundled tokenizer assets:

- `assets/Qwen3-VL-32B-Instruct`: configuration and tokenizer metadata from
  https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct (retrieved 2026-10-03).
  These are tokenizer/configuration assets only; the app uses the existing
  `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` model weights.
- `assets/umt5-xxl`: the tokenizer files shipped in the user's ComfyUI
  `comfy/text_encoders/t5_tokenizer` directory, copied as standalone data assets.
  The tokenizer identifies `google/t5-v1_1-xxl`; no ComfyUI code is imported.
- SAM3's BPE data and H3 LoRA affine maps are included in the upstream source tree.

Retain `LICENSE.txt` and the copyright/license notices in individual source files.
WanGP's Community License permits private use but restricts embedding in paid
products or offering paid API/SaaS access without a separate commercial agreement.
Third-party code and model/checkpoint licenses remain applicable. This snapshot
does not grant additional rights to any upstream code or weights.

No inference or tests were run for this integration.
