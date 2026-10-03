"""Standalone WAN 2.1 SCAIL-2 character replacement, including two-segment continuation."""

from copy import deepcopy
from pathlib import Path

from apps._media import image_tensor, read_image, read_video, resize_video, resolution, save_video
from apps._runtime import BACKEND_ROOT, MODEL_ROOT, checkpoint_paths, load_loras, prepare_backend, release_memory

CHECKPOINTS = {
    "transformer": "diffusion_models/wan2.1_14B_SCAIL_2_int8_convrot.safetensors",
    "text": "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "vae/Wan2_1_VAE_bf16.safetensors",
    "vision": "clip_vision/clip_vision_h.safetensors",
    "sam": "checkpoints/sam3.1_multiplex_fp16.safetensors",
    "dpo": "loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
}


def _driving_segments(source, requested_frames):
    generated_frames = (requested_frames - 1) // 4 * 4 + 1
    base = source[:, :generated_frames]
    extension = source[:, generated_frames - 5:generated_frames * 2 - 5]
    extension = extension[:, :(extension.shape[1] - 1) // 4 * 4 + 1]
    return (base, extension), generated_frames


class App:
    def __init__(self, config):
        self.turbo = config.get("turbo", True)
        self.steps = config.get("steps", 6 if self.turbo else 40)
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        root = Path(config.get("model_root", MODEL_ROOT)).resolve()
        filenames = dict(CHECKPOINTS)
        if self.turbo:
            filenames["turbo"] = "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
        self.paths = checkpoint_paths(root, filenames)
        prepare_backend(root)
        import torch
        from mmgp import offload
        from models.wan.any2video import WanAny2V
        from models.wan.configs import WAN_CONFIGS
        from shared.utils.text_encoder_cache import TextEncoderCache
        from .processing import VisionEncoder

        self.pipeline = self.offload = None
        try:
            config = deepcopy(WAN_CONFIGS["i2v-14B"])
            del config.clip_checkpoint
            config.sample_neg_prompt = ""  # The graph's negative CLIPTextEncode is empty.
            self.pipeline = WanAny2V(
                config, str(root), model_filename=[self.paths["transformer"]],
                model_type="scail2_14B", base_model_type="scail2_14B",
                model_def={
                    "scail2": True, "i2v_class": True,
                    "config_file": str(BACKEND_ROOT / "models/wan/configs/scail2_14B.json"),
                    "text_encoder_folder": str(BACKEND_ROOT / "assets/umt5-xxl"),
                    "VAE_URLs": self.paths["vae"], "flexy_simple_schedule": True,
                }, text_encoder_filename=self.paths["text"], VAE_dtype=torch.bfloat16,
            )
            self.pipeline._interrupt = False
            self.pipeline.text_encoder_cache = TextEncoderCache(max_size_mb=0)
            self.pipeline.clip = VisionEncoder(self.paths["vision"])
            adapters = [self.paths["dpo"]]
            strengths = [1.0]
            if self.turbo:
                adapters.append(self.paths["turbo"])
                strengths.append(0.8)
            load_loras(self.pipeline.model, "scail2_14B", adapters, strengths)
            self.offload = offload.profile({
                "transformer": self.pipeline.model,
                "text_encoder": self.pipeline.text_encoder.model,
                "text_encoder_2": self.pipeline.clip.model,
                "vae": self.pipeline.vae.model,
            }, profile_no=4, quantizeTransformer=False)
        except Exception:
            self.unload()
            raise

    def infer(self, payload):
        if self.pipeline is None:
            raise RuntimeError("App has been unloaded; load it before inference")
        import torch
        from .processing import match_colors, segment_inputs

        prompt = payload["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        source, _, audio, audio_rate = read_video(payload["driving_video"])
        reference = image_tensor(read_image(payload["reference_image"]))
        width, height = resolution(payload.get("megapixels", 0.2), (9, 16), multiple=8)
        width, height = width // 32 * 32, height // 32 * 32
        if min(width, height) < 32:
            raise ValueError("Requested resolution is too small for WAN")
        length = int(payload.get("segment_frames", source.shape[1]))
        if length < 9:
            raise ValueError("segment_frames must be at least 9 for the five-frame continuation")
        if length > source.shape[1]:
            raise ValueError("segment_frames cannot exceed the driving video's frame count")
        # ImageFromBatch selects the driving slices before WanSCAILToVideo encoding.
        driving, generated_length = _driving_segments(source, length)
        driving = [resize_video(frames, width, height) for frames in driving]
        ref = resize_video(reference[:, None], width, height)
        self.offload.unload_all()
        try:
            ref_mask, masks = segment_inputs(
                self.paths["sam"], ref, driving,
                [payload.get("base_detection_prompt", "girl,"),
                 payload.get("extension_detection_prompt", "girl, handkerchief")],
                payload.get("reference_detection_prompt", "human"),
            )
            # The reference VAE sees a black background in replacement mode.
            matted_ref = torch.where((ref_mask > -0.8).any(dim=0, keepdim=True), ref, -1.0)
            self.pipeline.clip.reference = reference
            segments = []
            for index in range(2):
                history = None
                if index:
                    history = segments[0][:, -5:].float() / 127.5 - 1
                with torch.inference_mode():
                    result = self.pipeline.generate(
                        input_prompt=prompt, n_prompt=payload.get("negative_prompt", ""),
                        input_frames=driving[index], input_masks=masks[index],
                        input_ref_images=[matted_ref, ref_mask],
                        input_video=history, prefix_frames_count=5 if index else 0,
                        width=width, height=height, frame_num=generated_length,
                        sampling_steps=self.steps, guide_scale=1 if self.turbo else 5,
                        shift=5, sample_solver="euler", seed=int(payload.get("seed", 1)),
                        model_type="scail2_14B", video_prompt_type="0",
                        offloadobj=self.offload, cfg_star_switch=False, cfg_zero_step=0,
                        color_correction_strength=0, save_masks=False, fps=15,
                        custom_settings={"flexy_frame_count": generated_length},
                    )
                if result is None:
                    raise RuntimeError("WAN generation was interrupted")
                segments.append(result["x"].cpu())
                self.offload.unload_all()
            extension = match_colors(segments[1][:, 5:], segments[0][:, -1])
            first = segments[0].float() / 127.5 - 1
            video = torch.cat([first, extension], dim=1)
            result = save_video(payload["output_path"], video, 15, audio, audio_rate)
            if payload.get("base_output_path"):
                result["base_output"] = save_video(payload["base_output_path"], segments[0], 15, audio, audio_rate)
            return result
        finally:
            self.pipeline.clip.reference = None
            self.offload.unload_all()

    def unload(self):
        if self.offload is not None:
            self.offload.release()
            self.offload = None
        self.pipeline = None
        release_memory()


def load(config=None):
    return App(config or {})
