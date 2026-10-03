"""Shared model lifecycle for the two native MiniMax H3 apps."""

from pathlib import Path

from ._media import h3_frame_count, read_image, read_video, resolution, save_video
from ._runtime import MODEL_ROOT, checkpoint_paths, load_loras, prepare_backend, release_memory


class MiniMaxApp:
    def __init__(self, *, reference_mode, config):
        self.reference_mode = reference_mode
        self.turbo = config.get("turbo", not reference_mode)
        self.steps = config.get("steps", (4 if reference_mode else 6) if self.turbo else 20)
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        root = Path(config.get("model_root", MODEL_ROOT)).resolve()
        variant = "ref2va" if reference_mode else "fl2va"
        files = {
            "transformer": f"diffusion_models/minimax_h3_{variant}_pruned_int8_convrot.safetensors",
            "text": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "video_vae": "vae/minimax_h3_video_vae_fp16.safetensors",
            "audio_vae": "vae/minimax_h3_audio_vae_fp32.safetensors",
        }
        adapters, strengths = [], []
        if self.turbo:
            files["turbo"] = "loras/" + (
                "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" if reference_mode
                else "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")
            adapters.append("turbo")
            strengths.append(1.0)
        # These two nodes are active in I2V and bypassed in R2V.
        if not reference_mode:
            files.update(unlocked="loras/Minimax_H3_Unlocked_V2.safetensors",
                         vbvr="loras/H3_VBVR_Pro_attn_only.safetensors")
            adapters.extend(["unlocked", "vbvr"])
            strengths.extend([0.7, 0.7])
        paths = checkpoint_paths(root, files)
        prepare_backend(root)
        import torch
        from mmgp import offload
        from models.minimax_h3.minimax_h3_main import (
            _load_audio_vae, _load_text_encoder, _load_transformer, _load_video_vae,
        )
        from models.minimax_h3.pipeline import MiniMaxH3Pipeline
        from shared.utils.text_encoder_cache import TextEncoderCache

        self.pipeline = None
        self.offload = None
        try:
            transformer = _load_transformer(paths["transformer"], torch.bfloat16)
            if adapters:
                load_loras(transformer, f"minimax_h3_{variant}_pruned",
                           [paths[key] for key in adapters], strengths)
            self.pipeline = MiniMaxH3Pipeline(
                transformer, _load_text_encoder(paths["text"], torch.bfloat16),
                _load_video_vae(paths["video_vae"]), _load_audio_vae(paths["audio_vae"]),
                reference_mode=reference_mode,
            )
            self.pipeline.text_encoder_cache = TextEncoderCache(max_size_mb=0)
            self.offload = offload.profile({
                "transformer": transformer,
                "vae": self.pipeline.video_decoder,
                "video_encoder": self.pipeline.video_encoder,
                "audio_vae": self.pipeline.audio_vae,
                "text_encoder": self.pipeline.text_encoder.language_model,
                "vision_encoder": self.pipeline.text_encoder.visual,
            }, profile_no=4, quantizeTransformer=False)
        except Exception:
            self.unload()
            raise

    def infer(self, payload):
        if self.pipeline is None:
            raise RuntimeError("App has been unloaded; load it before inference")
        prompt = payload["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        output_path = payload["output_path"]
        seconds = payload.get("duration_seconds", 10 if self.reference_mode else 5)
        kwargs = {}
        if self.reference_mode:
            images = payload.get("reference_images", [])
            if not images and not payload.get("reference_video") and not payload.get("reference_audio"):
                raise ValueError("Provide reference_images, reference_video, or reference_audio")
            width, height = resolution(payload.get("megapixels", 0.9), tuple(payload.get("aspect", (9, 16))))
            kwargs["input_ref_images"] = [read_image(path) for path in images]
            if payload.get("reference_video"):
                video, source_fps, _, _ = read_video(payload["reference_video"])
                import torch
                indices = (torch.arange(round(video.shape[1] * 24 / source_fps)) * source_fps / 24).long()
                kwargs.update(input_frames=video[:, indices.clamp(max=video.shape[1] - 1)], video_prompt_type="V-")
            if payload.get("reference_audio"):
                kwargs.update(audio_guide=str(payload["reference_audio"]), audio_prompt_type="A")
        else:
            from PIL import Image
            first = read_image(payload["first_frame"])
            width, height = resolution(payload.get("megapixels", 0.9), first.size)
            kwargs["image_start"] = first.resize((width, height), Image.Resampling.NEAREST)
            if payload.get("last_frame"):
                kwargs["image_end"] = read_image(payload["last_frame"]).resize((width, height), Image.Resampling.NEAREST)
        try:
            result = self.pipeline.generate(
                input_prompt=prompt, width=width, height=height,
                frame_num=h3_frame_count(seconds), fps=24, seed=int(payload.get("seed", 1)),
                sampling_steps=self.steps, sample_solver="res_multistep", shift=12,
                custom_settings={"flexy_simple_schedule": True}, **kwargs,
            )
            if result is None:
                raise RuntimeError("MiniMax generation was interrupted")
            return save_video(output_path, result["x"], 24, result.get("audio"), result.get("audio_sampling_rate"))
        finally:
            self.offload.unload_all()

    def unload(self):
        if self.offload is not None:
            self.offload.release()
            self.offload = None
        self.pipeline = None
        release_memory()
