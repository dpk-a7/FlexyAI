"""Python equivalents of the workflow's SAM3, colored-mask and color-transfer nodes."""

from apps._runtime import BACKEND_ROOT, release_memory

PALETTE = ((0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255))


def track_masks(predictor, frames, text, *, background):
    import numpy as np
    from PIL import Image
    import torch

    pixels = ((frames.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    images = [Image.fromarray(frame) for frame in pixels.permute(1, 2, 3, 0).numpy()]
    session = predictor.handle_request({
        "type": "start_session", "resource_path": images,
        "offload_video_to_cpu": True, "cache_frame_outputs": False,
    })["session_id"]
    outputs = {}
    try:
        first = predictor.handle_request({
            "type": "add_prompt", "session_id": session, "frame_index": 0,
            "text": text, "output_prob_thresh": 0.5,
        })
        outputs[0] = first["outputs"]
        if len(images) > 1:
            for result in predictor.handle_stream_request({
                "type": "propagate_in_video", "session_id": session,
                "propagation_direction": "forward", "start_frame_index": 0,
                "max_frame_num_to_track": len(images), "output_prob_thresh": 0.5,
            }):
                outputs[result["frame_index"]] = result["outputs"]
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session})

    # Preserve object identity across frames; assign colors by first appearance,
    # then left-to-right centroid, as SCAIL2ColoredMask does.
    tracks, first_seen = {}, {}
    h, w = frames.shape[-2:]
    for t, output in sorted(outputs.items()):
        masks = np.asarray(output["out_binary_masks"], dtype=bool).reshape(-1, h, w)
        for identity, mask in zip(output["out_obj_ids"], masks):
            identity = int(identity)
            if not mask.any():
                continue
            if identity not in tracks:
                tracks[identity] = {}
                first_seen[identity] = (t, float(np.nonzero(mask)[1].mean()))
            tracks[identity][t] = mask
    if not tracks:
        raise ValueError(f"SAM3 found no objects for the segmentation prompt {text!r}")
    selected = sorted(tracks, key=first_seen.get)[:4]
    colored = np.full((len(images), h, w, 3), background, dtype=np.uint8)
    for identity, color in zip(selected, PALETTE):
        for t, mask in tracks[identity].items():
            colored[t][mask] = color
    return torch.from_numpy(colored).permute(3, 0, 1, 2).float() / 127.5 - 1


def segment_inputs(checkpoint, reference, driving_segments, prompts, reference_prompt):
    from preprocessing.sam3.model_builder import build_sam3_predictor

    predictor = build_sam3_predictor(
        checkpoint_path=checkpoint,
        bpe_path=str(BACKEND_ROOT / "preprocessing/sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        version="sam3.1", max_num_objects=4, compile=False, warm_up=False,
        use_fa3=False, async_loading_frames=False,
        postprocess_batch_size=1, use_batched_grounding=True,
    )
    predictor.model.score_threshold_detection = 0.5
    predictor.model.new_det_thresh = 0.5
    try:
        reference_mask = track_masks(predictor, reference, reference_prompt, background=0)
        driving_masks = [track_masks(predictor, frames, prompt, background=255)
                         for frames, prompt in zip(driving_segments, prompts)]
        return reference_mask, driving_masks
    finally:
        predictor.shutdown()
        del predictor
        release_memory()


def match_colors(video, reference):
    """Reinhard Lab transfer, per output frame, using the previous segment's tail."""
    import cv2
    import numpy as np
    import torch

    ref_rgb = reference.detach().cpu().permute(1, 2, 0).float().numpy()
    if reference.dtype == torch.uint8:
        ref_rgb /= 255
    else:
        ref_rgb = (ref_rgb + 1) / 2
    ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB)
    mean = ref_lab.mean(axis=(0, 1))
    std = ref_lab.std(axis=(0, 1)).clip(1e-6)
    result = []
    for frame in video.permute(1, 2, 3, 0).detach().cpu():
        rgb = frame.float().numpy()
        rgb = rgb / 255 if frame.dtype == torch.uint8 else (rgb + 1) / 2
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        adjusted = (lab - lab.mean(axis=(0, 1))) * (std / lab.std(axis=(0, 1)).clip(1e-6)) + mean
        result.append(cv2.cvtColor(adjusted, cv2.COLOR_LAB2RGB).clip(0, 1))
    return torch.from_numpy(np.stack(result)).permute(3, 0, 1, 2) * 2 - 1


class VisionEncoder:
    """Load the existing vision-only CLIP-H checkpoint; no separate CLIP text model."""

    def __init__(self, checkpoint):
        import torch
        from accelerate import init_empty_weights
        from mmgp import offload
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

        config = CLIPVisionConfig(
            hidden_size=1280, intermediate_size=5120, num_hidden_layers=32,
            num_attention_heads=16, image_size=224, patch_size=14,
            projection_dim=1024, hidden_act="gelu",
        )
        with init_empty_weights():
            self.model = CLIPVisionModelWithProjection(config)
        offload.load_model_data(self.model, checkpoint, default_dtype=torch.float16,
                               writable_tensors=False, ignore_unused_weights=True)
        self.model.eval().requires_grad_(False)
        self.model.image_size = 224
        self.reference = None

    def visual(self, videos):
        import torch
        import torch.nn.functional as F

        # The graph feeds the original reference to CLIP, before black-background matting.
        pixels = self.reference.unsqueeze(0).to(device="cuda", dtype=torch.float32)
        pixels = F.interpolate(pixels, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        pixels = (pixels.clamp(-1, 1) + 1) / 2
        mean = pixels.new_tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
        std = pixels.new_tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            return self.model(pixel_values=(pixels - mean) / std, output_hidden_states=True).hidden_states[-2]
