"""File media I/O and the resize/frame-count operations used by the workflows."""

from fractions import Fraction
import math
from pathlib import Path


def resolution(megapixels, aspect=(9, 16), multiple=32):
    try:
        aspect_width, aspect_height = aspect
        values = (megapixels, aspect_width, aspect_height)
        valid = all(math.isfinite(value) and value > 0 for value in values)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Resolution and aspect ratio must be positive")
    scale = math.sqrt(megapixels * 1024 * 1024 / (aspect_width * aspect_height))
    return tuple(max(multiple, round(side * scale / multiple) * multiple)
                 for side in (aspect_width, aspect_height))


def h3_frame_count(seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    frames = max(5, round(seconds * 24))
    return frames + (5 - frames % 17) % 17


def read_image(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def image_tensor(image):
    import numpy as np
    import torch
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1


def read_video(path):
    """Decode every frame; the WAN workflow relabels these at 15 fps."""
    import av
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"Video has no video stream: {path}")
        stream = container.streams.video[0]
        try:
            fps = float(stream.average_rate)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"Video has no valid frame rate: {path}") from None
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video has no valid frame rate: {path}")
        import numpy as np
        import torch
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    if not frames:
        raise ValueError(f"Video has no decodable frames: {path}")
    video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 127.5 - 1
    audio, rate = read_audio(path)
    return video, fps, audio, rate


def read_audio(path):
    import av
    import numpy as np
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None, None
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(f.to_ndarray() for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray() for f in resampler.resample(None))
    return (np.concatenate(chunks, axis=1).T, rate) if chunks else (None, None)


def resize_video(video, width, height, *, mode="bicubic", crop=True):
    import torch.nn.functional as F
    frames = video.permute(1, 0, 2, 3)
    h, w = frames.shape[-2:]
    if crop:
        if w / h > width / height:
            keep = max(1, round(h * width / height))
            frames = frames[..., (w - keep) // 2:(w - keep) // 2 + keep]
        else:
            keep = max(1, round(w * height / width))
            frames = frames[..., (h - keep) // 2:(h - keep) // 2 + keep, :]
    kwargs = {"align_corners": False} if mode in ("bicubic", "bilinear") else {}
    return F.interpolate(frames, size=(height, width), mode=mode, **kwargs).permute(1, 0, 2, 3).clamp(-1, 1)


def save_video(path, video, fps, audio=None, audio_rate=None):
    import av
    import numpy as np
    import torch
    path = Path(path).resolve()
    if path.suffix.lower() != ".mp4":
        raise ValueError("output_path must end in .mp4")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = video.detach().cpu()
    if frames.dtype != torch.uint8:
        frames = ((frames.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).contiguous().numpy()
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps))
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        audio_stream = None
        if audio is not None:
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 1:
                audio = np.repeat(audio[:, None], 2, axis=1)
            audio = audio[:round(len(frames) / fps * audio_rate)]
            audio_stream = container.add_stream("aac", rate=audio_rate)
            audio_stream.layout = "stereo"
        for pixels in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio_stream is not None:
            for start in range(0, len(audio), 1024):
                chunk = np.ascontiguousarray(audio[start:start + 1024].T)
                frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
                frame.sample_rate, frame.pts, frame.time_base = audio_rate, start, Fraction(1, audio_rate)
                for packet in audio_stream.encode(frame):
                    container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)
    return {"output_path": str(path), "frame_count": len(frames), "fps": fps}
