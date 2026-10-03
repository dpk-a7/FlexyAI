import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from apps import mmh3_fi2v, mmh3_fr2v, wan_p_iv2v
from apps._media import h3_frame_count, read_video, resolution
from apps._runtime import checkpoint_paths


class FrameRange:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.shape = (3, stop - start, 1, 1)

    def __getitem__(self, index):
        channels, frames = index
        if channels != slice(None):
            raise AssertionError("Expected all channels")
        selected = range(self.start, self.stop)[frames]
        return FrameRange(selected.start, selected.stop)


class FakeContainer:
    def __init__(self, videos):
        self.streams = types.SimpleNamespace(video=videos)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class InferenceHelperTests(unittest.TestCase):
    def test_h3_frame_count_uses_the_workflow_frame_rule(self):
        for seconds, expected in ((5, 124), (10, 243)):
            frames = h3_frame_count(seconds)
            self.assertEqual(frames, expected)
            self.assertEqual(frames % 17, 5)
        for seconds in (0, -1):
            with self.assertRaises(ValueError):
                h3_frame_count(seconds)

    def test_resolution_is_positive_and_aligned(self):
        width, height = resolution(0.9, (9, 16))
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)
        with self.assertRaises(ValueError):
            resolution(0, (9, 16))
        for aspect in ((9,), (9, 16, 1), ("wide", 16)):
            with self.subTest(aspect=aspect):
                with self.assertRaises(ValueError):
                    resolution(0.9, aspect)

    def test_read_video_rejects_missing_video_or_frame_rate_before_decode(self):
        for videos, message in (([], "video stream"), ([types.SimpleNamespace(average_rate=None)], "frame rate")):
            container = FakeContainer(videos)
            with self.subTest(message=message), patch.dict("sys.modules", {
                "av": types.SimpleNamespace(open=lambda path: container),
            }):
                with self.assertRaisesRegex(ValueError, message):
                    read_video("input.mp4")

    def test_checkpoint_paths_accepts_one_nested_category_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "diffusion_models" / "nested" / "model.safetensors"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            paths = checkpoint_paths(root, {"model": "diffusion_models/model.safetensors"})
        self.assertEqual(paths, {"model": str(checkpoint)})

    def test_loaders_list_missing_workflow_checkpoints_before_gpu_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                (mmh3_fi2v.load, "minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
                (mmh3_fr2v.load, "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
                (wan_p_iv2v.load, "wan2.1_14B_SCAIL_2_int8_convrot.safetensors"),
            )
            for load, filename in cases:
                with self.subTest(filename=filename):
                    with self.assertRaisesRegex(FileNotFoundError, filename):
                        load({"model_root": root})

    def test_wan_extension_starts_from_the_base_segment_history(self):
        driving, generated_frames = wan_p_iv2v._driving_segments(FrameRange(0, 20), 10)
        base, extension = driving
        self.assertEqual(generated_frames, 9)
        self.assertEqual((base.start, base.stop), (0, 9))
        self.assertEqual((extension.start, extension.stop), (4, 13))
        self.assertEqual(extension.start, base.stop - 5)


if __name__ == "__main__":
    unittest.main()
