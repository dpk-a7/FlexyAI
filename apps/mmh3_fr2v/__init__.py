"""MiniMax H3 reference-to-video/audio, using the existing INT8 checkpoint."""

from apps._minimax import MiniMaxApp


def load(config=None):
    """Return a loaded app with infer(payload) and unload() methods."""
    return MiniMaxApp(reference_mode=True, config=config or {})
