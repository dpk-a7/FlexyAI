"""MiniMax H3 first/last-image-to-video/audio with the workflow's three LoRAs."""

from apps._minimax import MiniMaxApp


def load(config=None):
    """Return a loaded app with infer(payload) and unload() methods."""
    return MiniMaxApp(reference_mode=False, config=config or {})
