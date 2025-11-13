import os
import sys
import numpy as np


def _ensure_import():
    here = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(here, os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import importlib
    return importlib.import_module("audio_forwarder")


def test_prepare_audio_frames_from_types():
    af = _ensure_import()
    streamer = af.WhisperStreamer("ws://example/ws")

    # bytes input (already int16 little-endian)
    arr = np.arange(640, dtype=np.int16)
    frames = streamer._prepare_audio_frames(arr.tobytes(), 16000)
    assert isinstance(frames, list)
    assert all(isinstance(b, (bytes, bytearray)) for b in frames)
    assert all(len(b) == 320*2 for b in frames)

    # list input
    frames2 = streamer._prepare_audio_frames(arr.tolist(), 16000)
    assert len(frames2) == len(frames)

    # ndarray int16 input
    frames3 = streamer._prepare_audio_frames(arr, 16000)
    assert len(frames3) == len(frames)


def test_prepare_audio_frames_resample():
    af = _ensure_import()
    streamer = af.WhisperStreamer("ws://example/ws")

    # Build a 8kHz 40ms tone (should resample to 16k)
    sr = 8000
    dur = 0.04
    t = np.linspace(0, dur, int(sr*dur), False)
    audio = (np.sin(2*np.pi*440*t) * 0.3 * 32767).astype(np.int16)
    frames = streamer._prepare_audio_frames(audio, sr)

    # Expect roughly dur/20ms frames
    assert len(frames) in (1, 2, 3)
    assert all(len(b) == 320*2 for b in frames)
