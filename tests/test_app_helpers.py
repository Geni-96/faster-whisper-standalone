import numpy as np
import os
import sys


def test_bytes_float_roundtrip():
    # Ensure project root is on path and import helpers
    here = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(here, os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Force lightweight fake model to avoid heavy download in CI
    os.environ.setdefault("FWS_SKIP_MODEL_INIT", "1")
    from app import bytes_to_int16, int16_to_float32  # type: ignore

    # Create a deterministic int16 buffer
    arr = (np.sin(np.linspace(0, 2*np.pi, 320)) * 10000).astype(np.int16)
    buf = arr.tobytes()

    # bytes -> int16
    recovered = bytes_to_int16(buf)
    assert recovered.dtype == np.int16
    assert recovered.shape == arr.shape
    assert np.array_equal(recovered, arr)

    # int16 -> float32 range
    f32 = int16_to_float32(arr)
    assert f32.dtype == np.float32
    assert np.all(f32 <= 1.0 + 1e-6)
    assert np.all(f32 >= -1.0 - 1e-6)
