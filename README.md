# Whisper Standalone Real-time Transcription Server

This is a FastAPI-based real-time speech transcription server using OpenAI's Whisper model via faster-whisper.

## Features

- Real-time audio transcription via WebSocket
- Voice Activity Detection (VAD) to filter out silence
- Rolling audio buffer for continuous transcription
- Interim and final transcription results
- Uses faster-whisper for optimized performance

## Setup

### Prerequisites

- Python 3.11 or higher
- macOS (current setup)

### Installation

1. Activate the virtual environment:
```bash
source bin/activate
```

2. Install dependencies (if not already installed):
```bash
pip install -r requirements.txt
```

### Running the Server

```bash
# Activate virtual environment
source bin/activate

# Start the server
uvicorn app:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`

### WebSocket Endpoint

Connect to: `ws://localhost:8000/ws`

#### Audio Format Requirements

- **Sample Rate**: 16,000 Hz
- **Sample Width**: 16-bit (2 bytes)
- **Channels**: Mono (1 channel)
- **Frame Size**: 20ms (320 samples per frame)
- **Data Format**: Raw PCM16 bytes

#### Expected Input

Send raw PCM16 audio data as binary WebSocket messages. Each message should contain exactly 640 bytes (320 samples × 2 bytes per sample).

#### Response Message Types

The server now emits structured JSON events for finer-grained, low‑latency segmentation:

1. Partial (previously "interim")
```json
{
  "type": "partial",
  "text": "partial transcription so far",
  "tStart": 12.34,          // seconds since connection start (utterance start time)
  "tEnd": 13.02             // current partial end time (seconds)
}
```
2. Boundary (utterance end detected by VAD + silence grace)
```json
{
  "type": "boundary",
  "event": "utterance_end",
  "tEnd": 14.27              // end time of the utterance in seconds
}
```
3. Final (decoded complete utterance after boundary)
```json
{
  "type": "final",
  "text": "complete utterance text",
  "tStart": 12.34,
  "tEnd": 14.27,
  "speaker": "Speaker 1",  // placeholder label; diarization not yet implemented
  "model": "small.en"
}
```
4. Error
```json
{
  "type": "error",
  "error": "description"
}
```

Client handling recommendations:
- Use `partial` events to update an in‑progress line.
- On `boundary`, finalize the current line immediately in the UI (even before the `final` arrives).
- Replace that finalized line with the `final` text when it comes (may include corrections due to context/prompting).
- Accumulate `final` entries for downloadable transcript.

## Model Configuration

Runtime model and decoding parameters are now environment configurable:

| Variable | Purpose | Default |
|----------|---------|---------|
| `FWS_MODEL` | Whisper model name | `small.en` |
| `FWS_DEVICE` | Device (`auto`, `cpu`, `cuda`) | `auto` |
| `FWS_COMPUTE_TYPE` | Quantization/precision | `int8` |
| `FWS_FINAL_BEAM` | Beam size for final decodes | `3` |
| `FWS_INTERIM_BEAM` | Beam size for partial decodes | `1` |
| `FWS_VAD_MODE` | WebRTC VAD aggressiveness (0–3) | `1` |
| `FWS_PRE_ROLL_FRAMES` | Frames prepended at utterance start | `50` (≈1s) |
| `FWS_TAIL_SAMPLES` | Overlap tail samples (context carryover) | `8000` (0.5s) |
| `FWS_SILENCE_FINAL_MS` | Base silence threshold for finalization | `450` |
| `FWS_SILENCE_GRACE_MS` | Extra silence grace after threshold | `200` |
| `FWS_ALLOW_SHORT_SILENCE_FRAMES` | Allowed brief pauses within speech | `3` |
| `FWS_EMIT_EVERY` | Partial emission cadence (seconds) | `0.5` |
| `FWS_MAX_INTERIM_RTF` | Max RTF to still emit partials | `1.1` |
| `FWS_COND_PREV` | Condition finals on previous text | `0` (disabled) |
| `FWS_SPEAKER` | Speaker label placeholder | `Speaker 1` |

Example:
```bash
export FWS_MODEL=base.en
export FWS_VAD_MODE=2
export FWS_COND_PREV=1
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Performance & Segmentation

- VAD + silence grace drives quick segmentation and boundary events.
- Tail overlap (`FWS_TAIL_SAMPLES`) preserves trailing context for next utterance.
- Pre-roll (`FWS_PRE_ROLL_FRAMES`) ensures leading words aren’t lost.
- Adaptive partial suppression when real-time factor (RTF) exceeds `FWS_MAX_INTERIM_RTF` keeps accuracy.
- Optional previous-text conditioning (`FWS_COND_PREV=1`) can improve continuity at utterance boundaries.

## Dependencies

See `requirements.txt` for the complete list. Key dependencies:
- `fastapi`: Web framework
- `faster-whisper`: Optimized Whisper implementation
- `webrtcvad`: Voice Activity Detection
- `numpy`: Audio processing
- `uvicorn`: ASGI server

## Streaming Audio from Another Server

To stream audio packets from another server for live transcription:

### Quick Test
```bash
# 1. Start your Whisper server
./bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000

# 2. Test the connection
./bin/python simple_test.py
```

### Integration Examples

See `audio_forwarder.py` for complete examples of:
- Streaming from microphone
- Streaming from audio files  
- Forwarding from another audio server
- Handling different audio formats

### Key Points for Your Audio Server Integration:

1. **Connect to WebSocket**: `ws://localhost:8000/ws`
2. **Send binary audio frames**: raw PCM16, 640 bytes each (20ms at 16kHz mono)
3. **Receive JSON events**: `partial`, `boundary`, `final`, `error`
4. **Finalize logic**: On `boundary`, close current line in UI; replace with `final` on arrival.

### Python Example:
```python
import asyncio
import websockets

async def stream_to_whisper():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Send 640-byte audio frames
        await ws.send(audio_frame_bytes)
        
        # Listen for transcription
        response = await ws.recv()
        result = json.loads(response)
        print(f"Transcription: {result['text']}")
```

For complete examples, see `STREAMING_EXAMPLES.md`.

## Notes

- First run downloads the selected model.
- VAD + grace logic aims to minimize truncated words.
- Real-time factor (RTF) logging (in server logs) helps diagnose performance limits.
- Speaker diarization is not yet implemented; `speaker` is a placeholder.

## Testing & CI

Automated tests (helpers, audio forwarding utility, and WebSocket flow) live under `tests/` and are executed in CI via GitHub Actions (workflow: `.github/workflows/ci.yml`).

### Run Tests Locally

Activate your virtualenv, install deps, then run:

```bash
source bin/activate
pytest -q
```

The test suite sets `FWS_SKIP_MODEL_INIT=1` to use a lightweight fake model—this avoids downloading Whisper weights and keeps tests fast. To run against a real model, unset that variable:

```bash
unset FWS_SKIP_MODEL_INIT
pytest -k websocket_flow -vv
```

### CI Environment Variables

In CI we deliberately lower timing thresholds and skip model initialization:

| Var | Purpose |
|-----|---------|
| `FWS_SKIP_MODEL_INIT=1` | Replace Whisper with deterministic fake for speed |
| `FWS_SILENCE_FINAL_MS=60` | Fast utterance finalization (3×20ms frames) |
| `FWS_MIN_UTTER_SEC=0.3` | Allow short test utterances |

### Adding More Tests

You can extend coverage by adding:
- Edge cases for malformed audio sequence lengths
- Stress tests with rapid alternating speech/silence
- Timing/RTF assertions using the `metrics` payload (enable in fake model if needed)

### Troubleshooting Hanging Tests

If a WebSocket test appears to hang, ensure it either:
1. Streams >1s of audio (required for a partial emission), or
2. Sends adequate silence frames after speech to trigger boundary/final events.

Short audio (<1s) plus no silence will produce no server messages, so tests must not block waiting indefinitely.