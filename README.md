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

#### Response Format

The server sends JSON messages with the following structure:

```json
{
  "type": "interim",  // or "final" or "error"
  "text": "transcribed text"
}
```

- `interim`: Partial transcription results for low latency
- `final`: Completed transcription segments
- `error`: Error messages

## Model Configuration

The current configuration uses:
- Model: `small.en` (English-only, ~244MB)
- Device: `auto` (will use GPU if available, otherwise CPU)
- Compute Type: `int8` for efficiency

You can modify these settings in `app.py`:

```python
model = WhisperModel("small.en", device="auto", compute_type="int8")
```

Available models: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v1`, `large-v2`, `large-v3`

## Performance Tuning

- **Rolling Buffer**: Currently set to 8 seconds (`ROLLING_SECONDS = 8`)
- **VAD Aggressiveness**: Set to 2 (0=most aggressive, 3=least aggressive)
- **Emission Frequency**: Interim results every 0.7 seconds (`EMIT_EVERY = 0.7`)

These can be adjusted in `app.py` based on your latency and accuracy requirements.

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
2. **Send binary audio data** (not JSON)
3. **Audio format**: 16kHz, 16-bit, mono, 20ms frames (640 bytes each)
4. **Receive JSON responses**:
   ```json
   {"type": "interim", "text": "partial transcription..."}
   {"type": "final", "text": "complete transcription"}
   ```

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

- The first run will download the Whisper model (may take a few minutes)
- VAD filtering helps reduce unnecessary processing of silence
- The system maintains a rolling audio buffer for context
- Transcription happens on voice-active segments only