# Audio Streaming Examples for Whisper WebSocket Server

This directory contains examples for streaming audio to the Whisper transcription server.

## Server Requirements

Your server expects:
- **WebSocket URL**: `ws://localhost:8000/ws`
- **Audio Format**: 16-bit PCM, 16kHz, Mono
- **Frame Size**: 20ms frames (320 samples = 640 bytes per frame)
- **Data Type**: Raw binary data (not JSON)

## Example 1: Python Client with websockets

```python
import asyncio
import json
import numpy as np
import websockets

SAMPLE_RATE = 16000
FRAME_MS = 20
SAMPLES_PER_FRAME = 320  # 16000 * 0.02
BYTES_PER_FRAME = 640    # 320 * 2 bytes

async def stream_audio():
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as websocket:
        # Generate test audio (sine wave)
        duration = 3.0  # 3 seconds
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        audio_int16 = (audio * 32767 * 0.3).astype(np.int16)
        
        # Send in 20ms frames
        for i in range(0, len(audio_int16), SAMPLES_PER_FRAME):
            frame = audio_int16[i:i + SAMPLES_PER_FRAME]
            
            # Pad if necessary
            if len(frame) < SAMPLES_PER_FRAME:
                frame = np.pad(frame, (0, SAMPLES_PER_FRAME - len(frame)))
            
            # Send binary data
            await websocket.send(frame.tobytes())
            
            # Listen for responses
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                data = json.loads(response)
                print(f"{data['type']}: {data['text']}")
            except asyncio.TimeoutError:
                pass
            
            await asyncio.sleep(0.02)  # 20ms delay

# Run the client
asyncio.run(stream_audio())
```

## Example 2: JavaScript/Node.js Client

```javascript
const WebSocket = require('ws');

const SAMPLE_RATE = 16000;
const SAMPLES_PER_FRAME = 320;
const BYTES_PER_FRAME = 640;

const ws = new WebSocket('ws://localhost:8000/ws');

ws.on('open', () => {
    console.log('Connected to WebSocket');
    
    // Generate test audio
    const duration = 3.0;
    const totalSamples = Math.floor(SAMPLE_RATE * duration);
    const audioBuffer = new Int16Array(totalSamples);
    
    for (let i = 0; i < totalSamples; i++) {
        const t = i / SAMPLE_RATE;
        audioBuffer[i] = Math.floor(Math.sin(2 * Math.PI * 440 * t) * 32767 * 0.3);
    }
    
    // Send in frames
    let frameIndex = 0;
    const sendFrame = () => {
        if (frameIndex < totalSamples) {
            const start = frameIndex;
            const end = Math.min(start + SAMPLES_PER_FRAME, totalSamples);
            const frame = audioBuffer.slice(start, end);
            
            // Pad if necessary
            if (frame.length < SAMPLES_PER_FRAME) {
                const paddedFrame = new Int16Array(SAMPLES_PER_FRAME);
                paddedFrame.set(frame);
                ws.send(Buffer.from(paddedFrame.buffer));
            } else {
                ws.send(Buffer.from(frame.buffer));
            }
            
            frameIndex += SAMPLES_PER_FRAME;
            setTimeout(sendFrame, 20); // 20ms delay
        }
    };
    
    sendFrame();
});

ws.on('message', (data) => {
    const response = JSON.parse(data.toString());
    console.log(`${response.type}: ${response.text}`);
});
```

## Example 3: Streaming from Microphone (Python + pyaudio)

```python
import asyncio
import json
import pyaudio
import websockets

async def stream_microphone():
    # Audio configuration
    CHUNK = 320  # 20ms at 16kHz
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("🎤 Streaming from microphone... (Press Ctrl+C to stop)")
            
            # Start response listener
            async def listen_responses():
                while True:
                    try:
                        response = await websocket.recv()
                        data = json.loads(response)
                        print(f"[{data['type']}] {data['text']}")
                    except websockets.exceptions.ConnectionClosed:
                        break
            
            listener_task = asyncio.create_task(listen_responses())
            
            # Stream audio
            while True:
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                await websocket.send(audio_data)
                await asyncio.sleep(0.001)  # Small delay
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# Run microphone streaming
asyncio.run(stream_microphone())
```

## Example 4: Streaming from Audio File

```python
import asyncio
import json
import numpy as np
import soundfile as sf
import websockets

async def stream_audio_file(file_path: str):
    # Load audio file
    audio, sample_rate = sf.read(file_path)
    
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as websocket:
        print(f"🎵 Streaming audio file: {file_path}")
        
        # Listen for responses
        async def listen_responses():
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)
                    print(f"[{data['type']}] {data['text']}")
                except websockets.exceptions.ConnectionClosed:
                    break
        
        listener_task = asyncio.create_task(listen_responses())
        
        # Stream in 20ms chunks
        SAMPLES_PER_FRAME = 320
        for i in range(0, len(audio_int16), SAMPLES_PER_FRAME):
            frame = audio_int16[i:i + SAMPLES_PER_FRAME]
            
            # Pad if necessary
            if len(frame) < SAMPLES_PER_FRAME:
                frame = np.pad(frame, (0, SAMPLES_PER_FRAME - len(frame)))
            
            await websocket.send(frame.tobytes())
            await asyncio.sleep(0.02)  # 20ms real-time simulation
        
        # Wait for final transcription
        await asyncio.sleep(2)
        listener_task.cancel()

# Usage
# asyncio.run(stream_audio_file("your_audio_file.wav"))
```

## Required Dependencies

For Python examples:
```bash
pip install websockets numpy pyaudio soundfile scipy
```

For Node.js examples:
```bash
npm install ws
```

## Testing Your Setup

1. Start your server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

2. Test with curl (basic connection test):
   ```bash
   curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Key: test" -H "Sec-WebSocket-Version: 13" http://localhost:8000/ws
   ```

3. Use one of the Python examples above to stream audio

## Troubleshooting

- **Connection Refused**: Make sure server is running on port 8000
- **No Transcription**: Check that audio format matches requirements (16kHz, 16-bit, mono)
- **Poor Quality**: Adjust VAD settings or try different audio input
- **Latency Issues**: Tune the `EMIT_EVERY` parameter in your server code