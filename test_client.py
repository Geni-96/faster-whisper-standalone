#!/usr/bin/env python3
"""
WebSocket client to test the Whisper transcription server.
This demonstrates how to stream audio packets for live transcription.
"""

import asyncio
import json
import numpy as np
import websockets
import struct
import time
from typing import Optional

# Audio configuration matching the server
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples
BYTES_PER_FRAME = SAMPLES_PER_FRAME * SAMPLE_WIDTH   # 640 bytes

def generate_test_audio(duration_seconds: float = 5.0, frequency: float = 440.0) -> np.ndarray:
    """
    Generate a test sine wave audio signal.
    
    Args:
        duration_seconds: Length of audio to generate
        frequency: Frequency of the sine wave (440 Hz = A4 note)
    
    Returns:
        numpy array of int16 audio samples
    """
    total_samples = int(SAMPLE_RATE * duration_seconds)
    t = np.linspace(0, duration_seconds, total_samples, False)
    
    # Generate sine wave
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit integer
    audio_int16 = (audio * 32767 * 0.3).astype(np.int16)  # 30% volume
    
    return audio_int16

def audio_to_frames(audio: np.ndarray):
    """
    Convert audio array to frames of the correct size for the server.
    
    Args:
        audio: numpy array of int16 audio samples
        
    Yields:
        bytes: Audio frames of exactly BYTES_PER_FRAME size
    """
    for i in range(0, len(audio), SAMPLES_PER_FRAME):
        frame = audio[i:i + SAMPLES_PER_FRAME]
        
        # Pad with zeros if frame is too short
        if len(frame) < SAMPLES_PER_FRAME:
            frame = np.pad(frame, (0, SAMPLES_PER_FRAME - len(frame)), 'constant')
        
        # Convert to bytes (little-endian)
        frame_bytes = frame.tobytes()
        yield frame_bytes

async def test_websocket_connection(websocket_url: str = "ws://localhost:8000/ws"):
    """
    Test the WebSocket connection with generated audio.
    
    Args:
        websocket_url: URL of the WebSocket endpoint
    """
    try:
        print(f"Connecting to {websocket_url}...")
        
        async with websockets.connect(websocket_url) as websocket:
            print("✅ Connected successfully!")
            
            # Generate test audio (5 seconds of 440 Hz tone)
            print("🎵 Generating test audio (5 seconds, 440 Hz tone)...")
            test_audio = generate_test_audio(duration_seconds=5.0, frequency=440.0)
            
            # Start listening for responses
            async def listen_for_responses():
                try:
                    while True:
                        response = await websocket.recv()
                        data = json.loads(response)
                        print(f"📝 Received: {data['type']} - '{data['text']}'")
                except websockets.exceptions.ConnectionClosed:
                    print("🔌 Connection closed")
                except Exception as e:
                    print(f"❌ Error receiving: {e}")
            
            # Start the listener task
            listener_task = asyncio.create_task(listen_for_responses())
            
            # Stream audio frames
            print("🎤 Streaming audio frames...")
            frame_count = 0
            
            for frame_bytes in audio_to_frames(test_audio):
                try:
                    await websocket.send(frame_bytes)
                    frame_count += 1
                    
                    if frame_count % 50 == 0:  # Progress every 1 second
                        print(f"📡 Sent {frame_count} frames ({frame_count * FRAME_MS / 1000:.1f}s)")
                    
                    # Wait to simulate real-time streaming
                    await asyncio.sleep(FRAME_MS / 1000)  # 20ms per frame
                    
                except Exception as e:
                    print(f"❌ Error sending frame {frame_count}: {e}")
                    break
            
            print(f"✅ Finished streaming {frame_count} frames")
            
            # Wait a bit for final responses
            await asyncio.sleep(2)
            
            # Cancel the listener task
            listener_task.cancel()
            
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Connection error: {e}")

async def stream_from_microphone(websocket_url: str = "ws://localhost:8000/ws"):
    """
    Example function showing how to stream from a real microphone.
    Note: This requires additional dependencies like pyaudio or sounddevice.
    """
    print("🎤 Microphone streaming example (requires pyaudio):")
    print("""
    import pyaudio
    
    def stream_microphone():
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=SAMPLES_PER_FRAME
        )
        
        async with websockets.connect(websocket_url) as websocket:
            while True:
                # Read audio frame from microphone
                audio_data = stream.read(SAMPLES_PER_FRAME)
                
                # Send to WebSocket
                await websocket.send(audio_data)
                
                # Listen for responses
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                    data = json.loads(response)
                    print(f"Transcription: {data['text']}")
                except asyncio.TimeoutError:
                    pass  # No response yet, continue streaming
    """)

def stream_from_file_example():
    """Show how to stream from an audio file."""
    print("📁 File streaming example (requires librosa or soundfile):")
    print("""
    import soundfile as sf
    
    def stream_audio_file(file_path: str):
        # Load audio file
        audio, sr = sf.read(file_path)
        
        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Stream to WebSocket
        async with websockets.connect(websocket_url) as websocket:
            for frame_bytes in audio_to_frames(audio_int16):
                await websocket.send(frame_bytes)
                await asyncio.sleep(FRAME_MS / 1000)  # Real-time streaming
    """)

async def main():
    """Main function to run the test client."""
    print("🎯 Whisper WebSocket Test Client")
    print("=" * 40)
    
    # Test basic connection
    await test_websocket_connection()
    
    print("\n" + "=" * 40)
    print("📚 Additional Examples:")
    print("=" * 40)
    
    # Show microphone example
    await stream_from_microphone()
    
    # Show file example
    stream_from_file_example()

if __name__ == "__main__":
    print("🚀 Starting WebSocket client test...")
    print("Make sure your server is running with: uvicorn app:app --host 0.0.0.0 --port 8000")
    print()
    
    asyncio.run(main())