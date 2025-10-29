#!/usr/bin/env python3
"""
Simple WebSocket test for the Whisper transcription server.
Tests basic connectivity and sends a test tone.
"""

import asyncio
import json
import struct
import time

# Try to import websockets, provide fallback instructions if not available
try:
    import websockets
except ImportError:
    print("❌ websockets library not found.")
    print("Install it with: pip install websockets")
    exit(1)

# Try to import numpy, provide fallback if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("⚠️ numpy not found. Using basic test without audio generation.")
    HAS_NUMPY = False

# Audio configuration
SAMPLE_RATE = 16000
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2             # 640 bytes (16-bit)

def generate_simple_tone():
    """Generate a simple test tone without numpy."""
    duration = 2.0  # 2 seconds
    frequency = 440.0  # A4 note
    total_samples = int(SAMPLE_RATE * duration)
    
    audio_data = []
    for i in range(total_samples):
        t = i / SAMPLE_RATE
        # Simple sine wave
        import math
        value = math.sin(2 * math.pi * frequency * t) * 0.3  # 30% volume
        # Convert to 16-bit integer
        sample = int(value * 32767)
        # Clamp to 16-bit range
        sample = max(-32768, min(32767, sample))
        audio_data.append(sample)
    
    return audio_data

def generate_numpy_tone():
    """Generate test tone using numpy (better quality)."""
    duration = 2.0
    frequency = 440.0
    total_samples = int(SAMPLE_RATE * duration)
    
    t = np.linspace(0, duration, total_samples, False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16.tolist()

def audio_to_frames(audio_samples):
    """Convert audio samples to frames for streaming."""
    frames = []
    for i in range(0, len(audio_samples), SAMPLES_PER_FRAME):
        frame_samples = audio_samples[i:i + SAMPLES_PER_FRAME]
        
        # Pad with zeros if frame is too short
        while len(frame_samples) < SAMPLES_PER_FRAME:
            frame_samples.append(0)
        
        # Convert to bytes (little-endian 16-bit)
        frame_bytes = b''
        for sample in frame_samples:
            frame_bytes += struct.pack('<h', sample)  # little-endian signed short
        
        frames.append(frame_bytes)
    
    return frames

async def test_websocket(websocket_url="ws://localhost:8000/ws"):
    """Test the WebSocket connection and stream audio."""
    
    print(f"🔗 Connecting to {websocket_url}...")
    
    try:
        async with websockets.connect(websocket_url) as websocket:
            print("✅ Connected successfully!")
            
            # Generate test audio
            print("🎵 Generating test audio...")
            if HAS_NUMPY:
                audio_samples = generate_numpy_tone()
            else:
                audio_samples = generate_simple_tone()
            
            # Convert to frames
            frames = audio_to_frames(audio_samples)
            print(f"📦 Generated {len(frames)} audio frames")
            
            # Start response listener
            responses_received = []
            
            async def listen_for_responses():
                try:
                    while True:
                        response = await websocket.recv()
                        data = json.loads(response)
                        print(f"📝 [{data['type']}] {data.get('text', data.get('error', ''))}")
                        responses_received.append(data)
                except websockets.exceptions.ConnectionClosed:
                    print("🔌 WebSocket connection closed")
                except Exception as e:
                    print(f"❌ Error receiving response: {e}")
            
            # Start the listener
            listener_task = asyncio.create_task(listen_for_responses())
            
            # Stream frames
            print("🎤 Streaming audio frames...")
            for i, frame in enumerate(frames):
                try:
                    await websocket.send(frame)
                    
                    # Progress indicator
                    if (i + 1) % 25 == 0:  # Every 0.5 seconds
                        elapsed = (i + 1) * FRAME_MS / 1000
                        print(f"📡 Streamed {i + 1}/{len(frames)} frames ({elapsed:.1f}s)")
                    
                    # Wait for real-time streaming
                    await asyncio.sleep(FRAME_MS / 1000)  # 20ms
                    
                except Exception as e:
                    print(f"❌ Error sending frame {i}: {e}")
                    break
            
            print("✅ Finished streaming audio")
            
            # Wait for final responses
            print("⏳ Waiting for final transcription...")
            await asyncio.sleep(3)
            
            # Clean up
            listener_task.cancel()
            
            # Summary
            print(f"\n📊 Summary:")
            print(f"   - Frames sent: {len(frames)}")
            print(f"   - Responses received: {len(responses_received)}")
            
            if responses_received:
                print(f"   - Final transcription: '{responses_received[-1].get('text', 'None')}'")
            else:
                print("   - No transcription received (this might be normal for test tones)")
            
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused!")
        print("   Make sure your server is running:")
        print("   uvicorn app:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ Connection error: {e}")

async def test_basic_connection(websocket_url="ws://localhost:8000/ws"):
    """Test basic WebSocket connection without sending audio."""
    print(f"🔗 Testing basic connection to {websocket_url}...")
    
    try:
        async with websockets.connect(websocket_url) as websocket:
            print("✅ Basic connection successful!")
            
            # Send a single empty frame to test the endpoint
            empty_frame = b'\x00' * BYTES_PER_FRAME
            await websocket.send(empty_frame)
            print("📡 Sent test frame")
            
            # Wait for any response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(response)
                print(f"📝 Received: {data}")
            except asyncio.TimeoutError:
                print("⏰ No response (timeout) - this is normal for silence")
            
            return True
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

async def main():
    """Main test function."""
    print("🎯 Whisper WebSocket Connection Test")
    print("=" * 50)
    
    # Test basic connection first
    if await test_basic_connection():
        print("\n" + "=" * 50)
        print("🎵 Testing with audio stream...")
        print("=" * 50)
        await test_websocket()
    else:
        print("\n❌ Basic connection failed. Please check your server.")
        print("\nTo start the server:")
        print("1. cd to your project directory")
        print("2. source bin/activate")
        print("3. uvicorn app:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    print("🚀 Starting WebSocket test...")
    print("📋 Requirements:")
    print("   - Server running on localhost:8000")
    print("   - WebSocket endpoint at /ws")
    print("   - Audio format: 16kHz, 16-bit, mono")
    print("")
    
    asyncio.run(main())