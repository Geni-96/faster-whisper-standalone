"""
Example: Streaming audio from another server to Whisper WebSocket
This shows how to forward audio packets from your audio server to the transcription server.
"""

import asyncio
import json
import struct
import websockets
import numpy as np

class WhisperStreamer:
    def __init__(self, whisper_url="ws://localhost:8000/ws"):
        self.whisper_url = whisper_url
        self.websocket = None
        self.is_connected = False
        
    async def connect(self):
        """Connect to the Whisper WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.whisper_url)
            self.is_connected = True
            print(f"✅ Connected to Whisper server: {self.whisper_url}")
            
            # Start listening for transcription results
            asyncio.create_task(self._listen_for_transcriptions())
            
        except Exception as e:
            print(f"❌ Failed to connect to Whisper server: {e}")
            self.is_connected = False
    
    async def _listen_for_transcriptions(self):
        """Listen for transcription responses from the server."""
        try:
            while self.is_connected and self.websocket:
                response = await self.websocket.recv()
                data = json.loads(response)
                
                # Handle different types of responses
                if data['type'] == 'interim':
                    print(f"🔄 [INTERIM] {data['text']}")
                elif data['type'] == 'final':
                    print(f"✅ [FINAL] {data['text']}")
                elif data['type'] == 'error':
                    print(f"❌ [ERROR] {data.get('error', 'Unknown error')}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("🔌 Whisper connection closed")
            self.is_connected = False
        except Exception as e:
            print(f"❌ Error listening for transcriptions: {e}")
    
    async def stream_audio_chunk(self, audio_data, sample_rate=16000):
        """
        Stream a chunk of audio data to the Whisper server.
        
        Args:
            audio_data: Audio samples (list, numpy array, or bytes)
            sample_rate: Sample rate of the audio (default: 16000)
        """
        if not self.is_connected or not self.websocket:
            print("❌ Not connected to Whisper server")
            return
        
        try:
            # Convert audio to the required format
            frames = self._prepare_audio_frames(audio_data, sample_rate)
            
            # Send each frame
            for frame in frames:
                await self.websocket.send(frame)
                
        except Exception as e:
            print(f"❌ Error streaming audio: {e}")
    
    def _prepare_audio_frames(self, audio_data, sample_rate):
        """Prepare audio data for streaming to Whisper server."""
        
        # Convert different input types to numpy array
        if isinstance(audio_data, bytes):
            # Assume 16-bit little-endian PCM
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        elif isinstance(audio_data, list):
            audio_array = np.array(audio_data, dtype=np.int16)
        elif isinstance(audio_data, np.ndarray):
            if audio_data.dtype != np.int16:
                # Convert float to int16
                if audio_array.dtype in [np.float32, np.float64]:
                    audio_array = (audio_data * 32767).astype(np.int16)
                else:
                    audio_array = audio_data.astype(np.int16)
            else:
                audio_array = audio_data
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
        
        # Resample if necessary (simple example - you might want better resampling)
        if sample_rate != 16000:
            # Simple resampling - for production, use scipy.signal.resample
            ratio = 16000 / sample_rate
            new_length = int(len(audio_array) * ratio)
            audio_array = np.interp(
                np.linspace(0, len(audio_array), new_length),
                np.arange(len(audio_array)),
                audio_array
            ).astype(np.int16)
        
        # Split into 20ms frames (320 samples at 16kHz)
        SAMPLES_PER_FRAME = 320
        frames = []
        
        for i in range(0, len(audio_array), SAMPLES_PER_FRAME):
            frame_samples = audio_array[i:i + SAMPLES_PER_FRAME]
            
            # Pad with zeros if frame is too short
            if len(frame_samples) < SAMPLES_PER_FRAME:
                frame_samples = np.pad(frame_samples, 
                                     (0, SAMPLES_PER_FRAME - len(frame_samples)), 
                                     'constant')
            
            # Convert to bytes (little-endian)
            frame_bytes = frame_samples.tobytes()
            frames.append(frame_bytes)
        
        return frames
    
    async def disconnect(self):
        """Disconnect from the Whisper server."""
        self.is_connected = False
        if self.websocket:
            await self.websocket.close()

# Example usage functions:

async def example_stream_from_microphone():
    """Example: Stream from microphone to Whisper server."""
    import pyaudio
    
    # Audio configuration
    CHUNK = 320  # 20ms at 16kHz
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    # Initialize audio input
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    # Initialize Whisper streamer
    whisper = WhisperStreamer()
    await whisper.connect()
    
    print("🎤 Streaming from microphone... (Press Ctrl+C to stop)")
    
    try:
        while True:
            # Read audio from microphone
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Stream to Whisper server
            await whisper.stream_audio_chunk(audio_data)
            
            # Small delay to prevent overwhelming the server
            await asyncio.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        await whisper.disconnect()

async def example_stream_from_file():
    """Example: Stream from audio file to Whisper server."""
    import soundfile as sf
    
    # Load audio file
    file_path = "your_audio_file.wav"  # Replace with your file
    audio, sample_rate = sf.read(file_path)
    
    # Convert to int16 if needed
    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)
    
    # Initialize Whisper streamer
    whisper = WhisperStreamer()
    await whisper.connect()
    
    print(f"🎵 Streaming file: {file_path}")
    
    # Stream in chunks (simulate real-time)
    chunk_size = 320  # 20ms at 16kHz
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        await whisper.stream_audio_chunk(chunk, sample_rate)
        await asyncio.sleep(0.02)  # 20ms delay for real-time simulation
    
    print("✅ Finished streaming file")
    await whisper.disconnect()

async def example_forward_from_another_server():
    """
    Example: Forward audio from another server to Whisper.
    This simulates receiving audio packets from your audio server.
    """
    
    whisper = WhisperStreamer()
    await whisper.connect()
    
    # Simulate receiving audio packets from another server
    print("📡 Simulating audio server forwarding...")
    
    # This is where you'd replace with your actual audio server connection
    for i in range(100):  # Simulate 100 packets
        # Generate fake audio packet (replace with real server data)
        fake_audio = np.random.randint(-1000, 1000, 320, dtype=np.int16)
        
        # Forward to Whisper
        await whisper.stream_audio_chunk(fake_audio)
        
        # Simulate real-time streaming
        await asyncio.sleep(0.02)  # 20ms between packets
    
    await whisper.disconnect()

# For your specific use case - connecting to another audio server:
class AudioServerForwarder:
    """
    Forwards audio from your audio server to Whisper transcription server.
    """
    
    def __init__(self, audio_server_url, whisper_url="ws://localhost:8000/ws"):
        self.audio_server_url = audio_server_url
        self.whisper_streamer = WhisperStreamer(whisper_url)
        
    async def start_forwarding(self):
        """Start forwarding audio from audio server to Whisper."""
        
        # Connect to Whisper server
        await self.whisper_streamer.connect()
        
        # Connect to your audio server (replace with your actual connection logic)
        try:
            # Example with websockets (replace with your server's protocol)
            async with websockets.connect(self.audio_server_url) as audio_ws:
                print(f"📡 Connected to audio server: {self.audio_server_url}")
                
                while True:
                    # Receive audio data from your server
                    audio_packet = await audio_ws.recv()
                    
                    # Forward to Whisper (adjust based on your data format)
                    await self.whisper_streamer.stream_audio_chunk(audio_packet)
                    
        except Exception as e:
            print(f"❌ Error forwarding audio: {e}")
        finally:
            await self.whisper_streamer.disconnect()

# Usage examples:
if __name__ == "__main__":
    # Choose which example to run:
    
    # 1. Stream from microphone
    # asyncio.run(example_stream_from_microphone())
    
    # 2. Stream from file
    # asyncio.run(example_stream_from_file())
    
    # 3. Forward from another server
    # forwarder = AudioServerForwarder("ws://your-audio-server.com/audio")
    # asyncio.run(forwarder.start_forwarding())
    
    # 4. Simple test
    print("Run one of the example functions above!")