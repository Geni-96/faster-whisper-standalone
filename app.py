# app.py
import asyncio, json, struct, time, io
from collections import deque
from typing import Deque, Tuple

import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from starlette.websockets import WebSocketState

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1

# ----- Model -----
# Choose your model: "small.en", "base", "small", "medium", "large-v3", etc.
# device="cuda" with compute_type="int8_float16" (good balance) if GPU; else CPU defaults.
model = WhisperModel("small.en", device="auto", compute_type="int8")

# ----- App -----
app = FastAPI()

# 20 ms frame = 320 samples at 16k
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000

# VAD: 0 = very aggressive (more filtering), 3 = least aggressive
vad = webrtcvad.Vad(2)

# Rolling audio buffer ~ 8 seconds
ROLLING_SECONDS = 8
ROLLING_SAMPLES = SAMPLE_RATE * ROLLING_SECONDS

def bytes_to_int16(buf: bytes) -> np.ndarray:
    # little-endian int16
    return np.frombuffer(buf, dtype=np.int16)

def int16_to_float32(x: np.ndarray) -> np.ndarray:
    # Scale to [-1, 1]
    return (x.astype(np.float32) / 32768.0).clip(-1, 1)

def vad_keep(frame: bytes) -> bool:
    # webrtcvad expects 16bit mono PCM at 8/16/32/48k and specific frame lengths (10/20/30 ms)
    try:
        return vad.is_speech(frame, SAMPLE_RATE)
    except Exception:
        return True  # be permissive if anything goes off

async def stream_transcribe(ws: WebSocket):
    """
    Receives raw PCM16 frames (exactly SAMPLES_PER_FRAME samples per packet),
    maintains a rolling buffer, runs faster-whisper on voice-active windows,
    and sends {"type":"interim"/"final","text":"..."} messages.
    """
    await ws.accept()
    audio_q: Deque[np.ndarray] = deque(maxlen=ROLLING_SAMPLES)

    # simple debouncing for interim emissions
    last_emit = 0.0
    EMIT_EVERY = 0.7  # seconds
    last_final_text = ""

    try:
        while True:
            packet = await ws.receive_bytes()
            # Expect packet length == SAMPLES_PER_FRAME * SAMPLE_WIDTH
            if len(packet) != SAMPLES_PER_FRAME * SAMPLE_WIDTH:
                # Ignore malformed frames
                continue

            # VAD: drop silence early to save compute
            if not vad_keep(packet):
                continue

            chunk_i16 = bytes_to_int16(packet)
            audio_q.extend(chunk_i16.tolist())

            now = time.time()
            # Run interim decode periodically for low latency
            if now - last_emit >= EMIT_EVERY and len(audio_q) >= SAMPLE_RATE * 1.0:
                last_emit = now
                # Take last ~3.0s window for interim
                win = min(len(audio_q), SAMPLE_RATE * 3)
                buf = np.array(list(audio_q)[-win:], dtype=np.int16)

                # Convert to float32 range for faster-whisper
                audio_f32 = int16_to_float32(buf)

                # Quick decode: small beam, no temperature sampling for determinism
                segments, _ = model.transcribe(
                    audio_f32,
                    language="en",
                    vad_filter=True,
                    beam_size=3,
                    best_of=1,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    temperature=0.0,
                    word_timestamps=False,
                    log_prob_threshold=-1.0,  # be permissive for short windows
                )
                interim_text = "".join(seg.text for seg in segments).strip()
                if interim_text:
                    await ws.send_text(json.dumps({"type": "interim", "text": interim_text}))

            # Occasionally produce a "final" by decoding a longer window and
            # diffing against last_final_text to avoid repeats.
            if len(audio_q) >= SAMPLE_RATE * 4.0 and now - last_emit >= EMIT_EVERY:
                buf = np.array(list(audio_q), dtype=np.int16)
                audio_f32 = int16_to_float32(buf)

                segments, _ = model.transcribe(
                    audio_f32,
                    language="en",
                    vad_filter=True,
                    beam_size=5,
                    best_of=1,
                    condition_on_previous_text=False,
                    word_timestamps=False,
                    temperature=0.0,
                )
                full_text = "".join(seg.text for seg in segments).strip()
                final_text = full_text[len(last_final_text):].strip() if full_text.startswith(last_final_text) else full_text
                if final_text:
                    last_final_text = full_text
                    await ws.send_text(json.dumps({"type": "final", "text": final_text}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_text(json.dumps({"type":"error","error":str(e)}))

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await stream_transcribe(ws)
