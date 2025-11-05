# app.py
import asyncio, json, struct, time, io, os
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
# Tune CPU threads for better throughput on macOS/CPU-only.
CPU_THREADS = max(1, min(6, (os.cpu_count() or 4)))
model = WhisperModel(
    "small.en",
    device="auto",
    compute_type="int8",  # fastest on CPU; try "int8_float16" on GPU
    cpu_threads=CPU_THREADS,
    num_workers=1,  # keep 1 to avoid competing decoders on CPU
)

# ----- App -----
app = FastAPI()

# 20 ms frame = 320 samples at 16k
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000

# VAD: 0 = very aggressive (more filtering), 3 = least aggressive
vad = webrtcvad.Vad(2)

# Rolling audio buffer ~ 6 seconds (smaller buffer reduces re-decode cost)
ROLLING_SECONDS = 6
ROLLING_SAMPLES = SAMPLE_RATE * ROLLING_SECONDS

# End-of-speech detection and interim cadence
EMIT_EVERY = 0.5  # seconds between interim updates
SILENCE_FOR_FINAL_MS = 400  # how long of silence to consider utterance ended
SILENCE_FRAMES_FOR_FINAL = max(1, SILENCE_FOR_FINAL_MS // FRAME_MS)
MIN_UTTERANCE_SECONDS = 1.2  # don't finalize super short blips

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

    # Per-utterance buffer and VAD state
    utter_q: Deque[int] = deque(maxlen=SAMPLE_RATE * 30)  # up to 30s per utterance
    in_speech = False
    silence_frames = 0

    # simple debouncing for interim emissions
    last_emit = 0.0

    # Only one inference at a time to avoid contention/backoff
    infer_lock = asyncio.Lock()

    async def run_transcribe(buf_i16: np.ndarray, fast: bool) -> str:
        audio_f32 = int16_to_float32(buf_i16)
        # Faster settings: greedy decode, no timestamps, no VAD filter (we already filter)
        kwargs = dict(
            language="en",
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False,
            temperature=0.0,
        )
        if fast:
            kwargs.update(dict(beam_size=1, best_of=1, no_speech_threshold=0.6, log_prob_threshold=-1.0))
        else:
            kwargs.update(dict(beam_size=1, best_of=1))

        segments, _ = await asyncio.to_thread(model.transcribe, audio_f32, **kwargs)
        return "".join(seg.text for seg in segments).strip()

    async def do_interim(buf: np.ndarray):
        async with infer_lock:
            try:
                interim_text = await run_transcribe(buf, fast=True)
                if interim_text:
                    await ws.send_text(json.dumps({"type": "interim", "text": interim_text}))
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "error": f"interim: {e}"}))

    async def do_final(buf: np.ndarray):
        async with infer_lock:
            try:
                final_text = await run_transcribe(buf, fast=False)
                if final_text:
                    await ws.send_text(json.dumps({"type": "final", "text": final_text}))
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "error": f"final: {e}"}))

    try:
        while True:
            packet = await ws.receive_bytes()
            # Expect packet length == SAMPLES_PER_FRAME * SAMPLE_WIDTH
            if len(packet) != SAMPLES_PER_FRAME * SAMPLE_WIDTH:
                # Ignore malformed frames
                continue

            # VAD state per frame
            is_voice = vad_keep(packet)

            chunk_i16 = bytes_to_int16(packet)
            audio_q.extend(chunk_i16.tolist())

            now = time.time()

            if is_voice:
                # Enter/continue speech
                in_speech = True
                silence_frames = 0
                utter_q.extend(chunk_i16.tolist())

                # Interim update (skip if an inference is already running)
                if now - last_emit >= EMIT_EVERY and len(utter_q) >= SAMPLE_RATE * 1.0 and not infer_lock.locked():
                    last_emit = now
                    win = min(len(utter_q), SAMPLE_RATE * 2)
                    buf = np.array(list(utter_q)[-win:], dtype=np.int16)
                    asyncio.create_task(do_interim(buf))
            else:
                if in_speech:
                    silence_frames += 1
                    # Finalize after enough silence
                    if silence_frames >= SILENCE_FRAMES_FOR_FINAL and len(utter_q) >= int(SAMPLE_RATE * MIN_UTTERANCE_SECONDS):
                        # Snapshot and reset utterance buffer
                        buf = np.array(list(utter_q), dtype=np.int16)
                        utter_q.clear()
                        in_speech = False
                        silence_frames = 0

                        # Run final transcription (will wait if an interim is running)
                        asyncio.create_task(do_final(buf))
                # else: still in silence; do nothing

    except WebSocketDisconnect:
        pass
    except Exception as e:
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_text(json.dumps({"type":"error","error":str(e)}))

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await stream_transcribe(ws)
