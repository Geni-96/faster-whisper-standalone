# app.py
import asyncio, json, struct, time, io, os, logging
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
MODEL_NAME = os.getenv("FWS_MODEL", "small.en")
COMPUTE_TYPE = os.getenv("FWS_COMPUTE_TYPE", "int8")
model = WhisperModel(
    MODEL_NAME,
    device=os.getenv("FWS_DEVICE", "auto"),
    compute_type=COMPUTE_TYPE,  # fastest on CPU; try "int8_float16" on GPU
    cpu_threads=CPU_THREADS,
    num_workers=int(os.getenv("FWS_NUM_WORKERS", "1")),  # keep 1 to avoid competing decoders on CPU
)

# ----- App -----
app = FastAPI()

# 20 ms frame = 320 samples at 16k
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000

# VAD: 0 = very aggressive (more filtering), 3 = least aggressive
DEFAULT_VAD_MODE = int(os.getenv("FWS_VAD_MODE", "1"))  # 0=aggr,3=least; 1 is a good balance
vad = webrtcvad.Vad(DEFAULT_VAD_MODE)

# Rolling audio buffer ~ 6 seconds (smaller buffer reduces re-decode cost)
ROLLING_SECONDS = 6
ROLLING_SAMPLES = SAMPLE_RATE * ROLLING_SECONDS

# End-of-speech detection and interim cadence
EMIT_EVERY = float(os.getenv("FWS_EMIT_EVERY", "0.5"))  # seconds between interim updates
SILENCE_FOR_FINAL_MS = int(os.getenv("FWS_SILENCE_FINAL_MS", "450"))  # silence before final
SILENCE_FRAMES_FOR_FINAL = max(1, SILENCE_FOR_FINAL_MS // FRAME_MS)
MIN_UTTERANCE_SECONDS = float(os.getenv("FWS_MIN_UTTER_SEC", "1.0"))  # shorter allowed

# VAD hysteresis / pre-roll
PRE_ROLL_FRAMES = int(os.getenv("FWS_PRE_ROLL_FRAMES", "50"))  # default ~1s (50*20ms)
ALLOW_SHORT_SILENCE_FRAMES = int(os.getenv("FWS_ALLOW_SHORT_SILENCE_FRAMES", "3"))  # tolerate brief dips
# Silence finalization grace: extra silence required beyond threshold to reduce trailing word cuts
SILENCE_GRACE_MS = int(os.getenv("FWS_SILENCE_GRACE_MS", "200"))  # additional ms of silence after threshold
GRACE_FRAMES = max(0, SILENCE_GRACE_MS // FRAME_MS)

# Logging setup
logging.basicConfig(level=os.getenv("FWS_LOG_LEVEL", "INFO"))
log = logging.getLogger("fws")

# (Capture/log features removed per request)

# Energy-based fallback (to mitigate VAD misses)
ENERGY_ENABLE = os.getenv("FWS_ENERGY_ENABLE", "1") == "1"
ENERGY_FRAME_THRESHOLD = float(os.getenv("FWS_ENERGY_THRESH", "0.0008"))  # RMS threshold
ENERGY_MIN_VOICE_FRAMES = int(os.getenv("FWS_ENERGY_MIN_FRAMES", "4"))

# Beam sizes
FINAL_BEAM_SIZE = int(os.getenv("FWS_FINAL_BEAM", "3"))
INTERIM_BEAM_SIZE = int(os.getenv("FWS_INTERIM_BEAM", "1"))

# Previous text conditioning for finals
COND_PREV_FINAL = os.getenv("FWS_COND_PREV", "0") == "1"

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
    pre_roll: Deque[int] = deque(maxlen=PRE_ROLL_FRAMES * SAMPLES_PER_FRAME)
    in_speech = False
    silence_frames = 0
    dropped_frames = 0
    total_frames = 0
    last_latency_log = time.time()

    # simple debouncing for interim emissions
    last_emit = 0.0

    # Only one inference at a time to avoid contention/backoff
    infer_lock = asyncio.Lock()

    def _segments_to_text_and_metrics(segments) -> tuple[str, dict]:
        texts = []
        avg_logprobs = []
        comp_ratios = []
        no_speech_probs = []
        for seg in segments:
            texts.append(seg.text)
            # Attributes exist in faster-whisper Segment
            if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
                avg_logprobs.append(seg.avg_logprob)
            if hasattr(seg, "compression_ratio") and seg.compression_ratio is not None:
                comp_ratios.append(seg.compression_ratio)
            if hasattr(seg, "no_speech_prob") and seg.no_speech_prob is not None:
                no_speech_probs.append(seg.no_speech_prob)
        text = "".join(texts).strip()
        metrics = {
            "avg_logprob": (sum(avg_logprobs) / len(avg_logprobs)) if avg_logprobs else None,
            "max_comp_ratio": max(comp_ratios) if comp_ratios else None,
            "max_no_speech": max(no_speech_probs) if no_speech_probs else None,
        }
        return text, metrics

    def _looks_repetitive(text: str) -> bool:
        # Simple n-gram repetition detection
        words = text.lower().split()
        if len(words) < 6:
            return False
        n = 3
        counts = {}
        for i in range(len(words) - n + 1):
            key = tuple(words[i:i+n])
            counts[key] = counts.get(key, 0) + 1
            if counts[key] >= 5:
                return True
        return False

    async def run_transcribe(buf_i16: np.ndarray, fast: bool, prev_text: str = "") -> tuple[str, dict]:
        audio_f32 = int16_to_float32(buf_i16)
        t0 = time.time()
        # Faster settings: greedy decode, no timestamps, no VAD filter (we already filter)
        kwargs = dict(
            language="en",
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False,
            temperature=0.0,
        )
        if fast:
            kwargs.update(dict(beam_size=INTERIM_BEAM_SIZE, best_of=1, no_speech_threshold=0.8, log_prob_threshold=-0.8, compression_ratio_threshold=2.4))
        else:
            kwargs.update(dict(beam_size=FINAL_BEAM_SIZE, best_of=1, log_prob_threshold=-0.6, compression_ratio_threshold=2.4))
            if COND_PREV_FINAL and prev_text:
                kwargs["initial_prompt"] = prev_text[-400:]  # limit prompt length
                kwargs["condition_on_previous_text"] = True
        segments, _ = await asyncio.to_thread(model.transcribe, audio_f32, **kwargs)
        dt = time.time() - t0
        text, metrics = _segments_to_text_and_metrics(segments)
        # Real-time factor (RTF) estimation
        audio_dur = len(buf_i16) / SAMPLE_RATE
        rtf = dt / max(1e-6, audio_dur)
        metrics["rtf"] = rtf
        metrics["decode_sec"] = dt
        return text, metrics

    last_interim_sent = ""
    last_final_sent = ""

    async def do_interim(buf: np.ndarray):
        nonlocal last_interim_sent
        async with infer_lock:
            try:
                interim_text, metrics = await run_transcribe(buf, fast=True, prev_text=last_final_sent)
                # If model falling behind, skip interims to preserve finals
                if metrics.get("rtf", 0) > float(os.getenv("FWS_MAX_INTERIM_RTF", "1.1")):
                    log.info(f"skip interim due to high RTF {metrics['rtf']:.2f}")
                    return
                # Filter noisy/hallucinated outputs
                if not interim_text:
                    return
                if _looks_repetitive(interim_text):
                    log.debug("drop interim: repetitive")
                    return
                if metrics.get("max_no_speech") and metrics["max_no_speech"] > 0.9:
                    log.debug("drop interim: no_speech high")
                    return
                if interim_text == last_interim_sent:
                    return
                last_interim_sent = interim_text
                await ws.send_text(json.dumps({"type": "interim", "text": interim_text}))
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "error": f"interim: {e}"}))

    # Tail overlap between utterances to avoid boundary word loss
    TAIL_SAMPLES = int(os.getenv("FWS_TAIL_SAMPLES", str(SAMPLE_RATE // 2)))  # 0.5s

    async def do_final(buf: np.ndarray):
        nonlocal last_final_sent
        async with infer_lock:
            try:
                final_text, metrics = await run_transcribe(buf, fast=False, prev_text=last_final_sent)
                if metrics.get("rtf") is not None:
                    log.info(f"final decode rtf={metrics['rtf']:.2f} dur={len(buf)/SAMPLE_RATE:.2f}s dt={metrics['decode_sec']:.2f}s")
                if not final_text:
                    return
                if _looks_repetitive(final_text):
                    log.info("drop final: repetitive pattern")
                    return
                if metrics.get("avg_logprob") is not None and metrics["avg_logprob"] < -1.0:
                    log.info("drop final: low avg_logprob")
                    return
                if metrics.get("max_no_speech") and metrics["max_no_speech"] > 0.9:
                    log.info("drop final: high no_speech")
                    return
                if final_text == last_final_sent:
                    return
                # Provide final text and model name/meta
                last_final_sent = final_text
                await ws.send_text(json.dumps({"type": "final", "text": final_text, "model": MODEL_NAME}))
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "error": f"final: {e}"}))

    try:
        while True:
            packet = await ws.receive_bytes()
            # Expect packet length == SAMPLES_PER_FRAME * SAMPLE_WIDTH
            if len(packet) != SAMPLES_PER_FRAME * SAMPLE_WIDTH:
                # Ignore malformed frames
                continue

            # (Raw capture removed)

            # VAD state per frame
            is_voice = vad_keep(packet)
            total_frames += 1

            chunk_i16 = bytes_to_int16(packet)
            audio_q.extend(chunk_i16.tolist())

            now = time.time()

            # Energy fallback classification if VAD says silence
            if not is_voice and ENERGY_ENABLE:
                samples = bytes_to_int16(packet).astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(samples * samples) + 1e-12)
                if rms > ENERGY_FRAME_THRESHOLD:
                    is_voice = True

            if is_voice:
                # Enter/continue speech
                if not in_speech:
                    # Starting speech: prepend pre-roll audio to utterance
                    if pre_roll:
                        utter_q.extend(list(pre_roll))
                    in_speech = True
                silence_frames = 0
                utter_q.extend(chunk_i16.tolist())
                pre_roll.clear()

                # Interim update (skip if an inference is already running)
                if now - last_emit >= EMIT_EVERY and len(utter_q) >= SAMPLE_RATE * 1.0 and not infer_lock.locked():
                    last_emit = now
                    win = min(len(utter_q), SAMPLE_RATE * 2)
                    buf = np.array(list(utter_q)[-win:], dtype=np.int16)
                    asyncio.create_task(do_interim(buf))
            else:
                if in_speech:
                    # Tolerate brief silences inside speech
                    silence_frames += 1
                    if silence_frames <= ALLOW_SHORT_SILENCE_FRAMES:
                        # treat as still speech; keep the frame
                        utter_q.extend(chunk_i16.tolist())
                    else:
                        # Real silence region
                        if silence_frames >= (SILENCE_FRAMES_FOR_FINAL + GRACE_FRAMES) and len(utter_q) >= int(SAMPLE_RATE * MIN_UTTERANCE_SECONDS):
                            buf = np.array(list(utter_q), dtype=np.int16)
                            # Save tail for next utterance start
                            tail = buf[-TAIL_SAMPLES:].tolist() if len(buf) > TAIL_SAMPLES else buf.tolist()
                            utter_q.clear()
                            pre_roll.extend(tail)  # seed next utterance with tail overlap
                            in_speech = False
                            silence_frames = 0
                            asyncio.create_task(do_final(buf))
                        elif silence_frames >= (SILENCE_FRAMES_FOR_FINAL + GRACE_FRAMES):
                            # too short, discard utterance but reset state
                            utter_q.clear()
                            in_speech = False
                            silence_frames = 0
                else:
                    # Accumulate pre-roll frames while in silence
                    pre_roll.extend(chunk_i16.tolist())
                # else: still in silence; do nothing

            # periodic diagnostics
            if time.time() - last_latency_log > 5:
                last_latency_log = time.time()
                active_len = len(utter_q) / SAMPLE_RATE
                log.info(f"frames={total_frames} active_len={active_len:.2f}s in_speech={in_speech} lock={infer_lock.locked()}")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_text(json.dumps({"type":"error","error":str(e)}))

    # (No capture/log cleanup required)
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await stream_transcribe(ws)
