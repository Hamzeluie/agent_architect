import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tts.agent_architect.agent_architect.call_agent import *

# --- Configuration ---


TARGET_STREAM_SAMPLE_RATE = 8000
CHUNK_DURATION_MS = 40
CHUNK_SAMPLES = int(TARGET_STREAM_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# ✅ Add global flag to detect if we received any VAD event
received_vad_event = False

async def main():
    """Sends audio chunks to server."""
    AUDIO_FILE_PATH = Path("/home/mehdi/Documents/projects/tts/AI_STT/test.wav")
    global received_vad_event  # ← We'll use this to validate test success

    print(f"🎙️ Streaming audio from '{AUDIO_FILE_PATH}'...")

    audio_data, source_sr = librosa.load(AUDIO_FILE_PATH, sr=None, mono=True)
    print(f"Source audio loaded: {len(audio_data)/source_sr:.2f}s @ {source_sr}Hz")

    if source_sr != TARGET_STREAM_SAMPLE_RATE:
        print(f"Resampling from {source_sr}Hz → {TARGET_STREAM_SAMPLE_RATE}Hz...")
        audio_data = librosa.resample(
            y=audio_data, orig_sr=source_sr, target_sr=TARGET_STREAM_SAMPLE_RATE
        )

    audio_int16 = (audio_data * 32767).astype(np.int16)
    start_index = 0
    total_chunks = 0
    
    
    service = InferenceService(agent_name="call", timeout=100000)
    await service.start()
    await service.start_session("client_1")

    while start_index < len(audio_int16):
        end_index = start_index + CHUNK_SAMPLES
        chunk = audio_int16[start_index:end_index]
        if len(chunk) == 0:
            break

        audio_bytes = chunk.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # send EACH CHUNK
        result1 = await service.predict(input_data=audio_b64, sid="client_1")
        await service.stop()
        
        total_chunks += 1
        start_index = end_index
        await asyncio.sleep(CHUNK_DURATION_MS / 1000.0)

    print(f"✅ Sent {total_chunks} chunks. Finished streaming.")




if __name__ == "__main__":
    asyncio.run(main())
