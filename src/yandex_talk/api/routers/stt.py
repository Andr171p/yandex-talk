from fastapi import APIRouter, status, File, Query

import numpy as np

from src.yandex_talk.constants import DEFAULT_SAMPLE_RATE, MODEL_PATH
from src.yandex_talk.stt import SpeechRecognizer
from ..schemas import RecognizedSpeech, RecognizedSegments


speech_recognizer = SpeechRecognizer(
    model_path=MODEL_PATH,
    sample_rate=DEFAULT_SAMPLE_RATE
)


stt_router = APIRouter(
    prefix="/api/v1/speech",
    tags=["Speech to text"]
)


@stt_router.post(
    path="/recognize",
    status_code=status.HTTP_200_OK,
)
async def recognize_speech(
        audio_bytes: bytes = File(...),
        sample_rate: int = Query(DEFAULT_SAMPLE_RATE, description="Частота дискретизации")
) -> RecognizedSegments:
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    speech_recognizer.start_listening()
    chunk_size = sample_rate * 0.1  # 100ms chunks
    for i in range(0, len(audio_data), int(chunk_size)):
        chunk = audio_data[i:i + int(chunk_size)]
        if len(chunk) > 0:
            speech_recognizer.process_audio_chunk(chunk)
    speech_recognizer.stop_listening()
    transcript = speech_recognizer.get_transcript()
    segments: list[RecognizedSpeech] = []
    for speaker, segments in transcript.items():
        for segment in segments:
            segments.append(RecognizedSpeech(
                speaker_id=speaker,
                text=segment["text"],
                timestamp=segment["timestamp"]
            ))
    return RecognizedSegments(segments=segments)
