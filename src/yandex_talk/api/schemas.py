from pydantic import BaseModel


class RecognizedSpeech(BaseModel):
    speaker_id: int
    text: str
    timestamp: int


class RecognizedSegments(BaseModel):
    segments: list[RecognizedSpeech]
