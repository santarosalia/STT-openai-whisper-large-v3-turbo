from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from transcriber import WhisperTranscriber

app = FastAPI(title="STT API")

_transcriber: WhisperTranscriber | None = None


def get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
    return _transcriber


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    transcriber: WhisperTranscriber = Depends(get_transcriber),
) -> dict[str, str]:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일은 변환할 수 없습니다.")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
        result = transcriber.transcribe(path)
    finally:
        Path(path).unlink(missing_ok=True)

    return {"text": result.text}
