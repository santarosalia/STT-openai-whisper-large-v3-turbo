from pathlib import Path

from fastapi.testclient import TestClient

from api import app, get_transcriber
from transcriber import TranscriptionResult


class FakeTranscriber:
    def transcribe(self, audio, **kwargs):
        path = Path(audio)
        assert path.exists()
        assert path.read_bytes() == b"fake-audio"
        return TranscriptionResult(
            text="안녕하세요 테스트입니다",
            language="ko",
            language_probability=0.99,
            duration=1.5,
            segments=(),
        )


def test_transcribe_returns_text_from_uploaded_file():
    app.dependency_overrides[get_transcriber] = lambda: FakeTranscriber()
    client = TestClient(app)

    try:
        response = client.post(
            "/transcribe",
            files={"file": ("sample.wav", b"fake-audio", "audio/wav")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"text": "안녕하세요 테스트입니다"}


def test_transcribe_rejects_empty_file():
    app.dependency_overrides[get_transcriber] = lambda: FakeTranscriber()
    client = TestClient(app)

    try:
        response = client.post(
            "/transcribe",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json()["detail"] == "빈 파일은 변환할 수 없습니다."
