from types import SimpleNamespace

import pytest

from transcriber import (
    DEFAULT_MODEL_ID,
    TranscriptionResult,
    WhisperTranscriber,
    collect_result,
    resolve_inference_model_id,
    resolve_runtime,
)


class FakeSegment:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


class FakeModel:
    def __init__(self):
        self.calls: list[dict] = []
        self.segments = [
            FakeSegment(0.0, 1.2, "안녕하세요"),
            FakeSegment(1.2, 2.5, "테스트입니다"),
        ]
        self.info = SimpleNamespace(
            language="ko",
            language_probability=0.98,
            duration=2.5,
        )

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, **kwargs})
        return iter(self.segments), self.info


def test_resolve_runtime_uses_cuda_float16_when_gpu_available():
    assert resolve_runtime(cuda_available=True) == ("cuda", "float16")


def test_resolve_runtime_uses_cpu_int8_when_gpu_unavailable():
    assert resolve_runtime(cuda_available=False) == ("cpu", "int8")


def test_collect_result_joins_segment_text_and_keeps_timestamps():
    segments = [
        FakeSegment(0.0, 1.0, " 안녕 "),
        FakeSegment(1.0, 2.0, "하세요"),
    ]
    info = SimpleNamespace(language="ko", language_probability=0.9, duration=2.0)

    result = collect_result(segments, info)

    assert result.text == "안녕 하세요"
    assert result.language == "ko"
    assert result.language_probability == pytest.approx(0.9)
    assert result.duration == pytest.approx(2.0)
    assert result.segments[0].start == pytest.approx(0.0)
    assert result.segments[0].end == pytest.approx(1.0)
    assert result.segments[0].text == "안녕"
    assert result.segments[1].text == "하세요"


def test_collect_result_returns_empty_text_when_no_segments():
    info = SimpleNamespace(language="ko", language_probability=0.1, duration=0.0)

    result = collect_result([], info)

    assert result == TranscriptionResult(
        text="",
        language="ko",
        language_probability=0.1,
        duration=0.0,
        segments=(),
    )


def test_transcriber_default_model_id_is_openai_turbo():
    assert DEFAULT_MODEL_ID == "openai/whisper-large-v3-turbo"


def test_openai_turbo_maps_to_faster_whisper_large_v3_turbo():
    assert resolve_inference_model_id("openai/whisper-large-v3-turbo") == "large-v3-turbo"


def test_transcriber_loads_faster_whisper_alias_for_openai_turbo():
    calls: list[dict] = []

    def fake_factory(model_id, **kwargs):
        calls.append({"model_id": model_id, **kwargs})
        return FakeModel()

    WhisperTranscriber(model_factory=fake_factory)

    assert calls[0]["model_id"] == "large-v3-turbo"


def test_transcriber_transcribes_audio_with_korean_defaults():
    model = FakeModel()
    transcriber = WhisperTranscriber(model=model)

    result = transcriber.transcribe("sample.wav")

    assert result.text == "안녕하세요 테스트입니다"
    assert result.language == "ko"
    assert len(result.segments) == 2
    assert model.calls[0]["audio"] == "sample.wav"
    assert model.calls[0]["language"] == "ko"
    assert model.calls[0]["beam_size"] == 5
    assert model.calls[0]["vad_filter"] is True
    assert model.calls[0]["task"] == "transcribe"


def test_transcriber_forwards_custom_options():
    model = FakeModel()
    transcriber = WhisperTranscriber(model=model)

    transcriber.transcribe(
        "clip.mp3",
        language="en",
        beam_size=1,
        vad_filter=False,
    )

    assert model.calls[0]["language"] == "en"
    assert model.calls[0]["beam_size"] == 1
    assert model.calls[0]["vad_filter"] is False
