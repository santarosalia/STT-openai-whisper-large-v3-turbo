from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

DEFAULT_MODEL_ID = "openai/whisper-large-v3-turbo"
FASTER_WHISPER_MODEL_ALIASES = {
    "openai/whisper-large-v3-turbo": "large-v3-turbo",
}


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    language_probability: float | None
    duration: float | None
    segments: tuple[TranscriptSegment, ...]


def resolve_inference_model_id(model_id: str) -> str:
    return FASTER_WHISPER_MODEL_ALIASES.get(model_id, model_id)


def resolve_runtime(cuda_available: bool | None = None) -> tuple[str, str]:
    if cuda_available is None:
        cuda_available = _detect_cuda()
    if cuda_available:
        return "cuda", "float16"
    return "cpu", "int8"


def _detect_cuda() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def collect_result(segments: Iterable[Any], info: Any) -> TranscriptionResult:
    parsed = tuple(
        TranscriptSegment(start=segment.start, end=segment.end, text=segment.text.strip())
        for segment in segments
        if getattr(segment, "text", "").strip()
    )
    return TranscriptionResult(
        text=" ".join(segment.text for segment in parsed),
        language=getattr(info, "language", None),
        language_probability=getattr(info, "language_probability", None),
        duration=getattr(info, "duration", None),
        segments=parsed,
    )


class WhisperTranscriber:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        model: Any | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        model_factory: Any | None = None,
    ) -> None:
        self.model_id = model_id
        if model is not None:
            self._model = model
            return

        resolved_device, resolved_compute_type = resolve_runtime()
        factory = model_factory
        if factory is None:
            from faster_whisper import WhisperModel

            factory = WhisperModel

        self._model = factory(
            resolve_inference_model_id(model_id),
            device=device or resolved_device,
            compute_type=compute_type or resolved_compute_type,
        )

    def transcribe(
        self,
        audio: Any,
        language: str = "ko",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            task="transcribe",
        )
        return collect_result(list(segments), info)
