import os
import tempfile
from pathlib import Path

import streamlit as st

from transcriber import DEFAULT_MODEL_ID, WhisperTranscriber, resolve_runtime

st.set_page_config(page_title="한국어 STT 테스트", page_icon="🎙️", layout="wide")

AUDIO_EXTENSIONS = ["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4"]


@st.cache_resource(show_spinner=False)
def load_transcriber(model_id: str = DEFAULT_MODEL_ID) -> WhisperTranscriber:
    return WhisperTranscriber(model_id=model_id)


def format_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def transcribe_audio_bytes(data: bytes, suffix: str, **options):
    suffix = suffix if suffix.startswith(".") else f".{suffix or 'wav'}"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
        return load_transcriber(DEFAULT_MODEL_ID).transcribe(path, **options)
    finally:
        Path(path).unlink(missing_ok=True)


def run_transcription(data: bytes, suffix: str, source: str, **options) -> None:
    st.session_state.last_error = None
    try:
        st.session_state.last_result = transcribe_audio_bytes(data, suffix, **options)
        st.session_state.last_source = source
    except Exception as exc:
        st.session_state.last_result = None
        st.session_state.last_error = str(exc)


def render_result(result) -> None:
    st.subheader("인식 결과")
    if not result.text:
        st.warning("인식된 텍스트가 없습니다.")
        return

    st.text_area("전체 텍스트", result.text, height=180)
    meta = []
    if result.language:
        meta.append(f"언어: `{result.language}`")
    if result.language_probability is not None:
        meta.append(f"언어 확률: `{result.language_probability:.2%}`")
    if result.duration is not None:
        meta.append(f"길이: `{result.duration:.2f}초`")
    if meta:
        st.caption(" · ".join(meta))

    rows = [
        {
            "시작": format_timestamp(segment.start),
            "끝": format_timestamp(segment.end),
            "텍스트": segment.text,
        }
        for segment in result.segments
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


st.title("한국어 STT 테스트")
st.caption(f"모델: `{DEFAULT_MODEL_ID}` · faster-whisper / CTranslate2")

device, compute_type = resolve_runtime()
with st.sidebar:
    st.header("설정")
    st.write(f"런타임: `{device}` / `{compute_type}`")
    beam_size = st.slider("Beam size", min_value=1, max_value=10, value=5)
    vad_filter = st.checkbox("VAD 필터", value=True)
    auto_language = st.checkbox("언어 자동 감지", value=False)
    language = None if auto_language else "ko"
    st.info("첫 실행 시 Hugging Face에서 large-v3-turbo 모델을 받습니다.")

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_source" not in st.session_state:
    st.session_state.last_source = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None

options = {"language": language, "beam_size": beam_size, "vad_filter": vad_filter}
upload_tab, mic_tab = st.tabs(["파일 업로드", "마이크 녹음"])

with upload_tab:
    uploaded = st.file_uploader(
        "오디오/영상 파일을 올려주세요",
        type=AUDIO_EXTENSIONS,
        accept_multiple_files=False,
    )
    if uploaded is not None:
        st.audio(uploaded.getvalue())
        if st.button("파일 변환", type="primary", key="transcribe_file"):
            suffix = Path(uploaded.name).suffix or ".wav"
            with st.spinner("모델을 준비하고 음성을 인식하는 중..."):
                run_transcription(uploaded.getvalue(), suffix, "파일 업로드", **options)

with mic_tab:
    recording = st.audio_input("마이크에 대고 말한 뒤 변환하세요")
    if recording is not None:
        st.audio(recording)
        if st.button("녹음 변환", type="primary", key="transcribe_mic"):
            suffix = Path(recording.name).suffix or ".wav"
            with st.spinner("모델을 준비하고 음성을 인식하는 중..."):
                run_transcription(recording.getvalue(), suffix, "마이크 녹음", **options)

if st.session_state.last_error:
    st.error(st.session_state.last_error)
elif st.session_state.last_result is not None:
    if st.session_state.last_source:
        st.caption(f"출처: {st.session_state.last_source}")
    render_result(st.session_state.last_result)
