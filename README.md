# STT openai/whisper-large-v3-turbo

`openai/whisper-large-v3-turbo`로 한국어 음성을 텍스트로 변환하는 테스트 앱입니다.

- Streamlit: 파일 업로드 / 마이크 녹음 UI
- FastAPI: 오디오 파일을 올리면 인식 텍스트를 반환하는 API
- 추론: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2)

Hugging Face의 Transformers 가중치 대신, 같은 모델의 CTranslate2 변환본인 `large-v3-turbo`를 로드합니다.

## 요구 사항

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- GPU가 있으면 CUDA `float16`, 없으면 CPU `int8`

첫 변환 시 Hugging Face에서 모델을 받습니다.

## 설치

```bash
uv sync --group dev
```

## Streamlit

```bash
uv run streamlit run app.py
```

브라우저에서 [http://localhost:8501](http://localhost:8501) 을 엽니다.

- 파일 업로드: WAV, MP3, M4A, FLAC, OGG, WEBM, MP4
- 마이크 녹음 후 변환
- Beam size, VAD, 언어 자동 감지

## FastAPI

```bash
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 엔드포인트: `POST /transcribe`

```bash
curl -X POST http://127.0.0.1:8000/transcribe \
  -F "file=@sample.wav"
```

응답 예시:

```json
{"text": "안녕하세요 테스트입니다"}
```

빈 파일은 `400`을 반환합니다.

## 테스트

```bash
uv run pytest
```
