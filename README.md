# Medical Voice Assistant 🎙️

A medical audio processing API that transcribes clinical recordings, extracts structured medical features, and generates follow-up questions — streamed in real time via Server-Sent Events (SSE).

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation & Running](#installation--running)
- [API Reference](#api-reference)
- [SSE Event Reference](#sse-event-reference)

---

## Overview

MedVoice accepts an audio file from a clinician or a doctor-patient conversation, runs it through a multi-step pipeline, and streams results back as they complete:

1. **Transcription** — Whisper via Fireworks AI
2. **Refinement** — LLM cleanup of raw transcript (DeepSeek)
3. **Translation** — Arabic → English (when applicable)
4. **Feature Extraction** — Structured medical data extraction (DeepSeek)
5. **Question Generation** — Follow-up questions the doctor should ask (DeepSeek)

---

## How It Works

```
Audio File
    │
    ▼
[1] Transcription (Fireworks Whisper)
    │
    ▼
[2] Refinement (DeepSeek)
    │
    ▼
[3] Translation Arabic → English (DeepSeek)  ← Arabic only
    │
    ▼
    ├──────────────────────────────────┐
[4] Feature Extraction (DeepSeek)    [5] Question Generation (DeepSeek)
    │   (parallel)                     │   (parallel)
    └──────────────────────────────────┘
    │
    ▼
SSE Stream → Client
```

Steps 4 and 5 run **in parallel** to minimize total latency.

Results are **streamed progressively** — the client receives each event as soon as that step finishes, without waiting for the full pipeline to complete.

---

## Project Structure

```
medvoice/
├── src/
│   ├── app.py                  # FastAPI entrypoint
│   ├── core/
│   │   └── config.py           # Environment config
│   ├── model/
│   │   ├── speech_service.py   # Whisper transcription
│   │   ├── llm_service.py      # LLM wrapper (refine, extract, questions)
│   │   ├── refine_text.py      # Refinement Module
│   │   ├── translation.py      # Arabic → English translation
│   │   ├── extract_features.py # Feature extraction module
│   │   └── question_generator.py # Question generation module
│   ├── utils/
│   │   └── prompt.py           # Prompt templates
│   ├── uploads/                # Temp audio file storage
│   ├── .env                # Environment Variables
│   └── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Installation & Running

### Option 1 — Docker (recommended)

**Prerequisites:** Docker and Docker Compose installed.

```bash
# Clone the repo
git clone <repo-url>
cd medvoice

# Build and start
docker compose up --build

# Or detached
docker compose up --build -d
```

The API will be available at `http://localhost:9999`.

### Option 2 — Local (UV)

**Prerequisites:** Python 3.11+, [UV](https://github.com/astral-sh/uv) installed.

```bash
cd src

# Install dependencies
uv pip install --system -r requirements.txt

# Run
python app.py
```

---

## API Reference

### `POST /audio/process`

Runs the full audio processing pipeline and streams results via SSE.

**Content-Type:** `multipart/form-data`

#### Request Parameters

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | File | ✅ | — | Audio file (mp3, wav, m4a, ogg, …) |
| `language` | string | ✅ | `ar` | Audio language: `ar` (Arabic) or `en` (English) |
| `mode` | string | ✅ | `doctor` | `doctor` — single clinician recording; `conversation` — doctor-patient dialogue |

#### Example Request

```bash
curl -X POST http://localhost:9999/audio/process \
  -F "file=@recording.mp3" \
  -F "language=en" \
  -F "mode=doctor" \
  --no-buffer
```

#### Example Request (Python)

```python
import httpx

with open("recording.mp3", "rb") as f:
    with httpx.stream(
        "POST",
        "http://localhost:9999/audio/process",
        data={"language": "en", "mode": "doctor"},
        files={"file": ("recording.mp3", f, "audio/mpeg")},
        timeout=300,
    ) as response:
        for line in response.iter_lines():
            print(line)
```

---

## SSE Event Reference

The endpoint streams Server-Sent Events. Each event has a named `event` field and a JSON `data` payload.

### `transcription`

Fired after transcription, refinement, and translation complete.

```json
{
  "final_text": "Patient presents with chest pain radiating to the left arm...",
  "transcription_sec": 4.821,
  "refinement_sec": 2.103,
  "translation_sec": 1.847
}
```

| Field | Type | Description |
|---|---|---|
| `final_text` | string | Final English text after refinement and translation |
| `transcription_sec` | float | Time taken for Whisper transcription |
| `refinement_sec` | float | Time taken for LLM refinement |
| `translation_sec` | float \| null | Time taken for translation (null if English input) |

---

### `extraction`

Fired after structured medical features are extracted.

```json
{
  "extracted_features": {
    "chief_complaint": "Chest pain radiating to left arm",
    "history_of_illness": "Hypertension for 10 years",
    "current_medication": "Amlodipine 5mg daily",
    "assessment": "Possible acute coronary syndrome",
    "plan": "ECG, troponin levels, cardiology consult",
    "follow_up": "Return in 48 hours or go to ER if pain worsens"
  },
  "extraction_sec": 3.412
}
```

---

### `questions`

Fired after follow-up questions are generated. Also includes total pipeline duration.

```json
{
  "questions": [
    {
      "question": "How long have you been experiencing chest pain?",
      "answer": null,
      "needs_asking": true,
      "category": "History"
    },
    {
      "question": "Do you have a family history of heart disease?",
      "answer": null,
      "needs_asking": true,
      "category": "Family History"
    }
  ],
  "question_generation_sec": 3.412,
  "total_sec": 14.283
}
```

| Field | Type | Description |
|---|---|---|
| `questions` | array | List of generated question objects |
| `question.question` | string | The question to ask |
| `question.answer` | string \| null | Answer if already present in transcript, otherwise null |
| `question.needs_asking` | boolean | Whether the doctor still needs to ask this |
| `question.category` | string | Medical category (History, Medications, etc.) |
| `total_sec` | float | Total end-to-end pipeline duration in seconds |

---

### `error`

Fired if any pipeline step fails. The stream stops after this event.

```json
{
  "step": "transcription",
  "detail": "Audio file not found: /tmp/audio_xyz.mp3"
}
```

| Field | Description |
|---|---|
| `step` | Which pipeline step failed: `transcription`, `extraction/questions`, or `unknown` |
| `detail` | Error message |