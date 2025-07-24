# Clueless: AI Video Tutor Platform

A full-stack platform combining a Chrome extension and a FastAPI backend to deliver real-time, in-player AI tutoring for video content. The system transcribes video audio, stores transcripts, and uses OpenAI GPT-4o to answer user questions about the video context.

---

## Table of Contents
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Usage Guide](#usage-guide)
- [Backend API Reference](#backend-api-reference)
- [Development Workflow](#development-workflow)
- [Security & Best Practices](#security--best-practices)
- [Technologies Used](#technologies-used)
- [Limitations](#limitations)
- [License](#license)

---

## Features

### Frontend (Chrome Extension)
- **React + TypeScript** overlay injected into every `<video>` element
- **Manifest V3** with host permissions for all URLs
- **Web Audio API** for tab audio capture
- **Real-time audio streaming** to backend via WebSocket
- **Translucent control bar** with Ask, Settings buttons
- **Chat interface** for querying video content
- **Tailwind CSS** for modern styling
- **Automatic reconnection** and error handling for WebSocket
- **Streaming AI responses** in chat interface

### Backend (FastAPI)
- **WebSocket endpoint** for audio streaming and transcription
- **Whisper transcription** with audio buffering
- **SQLite database** for transcript segments
- **TF-IDF vector search** (scikit-learn) for context retrieval
- **Streaming GPT-4o responses** (Server-Sent Events)
- **Retrieval-Augmented Generation (RAG)** for question answering
- **CORS** enabled for extension communication
- **Health check and transcript endpoints**

---

## Architecture Overview

### High-Level Flow
1. **User opens a video page** in Chrome.
2. **Extension injects overlay** and starts capturing audio.
3. **Audio is streamed** to the backend via WebSocket.
4. **Backend transcribes audio** using Whisper and stores transcript segments in SQLite.
5. **User asks a question** in the overlay chat.
6. **Frontend sends the question** (with video ID and timestamp) to the backend.
7. **Backend finds relevant transcript segments** using TF-IDF similarity.
8. **Backend builds a prompt** with transcript context and user question, sends it to OpenAI GPT-4o.
9. **AI response is streamed** back to the frontend and displayed to the user.

### Component Diagram
```
[User] <-> [Chrome Extension (React/TS)] <-> [FastAPI Backend] <-> [Whisper, SQLite, OpenAI API]
```

### Audio Processing Flow
1. Content script captures tab audio via Web Audio API
2. PCM audio chunks stream to `/audio` WebSocket endpoint
3. Backend buffers audio, converts to WAV, transcribes with Whisper
4. Transcript segments saved to SQLite with timestamps

### Query Processing Flow
1. User asks question via overlay chat interface
2. Frontend sends `{videoId, timestamp, prompt}` to `/query` endpoint
3. Backend finds relevant transcript segments using TF-IDF similarity
4. RAG prompt construction with transcript context
5. Streaming GPT-4o response via Server-Sent Events

### Database Schema
```sql
CREATE TABLE segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Setup & Installation

### Prerequisites
- Node.js (v18+ recommended) and npm
- Python 3.11+
- [OpenAI API key](https://platform.openai.com/account/api-keys)

### 1. Clone the Repository
```bash
git clone https://github.com/sohumt123/VidLearn.git
cd VidLearn
```

### 2. Environment Variables
Create a `.env` file in the project root and in `backend/` as needed. Example:
```
# .env (root or backend)
OPENAI_API_KEY=sk-...
# Add other secrets as needed
```

### 3. Chrome Extension Setup
```bash
cd chrome-extension
npm install
npm run build
```
- Load the extension: Go to `chrome://extensions/`, enable Developer mode, and load the `chrome-extension/dist` folder.

### 4. Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Ensure .env is present with your OpenAI API key
./start.sh
```

---

## Environment Variables
- `OPENAI_API_KEY` (required): Your OpenAI API key for GPT-4o access.
- (Optional) Add any other secrets or configuration variables as needed for your deployment.
- **Security:** `.env` is gitignored and never pushed to GitHub.

---

## Usage Guide

1. **Start the backend** (`./start.sh` in `backend/`)
2. **Load the Chrome extension** in your browser
3. **Navigate to any video page**
4. **Click the "Ask" button** on the overlay to open the chat
5. **Ask questions** about the video content; the backend will transcribe, search, and answer in real time

---

## Backend API Reference

### `GET /health`
- Returns backend and model status.

### `WebSocket /audio`
- Receives PCM audio chunks, buffers, transcribes, and stores transcript segments.
- Responds with transcriptions in real time.

### `POST /query`
- Accepts `{ videoId, timestamp, prompt }` JSON.
- Finds relevant transcript segments, builds a RAG prompt, streams GPT-4o response.

### `GET /transcript/{video_id}`
- Returns all transcript segments for a given video.

---

## Development Workflow

### Frontend (Chrome Extension)
```bash
cd chrome-extension
npm run dev    # Development server with hot reload
npm run build  # Production build
```

### Backend
```bash
cd backend
source venv/bin/activate
python main.py  # Run with auto-reload for development
```

### Testing
- Use the extension on any video site and monitor backend logs for transcription and query handling.
- Use `/health` endpoint to verify backend status.

---

## Security & Best Practices
- **Never commit secrets:** `.env` is in `.gitignore`.
- **API keys** are loaded at runtime using `python-dotenv`.
- **Push protection:** GitHub will block pushes containing secrets.
- **CORS** is enabled for extension-backend communication.
- **Error handling:** Backend will error if `OPENAI_API_KEY` is missing or invalid.

---

## Technologies Used
- **Frontend:** React, TypeScript, Tailwind CSS, Chrome Extension APIs, Web Audio API
- **Backend:** FastAPI, Whisper, OpenAI GPT-4o, SQLite, scikit-learn, WebSockets
- **Infrastructure:** Chrome Manifest V3, CORS, Server-Sent Events, python-dotenv

---

## Limitations
- Requires user permission for tab audio capture
- Audio transcription accuracy depends on video quality
- Backend requires OpenAI API key and incurs costs per query
- Uses scikit-learn TF-IDF instead of FAISS for vector similarity
- Whisper model runs on CPU by default (can be slow for long audio)

---

## License

MIT 