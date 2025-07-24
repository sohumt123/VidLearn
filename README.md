# AI Video Tutor Chrome Extension

A minimal-viable Chrome extension + FastAPI backend that delivers real-time, in-player tutoring for video content.

## Features

🔹 **Frontend (Chrome Extension)**
- React + TypeScript overlay injected into every `<video>` element
- Manifest V3 with host permissions for all URLs
- Content script with Web Audio API for tab audio capture
- Real-time audio streaming to backend via WebSocket
- Translucent control bar with Ask, Settings buttons
- Chat interface for querying video content
- Tailwind CSS styling

🔹 **Backend (FastAPI)**
- WebSocket endpoint for audio processing
- Whisper transcription with audio buffering
- SQLite database for transcript segments
- TF-IDF vector search (scikit-learn replacement for FAISS)
- Streaming GPT-4o responses
- RAG-based question answering

## Setup

### Prerequisites
- Node.js and npm
- Python 3.11+
- OpenAI API key

### Chrome Extension
```bash
cd chrome-extension
npm install
npm run build
```

### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your_api_key_here"

# Start the backend
./start.sh
```

## Usage

1. **Load Extension**: Go to Chrome Extensions (chrome://extensions/), enable Developer mode, and load the `chrome-extension/dist` folder
2. **Start Backend**: Run the backend server on localhost:8000
3. **Use Extension**: Navigate to any video page, click the "Ask" button on the video overlay
4. **Real-time Tutoring**: Audio is automatically transcribed, and you can ask questions about the video content

## Architecture

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
    id INTEGER PRIMARY KEY,
    video_id TEXT,
    start_time REAL,
    end_time REAL,
    text TEXT,
    created_at TIMESTAMP
);
```

## Development

### Extension Build
```bash
cd chrome-extension
npm run dev    # Development server
npm run build  # Production build
```

### Backend Development
```bash
cd backend
source venv/bin/activate
python main.py  # Run with auto-reload
```

## API Endpoints

- `GET /health` - Health check
- `WebSocket /audio` - Audio streaming and transcription
- `POST /query` - RAG-based question answering (streaming)
- `GET /transcript/{video_id}` - Get full transcript

## Technologies

**Frontend**: React, TypeScript, Tailwind CSS, Web Audio API, Chrome Extension APIs  
**Backend**: FastAPI, Whisper, OpenAI GPT-4o, SQLite, scikit-learn, WebSockets  
**Infrastructure**: Chrome Manifest V3, CORS, Server-Sent Events

## Limitations

- Requires user permission for tab audio capture
- Audio transcription accuracy depends on video quality
- Backend requires OpenAI API key and costs per query
- Uses scikit-learn TF-IDF instead of FAISS for vector similarity

## License

MIT 