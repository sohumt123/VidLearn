import asyncio
import io
import json
import sqlite3
import time
import wave
from typing import Dict, List, Optional
import logging

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from faster_whisper import WhisperModel
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Video Tutor Backend")

# Enable CORS for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    videoId: str
    timestamp: float
    prompt: str

class TranscriptSegment(BaseModel):
    id: Optional[int] = None
    video_id: str
    start_time: float
    end_time: float
    text: str
    embedding: Optional[List[float]] = None

# Global variables
whisper_model = None
openai_client = None
active_connections: Dict[str, WebSocket] = {}
audio_buffers: Dict[str, List[bytes]] = {}
transcript_cache: Dict[str, List[TranscriptSegment]] = {}
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

load_dotenv()

def init_database():
    """Initialize SQLite database for transcript segments."""
    conn = sqlite3.connect('transcripts.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Get database connection."""
    return sqlite3.connect('transcripts.db')

def save_segment(segment: TranscriptSegment):
    """Save transcript segment to database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO segments (video_id, start_time, end_time, text)
        VALUES (?, ?, ?, ?)
    ''', (segment.video_id, segment.start_time, segment.end_time, segment.text))
    
    segment.id = cursor.lastrowid
    conn.commit()
    conn.close()

def get_segments_for_video(video_id: str) -> List[TranscriptSegment]:
    """Retrieve all segments for a video."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, video_id, start_time, end_time, text
        FROM segments 
        WHERE video_id = ?
        ORDER BY start_time
    ''', (video_id,))
    
    segments = []
    for row in cursor.fetchall():
        segments.append(TranscriptSegment(
            id=row[0],
            video_id=row[1],
            start_time=row[2],
            end_time=row[3],
            text=row[4]
        ))
    
    conn.close()
    return segments

async def load_models():
    """Load Whisper and OpenAI models."""
    global whisper_model, openai_client
    
    try:
        # Ensure .env is loaded
        load_dotenv()
        logger.info("Loading Whisper model...")
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully")
        
        logger.info("Initializing OpenAI client...")
        api_key = os.getenv("OPENAI_API_KEY")
        logger.info(f"API key read from env: {api_key[:20] if api_key else 'None'}...")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("Please set your OPENAI_API_KEY in the .env file")
        
        openai_client = openai.OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Convert PCM data to WAV format."""
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        return wav_buffer.getvalue()

def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe audio using Whisper."""
    if not whisper_model:
        return ""
    
    try:
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe with Whisper
        segments, info = whisper_model.transcribe(audio_np, beam_size=5)
        
        # Combine all segments into single text
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        return " ".join(text_parts)
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

def find_relevant_segments(video_id: str, query: str, max_segments: int = 5) -> List[TranscriptSegment]:
    """Find relevant transcript segments using TF-IDF similarity."""
    # Get segments from cache or database
    if video_id in transcript_cache:
        segments = transcript_cache[video_id]
    else:
        segments = get_segments_for_video(video_id)
        transcript_cache[video_id] = segments
    
    if not segments:
        return []
    
    # Prepare texts for vectorization
    texts = [segment.text for segment in segments]
    texts.append(query)  # Add query to get its vector
    
    try:
        # Compute TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate similarity between query and all segments
        query_vector = tfidf_matrix[-1]  # Last item is the query
        segment_vectors = tfidf_matrix[:-1]  # All but last are segments
        
        similarities = cosine_similarity(query_vector, segment_vectors).flatten()
        
        # Get top segments
        top_indices = np.argsort(similarities)[::-1][:max_segments]
        
        relevant_segments = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                relevant_segments.append(segments[idx])
        
        return relevant_segments
        
    except Exception as e:
        logger.error(f"Error finding relevant segments: {e}")
        return segments[:max_segments]  # Fallback to recent segments

async def stream_openai_response(prompt: str):
    """Stream OpenAI GPT-4o response."""
    if not openai_client:
        yield "data: {\"error\": \"OpenAI client not initialized\"}\n\n"
        return
    
    try:
        stream = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI tutor helping students understand video content. Provide clear, educational responses based on the video transcript context provided."},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            max_tokens=500,
            temperature=0.7
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"OpenAI streaming error: {e}")
        yield f"data: {{\"error\": \"Failed to get AI response: {str(e)}\"}}\n\n"

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    init_database()
    await load_models()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "whisper_loaded": whisper_model is not None,
        "openai_loaded": openai_client is not None
    }

@app.websocket("/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """WebSocket endpoint for audio streaming and transcription."""
    await websocket.accept()
    
    connection_id = f"conn_{int(time.time() * 1000)}"
    active_connections[connection_id] = websocket
    audio_buffers[connection_id] = []
    
    logger.info(f"Audio WebSocket connected: {connection_id}")
    
    last_process_time = time.time()
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            audio_buffers[connection_id].append(data)
            
            current_time = time.time()
            
            # Process buffer every 5 seconds or when it gets large
            buffer_size = sum(len(chunk) for chunk in audio_buffers[connection_id])
            time_elapsed = current_time - last_process_time
            
            if time_elapsed >= 5.0 or buffer_size > 160000:  # ~5 seconds at 16kHz
                # Combine audio chunks
                combined_audio = b''.join(audio_buffers[connection_id])
                
                if len(combined_audio) > 0:
                    # Transcribe audio
                    transcript = transcribe_audio(combined_audio)
                    
                    if transcript.strip():
                        logger.info(f"Transcription: {transcript}")
                        
                        # Send transcription back to client
                        await websocket.send_text(json.dumps({
                            "type": "transcription",
                            "text": transcript,
                            "timestamp": current_time
                        }))
                        
                        # Save to database (simplified - using connection_id as video_id)
                        segment = TranscriptSegment(
                            video_id=connection_id,
                            start_time=last_process_time,
                            end_time=current_time,
                            text=transcript
                        )
                        save_segment(segment)
                
                # Clear buffer and reset timer
                audio_buffers[connection_id] = []
                last_process_time = current_time
                
    except WebSocketDisconnect:
        logger.info(f"Audio WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Audio WebSocket error: {e}")
    finally:
        # Cleanup
        if connection_id in active_connections:
            del active_connections[connection_id]
        if connection_id in audio_buffers:
            del audio_buffers[connection_id]

@app.post("/query")
async def handle_query(request: QueryRequest):
    """Handle query requests with RAG and streaming response."""
    try:
        # Find relevant transcript segments
        relevant_segments = find_relevant_segments(request.videoId, request.prompt)
        
        # Build context from relevant segments
        context_parts = []
        for segment in relevant_segments:
            context_parts.append(f"[{segment.start_time:.1f}s] {segment.text}")
        
        context = "\n".join(context_parts) if context_parts else "No relevant transcript context found."
        
        # Build enhanced prompt with context
        enhanced_prompt = f"""Based on the following video transcript context, please answer the user's question:

VIDEO TRANSCRIPT CONTEXT:
{context}

USER QUESTION: {request.prompt}

Please provide a helpful response based on the video content. If the transcript context doesn't contain enough information, acknowledge this and provide general educational guidance."""
        
        # Return streaming response
        return StreamingResponse(
            stream_openai_response(enhanced_prompt),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
        
    except Exception as e:
        logger.error(f"Query handling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcript/{video_id}")
async def get_transcript(video_id: str):
    """Get full transcript for a video."""
    segments = get_segments_for_video(video_id)
    return {
        "video_id": video_id,
        "segments": [segment.dict() for segment in segments]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")