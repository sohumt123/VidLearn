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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

whisper_model = None
openai_client = None
active_connections: Dict[str, WebSocket] = {}
audio_buffers: Dict[str, List[bytes]] = {}
transcript_cache: Dict[str, List[TranscriptSegment]] = {}
conversation_memory: Dict[str, List[Dict]] = {}
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

load_dotenv()

RAG_CONFIG = {
    "max_context_tokens": 15000,
    "max_segments": 10,
    "similarity_threshold": 0.05,
}

def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)

def save_conversation(video_id: str, user_msg: str, ai_msg: str):
    if not video_id or not user_msg or not ai_msg:
        return
        
    if video_id not in conversation_memory:
        conversation_memory[video_id] = []
    
    conversation_memory[video_id].append({
        "user": user_msg.strip(),
        "assistant": ai_msg.strip(),
        "timestamp": time.time()
    })
    
    if len(conversation_memory[video_id]) > 5:
        conversation_memory[video_id] = conversation_memory[video_id][-5:]

def get_chat_history(video_id: str) -> str:
    if video_id not in conversation_memory or not conversation_memory[video_id]:
        return ""
    
    history = []
    for chat in conversation_memory[video_id][-3:]:
        history.append(f"Human: {chat['user']}")
        
        response = chat['assistant']
        if len(response) > 250:
            sentences = response.split('. ')
            if len(sentences) > 1 and len(sentences[0]) < 200:
                response = sentences[0] + '. ' + sentences[1] + "..."
            else:
                response = sentences[0] + "..."
        
        history.append(f"Assistant: {response}")
    
    return "\n".join(history)


def init_db():
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

def get_db():
    return sqlite3.connect('transcripts.db')

def save_segment(segment: TranscriptSegment):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO segments (video_id, start_time, end_time, text)
        VALUES (?, ?, ?, ?)
    ''', (segment.video_id, segment.start_time, segment.end_time, segment.text))
    segment.id = cursor.lastrowid
    conn.commit()
    conn.close()

def get_video_segments(video_id: str) -> List[TranscriptSegment]:
    conn = get_db()
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
            id=row[0], video_id=row[1], start_time=row[2], 
            end_time=row[3], text=row[4]
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
    """Transcribe audio using OpenAI Whisper API."""
    if not openai_client:
        return ""
    
    try:
        # Convert PCM bytes to WAV format for OpenAI API
        wav_data = pcm_to_wav(audio_data)
        
        # Create a temporary file-like object
        audio_file = io.BytesIO(wav_data)
        audio_file.name = "audio.wav"  # Required for OpenAI API
        
        # Use OpenAI Whisper API
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en",
            response_format="text"
        )
        
        result = transcript.strip() if transcript else ""
        logger.info(f"🎤 OpenAI Whisper transcribed {len(audio_data)} bytes -> '{result}'")
        return result
    
    except Exception as e:
        logger.error(f"OpenAI Whisper transcription error: {e}")
        # Fallback to local whisper if available
        return transcribe_audio_local(audio_data)

def transcribe_audio_local(audio_data: bytes) -> str:
    """Fallback: Transcribe audio using local Whisper model."""
    if not whisper_model:
        return ""
    
    try:
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe with Whisper with optimized settings
        segments, info = whisper_model.transcribe(
            audio_np, 
            beam_size=5,
            word_timestamps=True,  # Get word-level timestamps
            language="en"  # Force English for better performance
        )
        
        # Combine all segments into single text
        text_parts = []
        for segment in segments:
            if segment.text.strip():  # Only add non-empty segments
                text_parts.append(segment.text.strip())
        
        result = " ".join(text_parts)
        logger.info(f"Local Whisper (fallback) transcribed {len(audio_np)/16000:.2f}s audio -> '{result}'")
        return result
    
    except Exception as e:
        logger.error(f"Local Whisper transcription error: {e}")
        return ""

def find_relevant_segments(video_id: str, query: str, max_segments: int = None) -> List[TranscriptSegment]:
    """Advanced RAG-based segment retrieval with token awareness."""
    if max_segments is None:
        max_segments = RAG_CONFIG["max_segments"]
    
    # Get segments from cache or database
    if video_id in transcript_cache:
        segments = transcript_cache[video_id]
    else:
        segments = get_video_segments(video_id)
        transcript_cache[video_id] = segments
    
    if not segments:
        logger.warning(f"RAG: No segments found for video_id: {video_id}")
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
        
# Removed recency weighting - keep it simple
        
        # Get candidate segments above threshold
        candidate_indices = []
        for idx, score in enumerate(similarities):
            if score >= RAG_CONFIG["similarity_threshold"]:
                candidate_indices.append((idx, score))
        
        # Sort by score descending
        candidate_indices.sort(key=lambda x: x[1], reverse=True)
        
        if not candidate_indices:
            # Fall back to most recent segments
            recent_count = min(max_segments, len(segments))
            return segments[-recent_count:]
        
        # Smart selection with token awareness
        selected_segments = []
        total_tokens = 0
        
        for idx, score in candidate_indices:
            segment = segments[idx]
            segment_tokens = estimate_tokens(segment.text)
            
            # Check if adding this segment would exceed token limit
            if total_tokens + segment_tokens > RAG_CONFIG["max_context_tokens"]:
                break
                
            selected_segments.append(segment)
            total_tokens += segment_tokens
            
            if len(selected_segments) >= max_segments:
                break
        
        # Sort selected segments by timestamp for coherent reading
        selected_segments.sort(key=lambda x: x.start_time)
        
# Removed complex segment combining - keep segments simple
        
        logger.info(f"RAG: Final selection: {len(selected_segments)} segments, ~{total_tokens} tokens")
        return selected_segments
        
    except Exception as e:
        logger.error(f"RAG: Error in segment selection: {e}")
        # Fallback to recent segments
        recent_count = min(max_segments, len(segments))
        return segments[-recent_count:]

async def stream_openai_response(prompt: str, video_id: str = None, user_message: str = None):
    """Stream OpenAI GPT-4o response with conversation memory."""
    if not openai_client:
        yield "data: {\"error\": \"OpenAI client not initialized\"}\n\n"
        return
    
    try:
        stream = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an AI Video Tutor - a specialized learning companion that provides instant, intelligent tutoring while users watch educational content. 

Core principles:
• Be confident and direct in explanations
• Use simple language for complex concepts  
• Focus on enhancing understanding and retention
• Reference specific video content when available
• Provide actionable learning insights
• Maintain pedagogical focus in all responses

Always structure responses to maximize learning impact."""
                },
                {"role": "user", "content": prompt}
            ],
            stream=True,
            max_tokens=300,  # Reduced for concise responses
            temperature=0.2  # Lower for more focused, consistent responses
        )
        
        # Collect full response for memory storage
        full_response = ""
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"
        
        if video_id and user_message and full_response:
            save_conversation(video_id, user_message, full_response)
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"OpenAI streaming error: {e}")
        yield f"data: {{\"error\": \"Failed to get AI response: {str(e)}\"}}\n\n"

@app.on_event("startup")
async def startup():
    init_db()
    await load_models()

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper": whisper_model is not None,
        "openai": openai_client is not None
    }

@app.websocket("/audio")
async def audio_handler(websocket: WebSocket):
    await websocket.accept()
    
    connection_id = f"conn_{int(time.time() * 1000)}"
    active_connections[connection_id] = websocket
    audio_buffers[connection_id] = []
    
    # Try to get video_id from query params
    video_id = websocket.query_params.get('video_id', connection_id)
    
    logger.info(f"Audio WebSocket connected: {connection_id} for video: {video_id}")
    
    last_process_time = time.time()
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            audio_buffers[connection_id].append(data)
            
            current_time = time.time()
            
            # Process buffer every 1 second or when it gets moderately large
            buffer_size = sum(len(chunk) for chunk in audio_buffers[connection_id])
            time_elapsed = current_time - last_process_time
            
            if time_elapsed >= 1.0 or buffer_size > 32000:  # ~1 second at 16kHz
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
                        
                        # Save to database using actual video_id
                        segment = TranscriptSegment(
                            video_id=video_id,
                            start_time=last_process_time,
                            end_time=current_time,
                            text=transcript
                        )
                        logger.info(f"Saving segment for video_id: {video_id} - '{transcript[:50]}...'")
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
async def query(request: QueryRequest):
    try:
        logger.info(f"Query for video {request.videoId}: '{request.prompt}'")
        
        chat_history = get_chat_history(request.videoId)
        
        all_segments = get_video_segments(request.videoId)
        total_transcript_text = " ".join([seg.text for seg in all_segments])
        total_words = len(total_transcript_text.split())
        
        if total_words <= 10000:
            relevant_segments = all_segments
        else:
            relevant_segments = find_relevant_segments(request.videoId, request.prompt)
            if len(relevant_segments) < 3 and len(all_segments) > 0:
                backup_segments = all_segments[-10:]
                relevant_segments.extend(backup_segments)
                relevant_segments = relevant_segments[-10:]
        
        context_parts = []
        total_context_tokens = 0
        
        for segment in relevant_segments:
            segment_text = f"[{segment.start_time:.1f}s] {segment.text}"
            context_parts.append(segment_text)
            total_context_tokens += estimate_tokens(segment_text)
        
        context = "\n".join(context_parts) if context_parts else "No transcript found."
        context_type = "COMPLETE VIDEO TRANSCRIPT" if total_words <= 10000 else "RELEVANT VIDEO SEGMENTS"
        
        prompt = f"""You are an AI tutor helping a user understand a video.

{f"CONVERSATION HISTORY:\n{chat_history}\n" if chat_history else ""}CURRENT QUESTION: {request.prompt}

{context_type} (from the video):
{context}

INSTRUCTIONS:
- Answer the current question using the video transcript as your primary source
{f"- Consider our conversation history above when relevant to the current question" if chat_history else ""}
- Use **bold** for key terms, keep response 2-3 sentences
- Be specific to what's in the video

Answer the question:"""
        
        return StreamingResponse(
            stream_openai_response(prompt, request.videoId, request.prompt),
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

@app.get("/debug/all-transcripts")
async def debug_all_transcripts():
    """Debug endpoint to see all stored transcripts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT video_id, COUNT(*) as segment_count, 
               MIN(start_time) as first_segment, MAX(end_time) as last_segment
        FROM segments 
        GROUP BY video_id
        ORDER BY MAX(created_at) DESC
    ''')
    
    results = []
    for row in cursor.fetchall():
        results.append({
            "video_id": row[0],
            "segment_count": row[1],
            "first_segment": row[2],
            "last_segment": row[3]
        })
    
    conn.close()
    return {"stored_video_ids": results}

# Removed debug RAG config endpoint

@app.get("/debug/conversation-memory")
async def debug_conversation_memory():
    """Debug endpoint to see current conversation memory."""
    return {
        "total_videos_with_conversations": len(conversation_memory),
        "conversations": {
            video_id: {
                "exchange_count": len(exchanges),
                "exchanges": [
                    {
                        "user": exchange["user"][:100] + "..." if len(exchange["user"]) > 100 else exchange["user"],
                        "assistant": exchange["assistant"][:100] + "..." if len(exchange["assistant"]) > 100 else exchange["assistant"],
                        "timestamp": exchange["timestamp"]
                    }
                    for exchange in exchanges[-3:]  # Show last 3
                ]
            }
            for video_id, exchanges in conversation_memory.items()
        }
    }

@app.get("/debug/transcript-segments/{video_id}")
async def debug_transcript_segments(video_id: str):
    """Debug endpoint to see transcript segments for a specific video."""
    segments = get_segments_for_video(video_id)
    return {
        "video_id": video_id,
        "total_segments": len(segments),
        "segments": [
            {
                "id": seg.id,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text,
                "length": len(seg.text)
            }
            for seg in segments[:20]  # Show first 20
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")