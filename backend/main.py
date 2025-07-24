import asyncio
import io
import json
import sqlite3
import time
import wave
from typing import Dict, List, Optional
import logging
import re

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
conversation_memory: Dict[str, List[Dict]] = {}  # Store conversation history by video_id
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

load_dotenv()

# RAG Configuration
RAG_CONFIG = {
    "max_context_tokens": 15000,  # ~12K words for context
    "max_segments": 10,           # Maximum segments to consider
    "similarity_threshold": 0.05,  # Minimum similarity score
    "recency_weight": 0.1,        # Boost for recent segments
    "adjacency_bonus": 0.05,      # Bonus for adjacent segments
}

# Tutoring Response Configuration
TUTORING_CONFIG = {
    "explanation_keywords": ["what", "how", "why", "explain", "meaning", "definition"],
    "summary_keywords": ["summarize", "summary", "about", "overview", "main points"],
    "clarification_keywords": ["clarify", "confused", "understand", "clear"],
    "application_keywords": ["example", "use", "apply", "practical", "real-world"],
}

def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~1.3 tokens per word for English"""
    word_count = len(text.split())
    return int(word_count * 1.3)

def detect_question_type(question: str) -> str:
    """Detect the type of question to provide more targeted responses."""
    question_lower = question.lower()
    
    # Check for different question types
    if any(keyword in question_lower for keyword in TUTORING_CONFIG["explanation_keywords"]):
        return "explanation"
    elif any(keyword in question_lower for keyword in TUTORING_CONFIG["summary_keywords"]):
        return "summary"
    elif any(keyword in question_lower for keyword in TUTORING_CONFIG["clarification_keywords"]):
        return "clarification"
    elif any(keyword in question_lower for keyword in TUTORING_CONFIG["application_keywords"]):
        return "application"
    else:
        return "general"

def get_response_focus(question_type: str) -> str:
    """Get specific response focus based on question type."""
    focus_map = {
        "explanation": "Provide a clear, step-by-step explanation focusing on the underlying concepts and mechanisms.",
        "summary": "Create a concise but comprehensive summary highlighting the key points and main takeaways.",
        "clarification": "Address the confusion directly with a clear, simplified explanation that removes ambiguity.",
        "application": "Focus on practical examples and real-world applications to make the concept tangible.",
        "general": "Provide a well-rounded response that addresses the question comprehensively."
    }
    return focus_map.get(question_type, focus_map["general"])

def add_to_conversation_memory(video_id: str, user_message: str, ai_response: str):
    """Add a conversation exchange to memory."""
    if video_id not in conversation_memory:
        conversation_memory[video_id] = []
    
    conversation_memory[video_id].append({
        "user": user_message,
        "assistant": ai_response,
        "timestamp": time.time()
    })
    
    # Keep only last 5 exchanges to manage memory
    if len(conversation_memory[video_id]) > 5:
        conversation_memory[video_id] = conversation_memory[video_id][-5:]

def get_conversation_context(video_id: str) -> str:
    """Get recent conversation history for context."""
    if video_id not in conversation_memory or not conversation_memory[video_id]:
        return ""
    
    context_parts = []
    for exchange in conversation_memory[video_id][-2:]:  # Last 2 exchanges for brevity
        context_parts.append(f"Q: {exchange['user']}")
        context_parts.append(f"A: {exchange['assistant'][:150]}...")  # Shorter truncation
    
    return "\n".join(context_parts) if context_parts else ""

def is_follow_up_question(question: str) -> bool:
    """Detect if this is a follow-up question."""
    follow_up_indicators = [
        "what about", "but what", "also", "and", "more about", "explain more",
        "what else", "another", "further", "additionally", "however", "though",
        "can you", "could you", "tell me more", "elaborate", "expand"
    ]
    question_lower = question.lower()
    return any(indicator in question_lower for indicator in follow_up_indicators)

def combine_adjacent_segments(segments: List[TranscriptSegment], similarities: np.ndarray) -> List[TranscriptSegment]:
    """Combine adjacent segments to reduce fragmentation and improve context"""
    if len(segments) <= 1:
        return segments
    
    combined = []
    i = 0
    
    while i < len(segments):
        current_segment = segments[i]
        combined_text = current_segment.text
        end_time = current_segment.end_time
        
        # Look ahead for adjacent segments
        j = i + 1
        while (j < len(segments) and 
               segments[j].start_time - segments[j-1].end_time < 2.0 and  # Within 2 seconds
               estimate_tokens(combined_text + " " + segments[j].text) < 500):  # Don't make chunks too large
            combined_text += " " + segments[j].text
            end_time = segments[j].end_time
            j += 1
        
        # Create combined segment
        combined_segment = TranscriptSegment(
            id=current_segment.id,
            video_id=current_segment.video_id,
            start_time=current_segment.start_time,
            end_time=end_time,
            text=combined_text
        )
        combined.append(combined_segment)
        i = j
    
    return combined

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
        logger.info(f"Whisper transcribed {len(audio_np)/16000:.2f}s audio -> '{result}'")
        return result
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

def find_relevant_segments(video_id: str, query: str, max_segments: int = None) -> List[TranscriptSegment]:
    """Advanced RAG-based segment retrieval with token awareness."""
    if max_segments is None:
        max_segments = RAG_CONFIG["max_segments"]
    
    logger.info(f"RAG: Looking for segments with video_id: {video_id}")
    
    # Get segments from cache or database
    if video_id in transcript_cache:
        segments = transcript_cache[video_id]
        logger.info(f"RAG: Found {len(segments)} segments in cache")
    else:
        segments = get_segments_for_video(video_id)
        transcript_cache[video_id] = segments
        logger.info(f"RAG: Found {len(segments)} segments in database")
    
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
        
        # Apply recency weighting (boost recent segments)
        if len(segments) > 1:
            recency_scores = np.linspace(0, RAG_CONFIG["recency_weight"], len(segments))
            similarities += recency_scores
        
        logger.info(f"RAG: Similarity scores - max={similarities.max():.3f}, min={similarities.min():.3f}")
        
        # Get candidate segments above threshold
        candidate_indices = []
        for idx, score in enumerate(similarities):
            if score >= RAG_CONFIG["similarity_threshold"]:
                candidate_indices.append((idx, score))
        
        # Sort by score descending
        candidate_indices.sort(key=lambda x: x[1], reverse=True)
        
        if not candidate_indices:
            logger.warning("RAG: No segments met similarity threshold, using recent segments")
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
                logger.info(f"RAG: Token limit reached ({total_tokens + segment_tokens} > {RAG_CONFIG['max_context_tokens']})")
                break
                
            selected_segments.append(segment)
            total_tokens += segment_tokens
            
            logger.info(f"RAG: Selected segment {idx} (score={score:.3f}, tokens={segment_tokens}): '{segment.text[:50]}...'")
            
            if len(selected_segments) >= max_segments:
                break
        
        # Sort selected segments by timestamp for coherent reading
        selected_segments.sort(key=lambda x: x.start_time)
        
        # Combine adjacent segments to reduce fragmentation
        if len(selected_segments) > 1:
            combined_segments = combine_adjacent_segments(selected_segments, similarities)
            logger.info(f"RAG: Combined {len(selected_segments)} segments into {len(combined_segments)} chunks")
            selected_segments = combined_segments
        
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
        
        # Store conversation in memory after streaming completes
        if video_id and user_message and full_response:
            add_to_conversation_memory(video_id, user_message, full_response)
            logger.info(f"Stored conversation exchange for video {video_id}: Q='{user_message[:50]}...' A='{full_response[:50]}...'")
            logger.info(f"Total conversations for {video_id}: {len(conversation_memory.get(video_id, []))}")
        
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
async def handle_query(request: QueryRequest):
    """Handle query requests with RAG and streaming response."""
    try:
        logger.info(f"Query request for videoId: {request.videoId}")
        
        # Get conversation context for follow-up questions
        conversation_context = get_conversation_context(request.videoId)
        is_follow_up = is_follow_up_question(request.prompt)
        
        # Detect question type for targeted response
        question_type = detect_question_type(request.prompt)
        response_focus = get_response_focus(question_type)
        logger.info(f"Query for video {request.videoId}: Question type: {question_type}, Follow-up: {is_follow_up}")
        logger.info(f"Conversation context length: {len(conversation_context)} chars")
        if conversation_context:
            logger.info(f"Context preview: {conversation_context[:200]}...")
        
        # Find relevant transcript segments using RAG
        relevant_segments = find_relevant_segments(request.videoId, request.prompt)
        
        logger.info(f"RAG: Selected {len(relevant_segments)} segments for query")
        
        # Build context from relevant segments with timestamps
        context_parts = []
        total_context_tokens = 0
        
        for segment in relevant_segments:
            segment_text = f"[{segment.start_time:.1f}s] {segment.text}"
            context_parts.append(segment_text)
            total_context_tokens += estimate_tokens(segment_text)
        
        context = "\n".join(context_parts) if context_parts else "No relevant transcript context found."
        logger.info(f"RAG: Context built with ~{total_context_tokens} tokens")
        
        # Build concise enhanced prompt
        enhanced_prompt = f"""You are an AI Video Tutor. Provide CONCISE, TO-THE-POINT explanations.

RESPONSE RULES:
- Keep answers SHORT and DIRECT (2-3 sentences max)
- Use simple language, avoid jargon
- Focus on the essential concept only
- Use markdown formatting (**bold**, `code`, ### headers, - bullets)

{f"CONTEXT FROM PREVIOUS CHAT:\n{conversation_context}\n" if conversation_context else ""}

VIDEO TRANSCRIPT:
{context}

QUESTION: {request.prompt}

{"This is a FOLLOW-UP question - reference previous context briefly." if is_follow_up else ""}

Give a concise, formatted answer:"""
        
        # Return streaming response with conversation memory
        return StreamingResponse(
            stream_openai_response(enhanced_prompt, request.videoId, request.prompt),
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

@app.get("/debug/rag-config")
async def debug_rag_config():
    """Debug endpoint to see current RAG configuration."""
    return {
        "rag_config": RAG_CONFIG,
        "vectorizer_features": vectorizer.max_features,
        "cache_info": {
            "cached_videos": list(transcript_cache.keys()),
            "cache_sizes": {vid: len(segments) for vid, segments in transcript_cache.items()}
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")