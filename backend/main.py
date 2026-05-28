import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from backend.config import STATIC_FRONTEND_DIR, LLM_MODE
from backend.rag_engine import RAGEngine

app = FastAPI(
    title="IRCTC Cancellation Policy RAG",
    description="An educational Retrieval-Augmented Generation implementation from scratch.",
    version="1.0.0"
)

# Global variables
rag_engine = RAGEngine()
current_llm_mode = LLM_MODE

# Request/Response Models
class ChatRequest(BaseModel):
    query: str

class ChunkResponse(BaseModel):
    doc_id: int
    title: str
    text: str
    similarity_score: float

class TimingResponse(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float

class ChatResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[ChunkResponse]
    raw_prompt: str
    llm_mode: str
    timings: TimingResponse

class ConfigResponse(BaseModel):
    llm_mode: str
    available_modes: List[str]

class ConfigUpdateRequest(BaseModel):
    llm_mode: str

@app.on_event("startup")
def startup_event():
    """Initializes the RAG Engine on startup."""
    print("FastAPI server starting...")
    try:
        rag_engine.initialize()
    except Exception as e:
        print(f"CRITICAL ERROR initializing RAG Engine: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Processes user query, performs vector similarity search, and generates answer."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    if not rag_engine.is_initialized:
        # Fallback initialization in case it failed on startup
        try:
            rag_engine.initialize()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG Engine not initialized: {str(e)}")
            
    start_total = time.time()
    
    # 1. Retrieve top-K relevant chunks
    try:
        retrieved_chunks, retrieval_time = rag_engine.retrieve(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in vector search: {str(e)}")
        
    # 2. Generate answer
    try:
        answer, raw_prompt, generation_time = rag_engine.generate_answer(
            request.query, 
            retrieved_chunks,
            mode=current_llm_mode
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in text generation: {str(e)}")
        
    total_time = time.time() - start_total
    
    return ChatResponse(
        query=request.query,
        answer=answer,
        retrieved_chunks=[
            ChunkResponse(
                doc_id=c["doc_id"],
                title=c["title"],
                text=c["text"],
                similarity_score=c["similarity_score"]
            ) for c in retrieved_chunks
        ],
        raw_prompt=raw_prompt,
        llm_mode=current_llm_mode,
        timings=TimingResponse(
            retrieval_ms=round(retrieval_time * 1000, 1),
            generation_ms=round(generation_time * 1000, 1),
            total_ms=round(total_time * 1000, 1)
        )
    )

@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Gets the current LLM mode and list of available modes."""
    return ConfigResponse(
        llm_mode=current_llm_mode,
        available_modes=["MOCK", "LOCAL", "CLOUD_GEMINI"]
    )

@app.post("/api/config", response_model=ConfigResponse)
async def update_config(request: ConfigUpdateRequest):
    """Updates the active LLM mode dynamically."""
    global current_llm_mode
    mode = request.llm_mode.upper()
    if mode not in ["MOCK", "LOCAL", "CLOUD_GEMINI"]:
        raise HTTPException(status_code=400, detail="Invalid LLM mode. Select from: MOCK, LOCAL, CLOUD_GEMINI.")
        
    current_llm_mode = mode
    print(f"LLM mode updated dynamically to: {current_llm_mode}")
    return ConfigResponse(
        llm_mode=current_llm_mode,
        available_modes=["MOCK", "LOCAL", "CLOUD_GEMINI"]
    )

# Static file serving
# ==========================================

# Catch-all route to serve index.html for UI
@app.get("/")
async def get_index():
    index_path = os.path.join(STATIC_FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail=f"Frontend index.html not found at {index_path}")

# Mount remaining static frontend files (style.css, app.js)
if os.path.exists(STATIC_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=STATIC_FRONTEND_DIR), name="static")
