import os

# RAG Configuration Settings
# ==========================================

# LLM Generation Mode
# Options: 
#   - "MOCK"         : Instant, structured generator using the retrieved context. (Zero setup, runs instantly)
#   - "LOCAL"        : Run Qwen/Qwen2.5-0.5B-Instruct completely offline on your CPU. (Requires torch & transformers, downloads ~1GB weights)
#   - "CLOUD_GEMINI" : Run high-quality answers using Google Gemini API. (Requires GEMINI_API_KEY environment variable)
LLM_MODE = os.environ.get("RAG_LLM_MODE", "MOCK")

# Local LLM Model Configuration
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Sentence Transformer model for local embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# RAG Hyperparameters
TOP_K = 3               # Number of context chunks to retrieve
SIMILARITY_THRESHOLD = 0.1  # Minimum cosine similarity score (0.0 to 1.0) to consider a chunk relevant

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "data", "knowledge_base.txt")
STATIC_FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
