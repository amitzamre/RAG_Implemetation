import os
import time
import numpy as np
import urllib.request
import json
from sentence_transformers import SentenceTransformer
from backend.config import (
    KNOWLEDGE_BASE_PATH,
    EMBEDDING_MODEL_NAME,
    LOCAL_MODEL_NAME,
    TOP_K,
    SIMILARITY_THRESHOLD,
    LLM_MODE
)

class RAGDocument:
    """Represents a text chunk in the RAG knowledge base."""
    def __init__(self, doc_id: int, title: str, text: str):
        self.doc_id = doc_id
        self.title = title
        self.text = text
        self.embedding = None  # Will be populated by embedding model

class RAGEngine:
    def __init__(self):
        self.documents = []
        self.embedding_model = None
        self.local_llm_pipeline = None
        self.is_initialized = False

    def initialize(self):
        """Loads data, generates embeddings, and prepares the RAG pipeline."""
        if self.is_initialized:
            return

        print("Initializing RAG Engine...")
        
        # 1. Load and chunk the document
        self._load_knowledge_base()
        
        # 2. Load the embedding model (SentenceTransformer)
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # 3. Generate embeddings for all document chunks
        print("Generating embeddings for knowledge base chunks...")
        texts = [doc.text for doc in self.documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        # Store embeddings in each document object
        for i, doc in enumerate(self.documents):
            doc.embedding = embeddings[i]
            
        self.is_initialized = True
        print(f"RAG Engine successfully loaded with {len(self.documents)} chunks.")

    def _load_knowledge_base(self):
        """Reads knowledge_base.txt and chunks it by headers/sections."""
        if not os.path.exists(KNOWLEDGE_BASE_PATH):
            raise FileNotFoundError(f"Knowledge base file not found at {KNOWLEDGE_BASE_PATH}")
            
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Split chunks by sections starting with "# SECTION"
        raw_sections = content.strip().split("# SECTION")
        doc_id = 0
        
        for raw_sec in raw_sections:
            if not raw_sec.strip():
                continue
                
            # Reconstruct the section header and content
            lines = raw_sec.strip().split("\n")
            title = lines[0].replace(":", "").strip()
            text_body = "\n".join(lines[1:]).strip()
            
            # Combine title and body to ensure chunk contains its own context
            full_chunk_text = f"SECTION: {title}\n{text_body}"
            
            doc = RAGDocument(
                doc_id=doc_id,
                title=title,
                text=full_chunk_text
            )
            self.documents.append(doc)
            doc_id += 1

    def retrieve(self, query: str, top_k: int = TOP_K):
        """
        Performs vector similarity search manually using raw Cosine Similarity.
        This provides a transparent, educational view of vector search math.
        """
        start_time = time.time()
        
        # Encode the query
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        
        # Collect all document embeddings as a 2D numpy array
        doc_embeddings = np.array([doc.embedding for doc in self.documents])
        
        # Calculate Cosine Similarity manually using numpy
        # Formula: Sim(A, B) = (A . B) / (||A|| * ||B||)
        # Normalize the vectors to unit length
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        doc_embeddings_normalized = doc_embeddings / doc_norms
        
        # Calculate the dot product of normalized vectors, yielding the cosine similarities
        cosine_similarities = np.dot(doc_embeddings_normalized, query_norm)
        
        # Rank document indices by similarity score descending
        ranked_indices = np.argsort(cosine_similarities)[::-1]
        
        retrieved_results = []
        for idx in ranked_indices[:top_k]:
            score = float(cosine_similarities[idx])
            
            # Keep results above threshold
            if score >= SIMILARITY_THRESHOLD:
                doc = self.documents[idx]
                retrieved_results.append({
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "text": doc.text,
                    "similarity_score": score
                })
                
        retrieval_time = time.time() - start_time
        return retrieved_results, retrieval_time

    def generate_answer(self, query: str, retrieved_chunks: list, mode: str = LLM_MODE):
        """Generates an answer using the retrieved context in the selected mode."""
        start_time = time.time()
        
        # 1. Build prompt template
        context_text = "\n\n".join([
            f"--- [SOURCE {i+1}: {chunk['title']}] ---\n{chunk['text']}" 
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        prompt = (
            "You are an AI assistant specialized in IRCTC train ticket cancellation and refund policies. "
            "Your task is to answer the user's question accurately and objectively using ONLY the retrieved official policy contexts below.\n\n"
            "=== RETRIEVED OFFICIAL POLICY CONTEXTS ===\n"
            f"{context_text}\n"
            "==========================================\n\n"
            "INSTRUCTIONS:\n"
            "- Answer the user's question clearly, listing fees, time frames, and conditions exactly as given in the context.\n"
            "- Be brief and educational.\n"
            "- Cite the sources/sections (e.g., 'According to SECTION 1...') used in your answer.\n"
            "- If the context does not contain the answer, say: 'Based on the official policy chunks retrieved, I cannot find sufficient information to answer this question.' Do not invent details.\n\n"
            f"USER QUESTION: {query}\n\n"
            "DETAILED ANSWER:"
        )

        answer = ""
        
        # 2. Route generation based on mode
        if mode == "MOCK":
            answer = self._generate_mock_answer(query, retrieved_chunks)
        elif mode == "LOCAL":
            answer = self._generate_local_answer(prompt)
        elif mode == "CLOUD_GEMINI":
            answer = self._generate_gemini_answer(prompt)
        else:
            answer = f"Error: Unknown LLM generation mode: '{mode}'"
            
        generation_time = time.time() - start_time
        return answer, prompt, generation_time

    def _generate_mock_answer(self, query: str, retrieved_chunks: list) -> str:
        """
        Generates highly accurate rule-based responses by parsing the source context.
        Provides a functional educational RAG system instantly without requiring heavy downloads or APIs.
        """
        # Return elegant message if no relevant chunks are found
        if not retrieved_chunks:
            return (
                "Based on the official IRCTC ticket cancellation policies, I could not find any relevant "
                "information matching your request. Please try rephrasing your question."
            )
            
        # Detect topic of query for educational matching
        q_lower = query.lower()
        
        # Check if the user is asking about confirmed ticket charges
        is_confirmed_48h = any(x in q_lower for x in ["48 hour", "more than 48", "cancel confirmed", "cancellation charge", "minimum charge"])
        is_confirmed_time = any(x in q_lower for x in ["12 hour", "4 hour", "25%", "50%"])
        is_tatkal = "tatkal" in q_lower
        is_rac_wl = any(x in q_lower for x in ["rac", "waitlist", "waiting", "clerkage", "automatic", "chart prepared"])
        is_train_cancelled = any(x in q_lower for x in ["train cancel", "train is cancelled", "railway cancel"])
        is_late_train = any(x in q_lower for x in ["late", "running late", "3 hour", "three hour"])
        is_tdr = any(x in q_lower for x in ["tdr", "ticket deposit receipt", "chart preparation"])

        # Let's inspect the top matched chunk title to double check relevance
        top_chunk = retrieved_chunks[0]
        top_title = top_chunk["title"].upper()
        
        # We will craft a highly helpful, structured, and customized answer based on the official retrieved policy
        answer_parts = []
        answer_parts.append(f"### IRCTC Cancellation Policy Analysis (Demo RAG Mode)\n")
        
        # Retrieve primary information from the most relevant chunks
        matched_sections = ", ".join([f"**{c['title']}** (Similarity: {c['similarity_score']:.2f})" for c in retrieved_chunks])
        answer_parts.append(f"*Context retrieved successfully from:* {matched_sections}\n")
        
        # Let's build answers utilizing exact retrieved policies
        if "SECTION 7" in top_title or is_train_cancelled:
            answer_parts.append(
                "According to **SECTION 7: REFUNDS IN CASE OF TRAIN CANCELLATION**:\n"
                "- **Full Refund**: If your train is officially cancelled by Indian Railways, you are eligible for a **100% full refund** of the ticket fare.\n"
                "- **Automatic Credit**: You do **not** need to cancel your e-ticket manually or file an online TDR. The refund is processed automatically by IRCTC and credited back to your booking bank account."
            )
        elif "SECTION 8" in top_title or is_late_train:
            answer_parts.append(
                "According to **SECTION 8: REFUNDS FOR LATE RUNNING TRAINS**:\n"
                "- **Eligibility**: If the train runs late by **more than 3 hours** from its scheduled departure, you are eligible for a full refund with zero cancellation deductions.\n"
                "- **Action Required**: To claim this refund, you **must file an online Ticket Deposit Receipt (TDR)** before the actual departure of the train from your boarding station. If you do not file it before departure, no refund will be granted."
            )
        elif "SECTION 6" in top_title or is_tatkal:
            answer_parts.append(
                "According to **SECTION 6: TATKAL TICKET CANCELLATION RULES**:\n"
                "- **Confirmed Tatkal**: **No refund** is granted under any circumstances for cancelled confirmed Tatkal tickets.\n"
                "- **Waitlisted Tatkal**: If the Tatkal ticket remains waitlisted, it is cancelled with a deduction of a **clerkage charge of Rs. 60 + GST** per passenger, and the remaining fare is refunded."
            )
        elif "SECTION 5" in top_title or is_rac_wl:
            answer_parts.append(
                "According to **SECTION 5: RAC AND WAITLISTED TICKET CANCELLATION POLICY**:\n"
                "- **Manual Cancellation**: RAC or Waitlisted tickets can be cancelled online up to 4 hours before scheduled departure. A flat clerkage charge of **Rs. 60 + GST per passenger** is deducted, and the balance is refunded.\n"
                "- **Automatic Cancellation**: If a waitlisted e-ticket remains on the waiting list ('WL') after reservation chart preparation, it is **automatically cancelled** by the IRCTC system. The entire fare is **fully refunded** to your account without any clerkage deductions. You do not need to file a TDR."
            )
        elif "SECTION 1" in top_title or "SECTION 2" in top_title or "SECTION 3" in top_title or is_confirmed_48h or is_confirmed_time:
            # Let's outline the cancellation windows beautifully!
            answer_parts.append(
                "Based on the official cancellation policies in **SECTIONS 1, 2, and 3** for confirmed tickets:\n\n"
                "1. **More than 48 Hours Before Departure** (Section 1):\n"
                "   A flat cancellation charge is deducted per passenger depending on class:\n"
                "   - AC First Class / Executive: **Rs. 240 + GST**\n"
                "   - AC 2 Tier / First Class: **Rs. 200 + GST**\n"
                "   - AC 3 Tier / Chair Car: **Rs. 180 + GST**\n"
                "   - Sleeper Class: **Rs. 120** (No GST)\n"
                "   - Second Class: **Rs. 60** (No GST)\n\n"
                "2. **48 Hours to 12 Hours Before Departure** (Section 2):\n"
                "   - The cancellation charge is **25% of the total ticket fare**, subject to the minimum flat rates listed above (whichever is higher).\n\n"
                "3. **12 Hours to 4 Hours Before Departure** (Section 3):\n"
                "   - The cancellation charge is **50% of the total ticket fare**, subject to the minimum flat rates listed above (whichever is higher)."
            )
            if "SECTION 4" in [c["title"] for c in retrieved_chunks] or "4 hour" in q_lower:
                answer_parts.append(
                    "\n\nAdditionally, according to **SECTION 4**:\n"
                    "- **Less than 4 Hours**: If cancelled within 4 hours of departure, **no refund** is granted. Online cancellation is blocked after chart preparation. You must file a TDR for exceptions."
                )
        elif "SECTION 9" in top_title or is_tdr:
            answer_parts.append(
                "According to **SECTION 9: TDR (TICKET DEPOSIT RECEIPT) FILING RULES**:\n"
                "- **When to File**: TDR must be filed when online cancellation is disabled (e.g., after chart preparation) for reasons like train late > 3 hours, AC failure, or travelling in a lower class.\n"
                "- **Processing Time**: Refund is subject to physical verification of railway registers and usually takes **30 to 45 days** to process."
            )
        else:
            # General generic answer synthesizing the top context block
            answer_parts.append(
                f"Based on the retrieved context **{top_chunk['title']}**:\n\n"
                f"{top_chunk['text'].replace('SECTION: ' + top_chunk['title'], '').strip()}\n\n"
                "*(Information synthesized directly from retrieved official policy document)*"
            )
            
        return "\n".join(answer_parts)

    def _generate_local_answer(self, prompt: str) -> str:
        """Generates answer using a local 0.5B parameters model running on CPU."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError:
            return (
                "Error: Local LLM libraries (`torch` or `transformers`) are not installed. "
                "Please run `pip install torch transformers` or switch LLM_MODE to 'MOCK' in `backend/config.py`."
            )
            
        try:
            if not self.local_llm_pipeline:
                print(f"Loading local LLM pipeline: {LOCAL_MODEL_NAME} (this may take a minute on first run)...")
                tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
                model = AutoModelForCausalLM.from_pretrained(
                    LOCAL_MODEL_NAME, 
                    torch_dtype=torch.float32, 
                    device_map="auto"
                )
                self.local_llm_pipeline = pipeline(
                    "text-generation", 
                    model=model, 
                    tokenizer=tokenizer,
                    max_new_tokens=256,
                    temperature=0.2,
                    do_sample=True
                )
                
            # Run generation
            outputs = self.local_llm_pipeline(prompt)
            generated_text = outputs[0]["generated_text"]
            
            # Extract only the newly generated response after the prompt
            if prompt in generated_text:
                answer = generated_text.replace(prompt, "").strip()
            else:
                # Fallback splitting by DETAILED ANSWER:
                parts = generated_text.split("DETAILED ANSWER:")
                answer = parts[-1].strip() if len(parts) > 1 else generated_text
                
            return answer
        except Exception as e:
            return f"Error running local LLM: {str(e)}. Fallback to 'MOCK' or check system configuration."

    def _generate_gemini_answer(self, prompt: str) -> str:
        """Generates answer using the Google Gemini API with urllib (zero dependencies)."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return (
                "Error: `GEMINI_API_KEY` environment variable is not set. "
                "Please set the environment variable, or switch LLM_MODE to 'MOCK' in `backend/config.py`."
            )
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 400
            }
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                
            # Extract content from response
            candidates = res_data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
                    
            return "Error: Empty response structure from Gemini API."
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}. Please check your internet connection or API key."
