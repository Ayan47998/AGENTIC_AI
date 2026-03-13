"""FastAPI server for the PDF RAG agent."""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config.config import UPLOAD_DIR
from app.ingestion.pdf_processor import extract_text_from_pdf
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_documents
from app.retrieval.vector_store import vector_store
from app.agent.graph import rag_agent

app = FastAPI(title="PDF RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory chat history per session (simple single-user setup)
chat_history: list[dict[str, str]] = []


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]


# ── Endpoints ───────────────────────────────────────────────────


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, process it, and add to the vector store."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract → chunk → embed → store
    pages = extract_text_from_pdf(file_path)
    if not pages:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    documents = chunk_text(pages, source=file.filename)
    embeddings = embed_documents(documents)
    vector_store.add_documents(documents, embeddings)

    return {
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(documents),
        "total_chunks_in_store": vector_store.total_documents,
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    """Ask a question about uploaded documents."""
    if vector_store.is_empty:
        return AnswerResponse(
            answer="No documents have been uploaded yet. Please upload a PDF first.",
            sources=[],
        )

    state = {
        "question": req.question,
        "standalone_question": "",
        "chat_history": chat_history,
        "documents": [],
        "answer": "",
        "rewrite_count": 0,
        "is_valid_query": True,
    }

    result = rag_agent.invoke(state)

    # Update chat history
    chat_history.append({"role": "user", "content": req.question})
    chat_history.append({"role": "assistant", "content": result["answer"]})

    # Keep history bounded
    if len(chat_history) > 20:
        chat_history[:] = chat_history[-20:]

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": doc.metadata.get("score", 0),
        }
        for doc in result.get("documents", [])
    ]

    return AnswerResponse(answer=result["answer"], sources=sources)


@app.post("/reset")
async def reset():
    """Clear the vector store and chat history."""
    vector_store.reset()
    chat_history.clear()
    return {"status": "reset"}


@app.get("/status")
async def status():
    """Return current system status."""
    return {
        "documents_loaded": vector_store.total_documents,
        "history_length": len(chat_history),
    }
