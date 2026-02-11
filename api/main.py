from typing import List
from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from rag_files.pipeline import answer_question, get_pipeline


app = FastAPI(title="RAG Chatbot API", version="0.1.0")

DATA_PDF_DIR = Path("data/pdf_files")
DATA_TXT_DIR = Path("data/text_files")


# ==== Models ====

class CorpusStats(BaseModel):
    num_docs: int
    num_chunks: int
    sources: List[str]


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = 5


class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]


# ==== Endpoints ====

@app.post("/query", response_model=QueryResponse)
async def query_rag(payload: QueryRequest):
    answer, contexts = answer_question(payload.question, top_k=payload.top_k or 5)
    return QueryResponse(answer=answer, contexts=contexts)


@app.post("/upload_docs")
async def upload_docs(files: List[UploadFile] = File(...)):
    DATA_PDF_DIR.mkdir(parents=True, exist_ok=True)
    DATA_TXT_DIR.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []

    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix == ".pdf":
            target = DATA_PDF_DIR / f.filename
        elif suffix == ".txt":
            target = DATA_TXT_DIR / f.filename
        else:
            continue

        with target.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(str(target))

    # force pipeline rebuild on next query
    get_pipeline.cache_clear()

    return {"saved": saved}


@app.get("/stats", response_model=CorpusStats)
async def get_corpus_stats():
    pipeline = get_pipeline()
    retriever = pipeline.retriever
    store = retriever.store

    chunks = store.docs
    num_chunks = len(chunks)
    sources = set()

    for c in chunks:
        meta = getattr(c, "metadata", {}) or {}
        src = meta.get("source") or meta.get("sourcefile")
        if src:
            sources.add(str(src))

    num_docs = len(sources)

    return CorpusStats(
        num_docs=num_docs,
        num_chunks=num_chunks,
        sources=sorted(sources)[:20],
    )
