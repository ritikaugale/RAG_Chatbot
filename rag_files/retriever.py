from dataclasses import dataclass
from typing import Any
from langchain_core.documents import Document
from .embeddings import EmbeddingManager
from .vector_store import VectorStore

@dataclass
class RetrievedDocument:
    content: str
    metadata: dict[str, Any]
    similarity_score: float
    distance: float
    rank: int

class RAGRetriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingManager, score_threshold: float = 0.0):
        self.store = store
        self.embedder = embedder
        self.score_threshold = score_threshold

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        q_vec = self.embedder.embed_query(query)
        scores, indices = self.store.search(q_vec, k=top_k)
        results: list[RetrievedDocument] = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0:
                continue
            doc: Document = self.store.docs[idx]
            if score < self.score_threshold:
                continue
            results.append(
                RetrievedDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    similarity_score=float(score),
                    distance=float(1 - score),
                    rank=rank,
                )
            )
        return results
