import faiss
import numpy as np
from langchain_core.documents import Document
from .embeddings import EmbeddingManager

class VectorStore:
    def __init__(self, dim: int, index: faiss.IndexFlatIP, docs: list[Document]):
        self.dim = dim
        self.index = index
        self.docs = docs

    @classmethod
    def from_documents(cls, documents: list[Document], embedder: EmbeddingManager) -> "VectorStore":
        texts = [d.page_content for d in documents]
        embeddings = embedder.embed_texts(texts).astype("float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(dim=embeddings.shape[1], index=index, docs=documents)

    def search(self, query_vector: np.ndarray, k: int = 5):
        q = query_vector.astype("float32")[None, :]
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        return scores[0], indices[0]
