from functools import lru_cache
from .data_ingestion import load_all_documents
from .chunking import split_documents
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .retriever import RAGRetriever
from .llm import build_llm, generate_answer
from .config import TOP_K


class RAGPipeline:
    def __init__(self):
        docs = load_all_documents()
        chunks = split_documents(docs)
        self.embedder = EmbeddingManager()
        self.store = VectorStore.from_documents(chunks, self.embedder)
        self.retriever = RAGRetriever(self.store, self.embedder)
        self.llm = build_llm()

    def answer(self, question: str, top_k: int = TOP_K) -> tuple[str, list[str]]:
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        context = "\n\n".join(r.content for r in retrieved)
        answer = generate_answer(self.llm, question, context)
        return answer, [r.content for r in retrieved]


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def answer_question(question: str, top_k: int = TOP_K) -> tuple[str, list[str]]:
    pipeline = get_pipeline()
    return pipeline.answer(question, top_k=top_k)
