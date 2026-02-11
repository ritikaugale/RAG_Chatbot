from sentence_transformers import SentenceTransformer
import numpy as np
from .config import EMBEDDING_MODEL_NAME
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EmbeddingManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], show_progress_bar=False)[0]
