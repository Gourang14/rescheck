from typing import List
import numpy as np

# Try to import sentence-transformers (local fallback)
try:
    from sentence_transformers import SentenceTransformer
    _has_sbert = True
except ImportError:
    _has_sbert = False


class EmbeddingsProvider:
    """Abstract base provider interface"""
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        raise NotImplementedError


class SBERTEmbeddingsProvider(EmbeddingsProvider):
    """
    Local embeddings using sentence-transformers (all-MiniLM-L6-v2).
    Works offline if the model is installed.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not _has_sbert:
            raise ImportError(
                "sentence-transformers not installed. Please run: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()


class OpenAIEmbeddingsProvider(EmbeddingsProvider):
    """
    Placeholder for OpenAI embeddings API.
    (Replace with real openai.Embedding.create call if needed.)
    """
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        # For now, just return random vectors as placeholder
        return np.random.rand(len(texts), 384).tolist()