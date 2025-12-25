from sentence_transformers import SentenceTransformer

from base import BaseModel, EmbeddingModelType


class StModel(BaseModel[str]):
    """
    For SentenceTransformer model
    """

    def __init__(self, model: SentenceTransformer):
        self.model = model
        pass

    def encode(self, data: str) -> list[float]:
        self.model.encode(data)
        pass