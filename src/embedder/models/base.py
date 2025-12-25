from enum import Enum
from typing import TypeVar, Generic

from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel

EmbeddingModelType = SparseTextEmbedding | SentenceTransformer | BGEM3FlagModel
class EmbeddingModelEnum(Enum):
    SPARSE_TEXT_EMBEDDING = SparseTextEmbedding
    SENTENCE_TRANSFORMER = SentenceTransformer
    BGEM3_FLAG_MODEL = BGEM3FlagModel

T = TypeVar('T')
# M = TypeVar('M', bound=EmbeddingModelType)

class BaseModel(Generic[T]):

    def encode(self, data: T) -> list[float]:
        raise NotImplementedError