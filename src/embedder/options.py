from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any

from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from .qtypes import DataItemRaw


DataObjVectorFn = Callable[[DataItemRaw], str] | None

class DenseOptionDataTypeEnum(str, Enum):
    IMAGE = "image"
    TEXT = "text"

@dataclass
class CommonOption:
    key: str
    batch: bool
    data_obj_vector_fn: DataObjVectorFn
    data_obj_vector_key: str

@dataclass
class DenseOption(CommonOption):
    model: SentenceTransformer
    normalize_embeddings: bool
    type: DenseOptionDataTypeEnum = DenseOptionDataTypeEnum.IMAGE
    max_workers: int = 10

@dataclass
class SparseOption(CommonOption):
    model: SparseTextEmbedding

EmbedderOptionType = DenseOption | SparseOption
