import logging
from typing import Optional, Any

from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer

from .runners import run_dense, run_sparse
from .qtypes import DataItem
from .options import DenseOption, SparseOption, DataObjVectorFn, EmbedderOptionType, DenseOptionDataTypeEnum

logger = logging.getLogger(__name__)


class QPointBuilder:
    def __init__(self, qclient: QdrantClient | None = None):
        self.qclient = qclient
        self._data: list[DataItem] = []
        self._encode_options: list[EmbedderOptionType] = []

    def data(self, datalist: list[DataItem]) -> "QPointBuilder":
        self._data = datalist
        return self

    def dense(self,
              key: str,
              model: SentenceTransformer | M3Embedder,
              data_type: DenseOptionDataTypeEnum,
              data_obj_vector_key: str = "vector_data",
              data_obj_vector_fn: Optional[DataObjVectorFn] = None,
              batch: bool = True,
              url_parse_workers: int = 10,
              normalize_embeddings: bool = False) -> "QPointBuilder":

        self._encode_options.append(DenseOption(
            key=key,
            model=model,
            batch=batch,
            max_workers=url_parse_workers,
            type=data_type,
            data_obj_vector_fn=data_obj_vector_fn,
            data_obj_vector_key=data_obj_vector_key,
            normalize_embeddings=normalize_embeddings
        ))
        return self

    def sparse(self,
               key: str,
               model: SparseTextEmbedding,
               data_obj_vector_key: str = "vector_data",
               data_obj_vector_fn: Optional[DataObjVectorFn] = None,
               batch: bool = True) -> "QPointBuilder":

        self._encode_options.append(SparseOption(
            key=key,
            model=model,
            batch=batch,
            data_obj_vector_fn=data_obj_vector_fn,
            data_obj_vector_key=data_obj_vector_key
        ))
        return self

    def build_points(self) -> list[qdrant_models.PointStruct]:
        if not self._data:
            return []
        if not self._encode_options:
            raise ValueError("No embedding options specified")

        # vector_map
        # { "vector_key": {item_id: vector, ...}, ... }
        vector_map: dict[str, dict[int, Any]] = {}
        for option in self._encode_options:
            if isinstance(option, DenseOption):
                valid_items, vectors = run_dense(option, self._data)
            elif isinstance(option, SparseOption):
                valid_items, vectors = run_sparse(option, self._data)
            else:
                raise TypeError(f"Invalid embedding option type: {type(option)}")

            if valid_items and vectors is not None:
                vector_map[option.key] = {}  # 딕셔너리 초기화
                for item, vector in zip(valid_items, vectors):
                    vector_map[option.key][item.id] = vector

        # intersection for get valid items
        id_set_list = [
            set(vector_map[option.key].keys())
            for option in self._encode_options
        ]
        valid_item_ids = set.intersection(*id_set_list) if id_set_list else set()

        points: list[qdrant_models.PointStruct] = []
        for item in self._data:
            if item.id not in valid_item_ids:
                continue

            vector: qdrant_models.VectorStruct = {}
            for vector_key in vector_map.keys():
                vector[vector_key] = vector_map[vector_key][item.id]

            points.append(
                qdrant_models.PointStruct(
                    id=item.id,
                    vector=vector,
                    payload=item.payload
                )
            )

        return points

    def upsert(self, collection_name: str):
        points = self.build_points()
        self.qclient.upsert(collection_name=collection_name, points=points)
        logger.info(f"Upserted {len(points)}/{len(self._data)} points to '{collection_name}'")
