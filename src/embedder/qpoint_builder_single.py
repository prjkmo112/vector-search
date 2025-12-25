import logging
from enum import Enum
from typing import Any, TypedDict

from fastembed import SparseTextEmbedding
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

from imgtool import ImageWD

logger = logging.getLogger(__name__)


QPointPayload = dict[str, Any] | None

class QPointVector(TypedDict):
    dense: list[float]
    sparse: models.SparseVector

class QPoint(TypedDict):
    id: int | None
    vector: QPointVector
    payload: QPointPayload

class QPointBatchItem(TypedDict):
    id: int | None
    text: str
    image_path: str
    payload: QPointPayload

class QPointBuilderSingle:
    def __init__(self,
                 qclient: QdrantClient,
                 sentence_transformer: str,
                 text_embedding_model: str):
        self.qclient = qclient
        self.clip_model = SentenceTransformer(sentence_transformer)
        self.sparse_model = SparseTextEmbedding(model_name=text_embedding_model)

    def _encode_images_sequential(self, items: list[QPointBatchItem]) -> tuple[list[QPointBatchItem], list]:
        """이미지를 순차적으로 인코딩"""
        valid_items = []
        dense_embeddings = []

        for item in items:
            try:
                image = ImageWD.open(item['image_path']).convert("RGB")
                embedding = self.clip_model.encode(image)
                valid_items.append(item)
                dense_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error encoding image for item {item['id']}: {e}")
                continue

        return valid_items, dense_embeddings

    def _encode_images_batch(self, items: list[QPointBatchItem], max_workers: int = 10) -> tuple[list[QPointBatchItem], list]:
        """이미지를 배치로 인코딩 (병렬 다운로드)"""
        # Step 1: 병렬로 모든 이미지 다운로드/로드
        image_paths = [item['image_path'] for item in items]
        loaded_images = ImageWD.open_batch(image_paths, max_workers=max_workers)

        # Step 2: 성공한 이미지만 필터링 및 RGB 변환
        valid_items = []
        images = []
        for item, img in zip(items, loaded_images):
            if img is not None:
                try:
                    images.append(img.convert("RGB"))
                    valid_items.append(item)
                except Exception as e:
                    logger.error(f"Error converting image for item {item['id']}: {e}")
                    continue
            else:
                logger.error(f"Failed to load image for item {item['id']}")

        if not images:
            return [], []

        # Step 3: Batch encode
        try:
            dense_embeddings = self.clip_model.encode(images)
            return valid_items, list(dense_embeddings)
        except Exception as e:
            logger.error(f"Error in batch image encoding: {e}")
            return [], []

    def _encode_texts_sequential(self, items: list[QPointBatchItem]) -> list:
        """텍스트를 순차적으로 인코딩"""
        sparse_embeddings = []

        for item in items:
            try:
                sparse_embedding = next(self.sparse_model.embed([item['text']]))
                sparse_embeddings.append(sparse_embedding)
            except Exception as e:
                logger.error(f"Error encoding text for item {item['id']}: {e}")
                sparse_embeddings.append(None)

        return sparse_embeddings

    def _encode_texts_batch(self, items: list[QPointBatchItem]) -> list:
        """텍스트를 배치로 인코딩"""
        if not items:
            return []

        texts = [item['text'] for item in items]

        try:
            sparse_embeddings = list(self.sparse_model.embed(texts))
            return sparse_embeddings
        except Exception as e:
            logger.error(f"Error in batch text encoding: {e}")
            return [None] * len(items)

    def upsert(
            self,
            collection_name: str,
            data: list[QPointBatchItem],
            use_batch: bool = True,
            url_request_max_workers: int = 10
    ):
        """
        데이터를 Qdrant에 업서트

        Args:
            collection_name: 컬렉션 이름
            data: 업서트할 데이터 리스트
            use_batch: True면 배치 처리, False면 순차 처리
            url_request_max_workers: url(이미지) 요청 병렬 worker 갯수
        """
        if not data:
            return

        points = []

        if use_batch:
            # 배치 처리 모드
            valid_items, dense_embeddings = self._encode_images_batch(data, max_workers=url_request_max_workers)

            if not valid_items:
                logger.warning("No valid items after image encoding")
                return

            sparse_embeddings = self._encode_texts_batch(valid_items)

            # Build points
            for item, dense_emb, sparse_emb in zip(valid_items, dense_embeddings, sparse_embeddings):
                if sparse_emb is None:
                    logger.warning(f"Skipping item {item['id']} due to text encoding failure")
                    continue

                try:
                    dense_vec = dense_emb.tolist()
                    sparse_vec = models.SparseVector(
                        indices=list(sparse_emb.indices),
                        values=list(sparse_emb.values)
                    )
                    points.append(
                        models.PointStruct(
                            id=item['id'],
                            vector={
                                "dense": dense_vec,
                                "sparse": sparse_vec
                            },
                            payload=item['payload']
                        )
                    )
                except Exception as e:
                    logger.error(f"Error building point for item {item['id']}: {e}")
                    continue

        else:
            # 순차 처리 모드 (기존 방식)
            for item in data:
                try:
                    # image -> embedding
                    image = ImageWD.open(item['image_path']).convert("RGB")
                    embedding = self.clip_model.encode(image)
                    dense_vec = embedding.tolist()

                    # text -> embedding
                    sparse_embedding = next(self.sparse_model.embed([ item['text'] ]))
                    sparse_vec = models.SparseVector(
                        indices=list(sparse_embedding.indices),
                        values=list(sparse_embedding.values)
                    )
                    points.append(
                        models.PointStruct(
                            id=item['id'],
                            vector={
                                "dense": dense_vec,
                                "sparse": sparse_vec
                            },
                            payload=item['payload']
                        )
                    )
                except Exception as e:
                    logger.error(f"Error: {item['id']} {e}")
                    continue

        if points:
            self.qclient.upsert(collection_name=collection_name, points=points)
            logger.info(f"Upserted {len(points)}/{len(data)} points to '{collection_name}'")

    def search_hybrid(self, collection_name: str, sparse_vector: models.SparseVector, dense_vector: list[float], limit: int = 20):
        return self.qclient.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=1000
                )
            ],
            query=dense_vector,
            using="dense",
            limit=limit,
            with_payload=True
        )

    def search_dense(self, collection_name: str, dense_vector: list[float], limit: int = 20):
        return self.qclient.query_points(
            collection_name=collection_name,
            query=dense_vector,
            using="dense",
            limit=limit,
            with_payload=True
        )

    def search_sparse(self, collection_name: str, sparse_vector: models.SparseVector, limit: int = 20):
        return self.qclient.query_points(
            collection_name=collection_name,
            query=sparse_vector,
            using="sparse",
            limit=limit,
            with_payload=True
        )

class SearchAlgorithmEnum(Enum):
    HYBRID = "hybrid"
    DENSE = "dense"
    SPARSE = "sparse"

