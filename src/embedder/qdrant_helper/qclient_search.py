from qdrant_client import models, QdrantClient


def search_hybrid(qclient: QdrantClient,
                  collection_name: str,
                  sparse_vector: models.SparseVector,
                  dense_vector: list[float],
                  limit: int = 20):
    return qclient.query_points(
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

def search_dense(qclient: QdrantClient,
                 collection_name: str,
                 dense_vector: list[float],
                 limit: int = 20):
    return qclient.query_points(
        collection_name=collection_name,
        query=dense_vector,
        using="dense",
        limit=limit,
        with_payload=True
    )


def search_sparse(qclient: QdrantClient,
                  collection_name: str,
                  sparse_vector: models.SparseVector,
                  limit: int = 20):
    return qclient.query_points(
        collection_name=collection_name,
        query=sparse_vector,
        using="sparse",
        limit=limit,
        with_payload=True
    )