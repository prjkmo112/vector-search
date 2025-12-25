import logging
from dataclasses import dataclass
from typing import Any

from PIL.ImageFile import ImageFile
from numpy import ndarray

from qdrant_client import models as qdrant_models
from qdrant_client.http.models import SparseVector

from .options import DenseOption, DenseOptionDataTypeEnum, SparseOption, EmbedderOptionType
from .qtypes import DataItem
from imgtool import ImageWD

logger = logging.getLogger(__name__)

RunDenseResult = tuple[None, None] | tuple[list[Any], ndarray]
RunSparseResult = tuple[None, None] | tuple[list[Any], list[SparseVector]]

@dataclass
class FilteredValidDataItem(DataItem):
    _filtered_valid_value: Any

def filter_valid_items(option: EmbedderOptionType, data_item_list: list[DataItem]) -> tuple[list[DataItem], list[Any]]:
    # valid_data_item_list: list[FilteredValidDataItem] = []

    # for resource efficient
    valid_data_values = []
    filtered_data_item_list = []

    for item in data_item_list:
        try:
            if option.data_obj_vector_fn:
                p = option.data_obj_vector_fn(item.raw)
            else:
                p = item.raw[option.data_obj_vector_key]

            if not p:
                logger.error(f"No data to convert vector : {item.id}")
                continue

            valid_data_values.append(p)
            filtered_data_item_list.append(item)
        except Exception:
            logger.error(f"Error: {item.id}", exc_info=True)
            continue

    return filtered_data_item_list, valid_data_values

def run_dense(option: DenseOption, data_item_list: list[DataItem]) -> RunDenseResult:
    # 1. process vector data parallel
    data_item_list, vector_data_list = filter_valid_items(option, data_item_list)

    if option.batch:
        loaded_data: list[ImageFile | None] = []
        if option.type == DenseOptionDataTypeEnum.IMAGE:
            loaded_data = ImageWD.open_batch(vector_data_list, max_workers=option.max_workers)
        elif option.type == DenseOptionDataTypeEnum.TEXT:
            loaded_data = vector_data_list

        if not loaded_data:
            logger.error("No valid items after encoding data.")
            return None, None

        # 2. filter and convert
        valid_items = []
        valid_values = []
        for data_item, loaded_item in zip(data_item_list, loaded_data):
            try:
                if loaded_item is None:
                    logger.warning(f"Failed to load image for item {data_item.id}")
                    continue

                if option.type == DenseOptionDataTypeEnum.IMAGE:
                    valid_values.append(loaded_item.convert("RGB"))
                elif option.type == DenseOptionDataTypeEnum.TEXT:
                    valid_values.append(loaded_item)

                valid_items.append(data_item)
            except Exception:
                logger.error(f"Error converting image for item {data_item.id}", exc_info=True)

        if not valid_items:
            return None, None

        # 3. batch encode
        try:
            # Check if model supports normalize_embeddings parameter
            import inspect
            encode_params = inspect.signature(option.model.encode).parameters

            if 'normalize_embeddings' in encode_params:
                dense_embeddings = option.model.encode(valid_values, normalize_embeddings=option.normalize_embeddings)
            else:
                dense_embeddings = option.model.encode(valid_values)

            dense_vector = None
            dense_emb_type = type(dense_embeddings)
            if dense_emb_type is dict:
                if 'dense_vecs' in dense_embeddings:
                    dense_vector = dense_embeddings['dense_vecs']
            elif dense_emb_type is list or dense_emb_type is ndarray:
                dense_vector = dense_embeddings.tolist()

            return valid_items, dense_vector
        except Exception:
            logger.error("Error in batch image encoding", exc_info=True)
            return None, None
    else:
        # implicit.
        return None, None

def run_sparse(option: SparseOption, data_item_list: list[DataItem]) -> RunSparseResult:
    # 1. process vector data
    data_item_list, valid_items = filter_valid_items(option, data_item_list)

    if not valid_items:
        logger.error("No valid items for sparse encoding.")
        return None, None

    # 2. batch encode
    try:
        sparse_vector = []

        # sparse vector not support batch encode
        sparse_embedding = option.model.embed(valid_items)
        for emb in sparse_embedding:
            sparse_vector.append(
                qdrant_models.SparseVector(
                    indices=emb.indices.tolist(),
                    values=emb.values.tolist()
                )
            )

        return data_item_list, sparse_vector
    except Exception as e:
        logger.error(f"Error in batch text encoding: {e}")
        return None, None
