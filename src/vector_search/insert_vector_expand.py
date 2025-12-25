import gc
import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from fastembed import SparseTextEmbedding
from phpserialize3 import loads
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from embedder import DenseOptionDataTypeEnum, QPointBuilder
from embedder.qtypes import DataItem

logger = logging.getLogger()

formatter = logging.Formatter(
    fmt="%(asctime)s [ %(name)s | %(levelname)s | %(filename)s:%(lineno)s | %(funcName)s() ]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

now = datetime.now()

log_file_dir = Path.cwd() / "logs" / str(now.year) / str(now.month) / str(now.day)
log_file_path = log_file_dir / "vector-search.log"
log_file_dir.mkdir(parents=True, exist_ok=True)

file_handler = TimedRotatingFileHandler(
    filename=log_file_path,
    when="midnight",
    interval=1,
    backupCount=10,
    encoding="utf-8",
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

# Model Selection
COLLECTION_NAME = "commerce_product"

# Processing Variables
CHUNK_SIZE = 300
START_ROW = 7000 # default: 1


def get_imageurl_from_fphp(value):
    if pd.isna(value):
        return None

    try:
        # 이미 PHP serialize된 문자열인 경우
        if isinstance(value, str) and value.startswith('a:'):
            parsed = loads(value.encode('utf-8'))
            return parsed['main'][0]['src']

        # 이미 객체인 경우 그대로 반환
        return value

    except Exception as e:
        print(f"Error deserializing value: {value[:100] if isinstance(value, str) else value}, Error: {e}")
        return None

def data_obj_vector_fn(data_obj):
    # data object vector function
    return "\n".join(map(str, [
        data_obj['title'],
        f"{data_obj['brand']} {data_obj['maker']} {data_obj['model']}",
        data_obj['origin'],
        data_obj['keywords']
    ]))


def main():
    # connect to qdrant
    qclient = QdrantClient(os.getenv("QDRANT_CLIENT_IP"))

    # Qdrant Point builder
    qbuilder = QPointBuilder(qclient)

    # load model
    dense_clip_model = SentenceTransformer('clip-ViT-B-32')
    dense_bgem3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    sparse_bm25_model = SparseTextEmbedding('Qdrant/bm25')

    # get product list
    df_prods = pd.read_csv(os.getcwd() + "/inputs/jc_lohas_product.csv", chunksize=CHUNK_SIZE, skiprows=range(1, START_ROW))

    chunk_idx = 1
    for product_list in df_prods:
        product_list: pd.DataFrame

        product_list['image_url'] = product_list['site_img_list'].apply(get_imageurl_from_fphp)
        fill_na_cols = ['brand', 'maker', 'model', 'keywords']
        product_list[fill_na_cols] = product_list[fill_na_cols].fillna('')
        product_list = product_list.dropna(subset=['product_id', 'title', 'site_img_list', 'image_url'])

        data_list: list[DataItem] = []
        for i, prod in product_list.iterrows():
            try:
                data_list.append(DataItem(
                    id=prod['product_id'],
                    payload={
                        "product_id": prod['product_id'],
                        "site_id": prod['site_id'],
                        "site_name": prod['site_name'],
                        "title": prod['title'],
                        "category": prod['category'],
                        "price": prod['price_sale'],
                        "brand": prod['brand'],
                        "maker": prod['maker'],
                        "model": prod['model'],
                        "origin": prod['origin'],
                        "state": prod['state'],
                        "allow": prod['allow'],
                        "duplication": prod['duplication'],
                        "url": prod['image_url'],
                    },
                    raw=prod.to_dict()
                ))
            except Exception:
                logger.error(f"Error caused. product_id = {prod['product_id']}")
                continue

        (qbuilder.data(data_list)
            .dense("image_dense", dense_clip_model, data_obj_vector_key="image_url", data_type=DenseOptionDataTypeEnum.IMAGE, normalize_embeddings=True)
            .dense("text_property_dense", dense_bgem3_model, data_obj_vector_fn=data_obj_vector_fn, data_type=DenseOptionDataTypeEnum.TEXT)
            .sparse("text_title_sparse", sparse_bm25_model, batch=False, data_obj_vector_key="title")
            .upsert(COLLECTION_NAME))

        logger.info(f"Inserted {len(data_list)} items. Total: {START_ROW + CHUNK_SIZE * chunk_idx}")
        chunk_idx += 1

        del data_list
        del product_list
        gc.collect()

if __name__ == "__main__":
    main()