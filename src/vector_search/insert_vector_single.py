import time

from dotenv import load_dotenv

import os
import pandas as pd
from qdrant_client import QdrantClient

from embedder import QPointBuilderSingle, qtypes

# logger
import logging
from logging.handlers import RotatingFileHandler

# Model Selection
SENTENCE_TRANSFORMER_MODEL = "clip-ViT-B-32"
TEXT_EMBEDDING_MODEL = "Qdrant/bm25"
COLLECTION_NAME = "test"

load_dotenv()

logger = logging.getLogger()

file_handler = RotatingFileHandler(
    filename=os.getcwd() + "/logs/vector-search.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8"
)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)


# Processing Variables
CHUNK_SIZE = 500
START_ROW = 72000 # default: 1

def main():
    # connect to qdrant
    qclient = QdrantClient(os.getenv("QDRANT_CLIENT_IP"))

    # QPointBuilder instance
    qpoint_builder = QPointBuilderSingle(qclient, SENTENCE_TRANSFORMER_MODEL, TEXT_EMBEDDING_MODEL)

    # get product list
    df_prods = pd.read_csv(os.getcwd() + "/inputs/jc_lohas_product_limit_0.2m_0.csv", chunksize=CHUNK_SIZE, skiprows=range(1, START_ROW))

    idx = 1
    for product_list in df_prods:
        data_list = []
        for i, prod in product_list.iterrows():
            try:
                if not prod['url']:
                    continue

                data_list.append(qtypes.QPointSingleBatchItem(
                    id=prod['product_id'],
                    text=prod['title'],
                    image_path=prod['url'],
                    payload=prod.to_dict(),
                ))
            except Exception as e:
                logger.error(f"Error: {prod['product_id']} {e}")
                continue

        qpoint_builder.upsert(
            collection_name=COLLECTION_NAME,
            data=data_list,
            url_request_max_workers=10
        )

        # # FOR TEST:
        # if i >= 10:
        #     break

        logger.info(f"Processed {START_ROW + CHUNK_SIZE * idx} products.")
        time.sleep(0.5)
        idx += 1

if __name__ == "__main__":
    main()