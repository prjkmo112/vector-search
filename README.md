# Vector Search

A multimodal vector search system powered by Qdrant, supporting both image and text search using dense and sparse embeddings.

## Overview

This project provides a complete vector search solution for e-commerce product data, combining:
- **Image embeddings** using CLIP models for visual similarity search
- **Text embeddings** using BGE-M3 for semantic text search
- **Sparse embeddings** using BM25 for keyword-based retrieval
- **Streamlit UI** for easy search and data management

## Features

- Multimodal search: Query by image, text, or both
- Hybrid search: Combines dense and sparse vectors for better results
- Batch data processing from CSV files
- Interactive web UI for search and data upload
- Docker-based Qdrant deployment

## Tech Stack

### Vector Database
- **Qdrant v1.16**: High-performance vector search engine

### Embedding Models
- **CLIP (clip-ViT-B-32)**: Image embeddings with 512 dimensions
- **BGE-M3 (BAAI/bge-m3)**: Multilingual text embeddings
- **BM25 (Qdrant/bm25)**: Sparse text embeddings for keyword matching

### Framework & Libraries
- **Python 3.12+**
- **Streamlit**: Interactive web UI
- **sentence-transformers**: Dense embedding models
- **FlagEmbedding**: BGE-M3 model support
- **fastembed**: Efficient sparse embeddings
- **pandas**: Data processing
- **PIL**: Image processing

## Project Structure

```
vector-search/
├── src/
│   ├── embedder/          # Core embedding logic
│   │   ├── qpoint_builder.py         # Flexible point builder
│   │   ├── qpoint_builder_single.py  # Single-mode builder
│   │   ├── runners.py                # Encoding runners
│   │   └── options.py                # Model configurations
│   ├── vector_search/     # Data insertion scripts
│   │   ├── insert_vector_single.py
│   │   └── insert_vector_expand.py
│   └── imgtool/           # Image utilities
├── ui/
│   └── streamlit_app.py   # Web interface
├── docker-compose.yml     # Qdrant service
└── pyproject.toml         # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vector-search
```

2. Install dependencies using uv (recommended):
```bash
pip install uv
uv sync
```

Or with pip:
```bash
pip install -e .
```

3. Create `.env` file:
```bash
QDRANT_CLIENT_IP=http://localhost:6333
QDRANT_COLLECTION=commerce_product
CLIP_MODEL=clip-ViT-B-32
BGEM3_MODEL=BAAI/bge-m3
TEXT_EMBEDDING_MODEL=Qdrant/bm25
```

## Usage

### Start Qdrant Database

```bash
docker-compose up -d
```

### Run Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

The UI provides:
- **Search Tab**: Search by image upload, image URL, or text query
- **CSV Upload Tab**: Batch insert products from CSV files
- **Collection Info Tab**: View collection statistics and health

### Batch Insert from CSV

```bash
python src/vector_search/insert_vector_single.py
```

Configure the script variables:
- `CHUNK_SIZE`: Number of rows per batch (default: 500)
- `START_ROW`: Starting row number (default: 1)
- `COLLECTION_NAME`: Target collection name

## Key Techniques

### Hybrid Search
Combines dense and sparse vectors for improved search quality:
- Dense vectors capture semantic similarity
- Sparse vectors preserve keyword matching
- Fusion ranking combines both scores

### Batch Processing
Efficient parallel processing:
- Batch image loading with multi-threading
- Batch encoding with model optimization
- Chunked CSV processing for large datasets

### Flexible Point Builder
The `QPointBuilder` class provides a fluent API:

```python
from embedder import QPointBuilder, DenseOptionDataTypeEnum

builder = QPointBuilder(qclient)
points = (builder
    .data(data_items)
    .dense("image_vector", clip_model, DenseOptionDataTypeEnum.IMAGE)
    .dense("text_vector", bgem3_model, DenseOptionDataTypeEnum.TEXT)
    .sparse("sparse_vector", bm25_model)
    .build_points())
```

### Multimodal Embeddings
Single product indexed with multiple representations:
- Image vector: Visual features from product images
- Text vector: Semantic features from titles/descriptions
- Sparse vector: Keyword features for exact matching
