# Vector Search

Qdrant 기반의 멀티모달 벡터 검색 시스템으로, 이미지와 텍스트를 모두 지원하는 Dense/Sparse 임베딩 검색 솔루션입니다.

## 개요

이 프로젝트는 이커머스 상품 데이터를 위한 완전한 벡터 검색 솔루션을 제공합니다:
- **이미지 임베딩**: CLIP 모델을 사용한 시각적 유사도 검색
- **텍스트 임베딩**: BGE-M3를 사용한 의미론적 텍스트 검색
- **희소 임베딩**: BM25를 사용한 키워드 기반 검색
- **Streamlit UI**: 쉬운 검색 및 데이터 관리

## 주요 기능

- 멀티모달 검색: 이미지, 텍스트 또는 둘 다로 쿼리 가능
- 하이브리드 검색: Dense와 Sparse 벡터를 결합하여 더 나은 결과 제공
- CSV 파일 배치 처리
- 검색 및 데이터 업로드를 위한 인터랙티브 웹 UI
- Docker 기반 Qdrant 배포


### Demo video
- Dense Image  


https://github.com/user-attachments/assets/48a7df0d-e0ef-470c-97ed-3459a1aa158d



- Dense Text  


https://github.com/user-attachments/assets/c81e42d1-1401-450f-9b49-bf8e1e3ea099



- Dense Text vs Sparse Text  


https://github.com/user-attachments/assets/a5f89af8-f739-4624-9a09-ae2d67962c6b



## 기술 스택

### 벡터 데이터베이스
- **Qdrant v1.16**: 고성능 벡터 검색 엔진

### 임베딩 모델
- **CLIP (clip-ViT-B-32)**: 512차원 이미지 임베딩
- **BGE-M3 (BAAI/bge-m3)**: 다국어 텍스트 임베딩
- **BM25 (Qdrant/bm25)**: 키워드 매칭을 위한 희소 텍스트 임베딩

### 프레임워크 & 라이브러리
- **Python 3.12+**
- **Streamlit**: 인터랙티브 웹 UI
- **sentence-transformers**: Dense 임베딩 모델
- **FlagEmbedding**: BGE-M3 모델 지원
- **fastembed**: 효율적인 희소 임베딩
- **pandas**: 데이터 처리
- **PIL**: 이미지 처리

## 프로젝트 구조

```
vector-search/
├── src/
│   ├── embedder/          # 핵심 임베딩 로직
│   │   ├── qpoint_builder.py         # 유연한 포인트 빌더
│   │   ├── qpoint_builder_single.py  # 단일 모드 빌더
│   │   ├── runners.py                # 인코딩 러너
│   │   └── options.py                # 모델 설정
│   ├── vector_search/     # 데이터 삽입 스크립트
│   │   ├── insert_vector_single.py
│   │   └── insert_vector_expand.py
│   └── imgtool/           # 이미지 유틸리티
├── ui/
│   └── streamlit_app.py   # 웹 인터페이스
├── docker-compose.yml     # Qdrant 서비스
└── pyproject.toml         # 의존성
```

## 설치

1. 저장소 클론:
```bash
git clone <repository-url>
cd vector-search
```

2. uv를 사용한 의존성 설치 (권장):
```bash
pip install uv
uv sync
```

또는 pip 사용:
```bash
pip install -e .
```

3. `.env` 파일 생성:
```bash
QDRANT_CLIENT_IP=http://localhost:6333
QDRANT_COLLECTION=commerce_product
CLIP_MODEL=clip-ViT-B-32
BGEM3_MODEL=BAAI/bge-m3
TEXT_EMBEDDING_MODEL=Qdrant/bm25
```

## 사용법

### Qdrant 데이터베이스 시작

```bash
docker-compose up -d
```

### Streamlit UI 실행

```bash
streamlit run ui/streamlit_app.py
```

UI는 다음을 제공합니다:
- **검색 탭**: 이미지 업로드, 이미지 URL 또는 텍스트 쿼리로 검색
- **CSV 업로드 탭**: CSV 파일에서 상품 일괄 삽입
- **컬렉션 정보 탭**: 컬렉션 통계 및 상태 확인

### CSV 배치 삽입

```bash
python src/vector_search/insert_vector_single.py
```

스크립트 변수 설정:
- `CHUNK_SIZE`: 배치당 행 수 (기본값: 500)
- `START_ROW`: 시작 행 번호 (기본값: 1)
- `COLLECTION_NAME`: 대상 컬렉션 이름

## 핵심 기술

### 하이브리드 검색
검색 품질 향상을 위해 Dense와 Sparse 벡터 결합:
- Dense 벡터: 의미론적 유사도 포착
- Sparse 벡터: 키워드 매칭 보존
- 퓨전 랭킹: 두 점수 결합

### 배치 처리
효율적인 병렬 처리:
- 멀티스레딩을 통한 배치 이미지 로딩
- 모델 최적화를 통한 배치 인코딩
- 대용량 데이터셋을 위한 청크 단위 CSV 처리

### 유연한 포인트 빌더
`QPointBuilder` 클래스는 Fluent API를 제공합니다:

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

### 멀티모달 임베딩
단일 상품을 여러 표현으로 인덱싱:
- 이미지 벡터: 상품 이미지의 시각적 특징
- 텍스트 벡터: 제목/설명의 의미론적 특징
- 희소 벡터: 정확한 매칭을 위한 키워드 특징
