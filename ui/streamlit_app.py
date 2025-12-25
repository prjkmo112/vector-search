import os
import sys
import json
import time
from typing import Any

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from FlagEmbedding import BGEM3FlagModel

from imgtool import ImageWD
from PIL import Image
import io

from embedder import QPointBuilder, DenseOptionDataTypeEnum
from embedder.qtypes import DataItem


# Defaults from your repo
DEFAULT_COLLECTION_NAME = "commerce_product"
TEXT_EMBEDDING_MODEL = "Qdrant/bm25"
CLIP_MODEL_NAME = "clip-ViT-B-32"
BGEM3_MODEL_NAME = "BAAI/bge-m3"


load_dotenv()


def get_qdrant_client(host: str) -> QdrantClient:
    return QdrantClient(host)


@st.cache_resource(show_spinner=False)
def get_models(clip_model_name: str, bgem3_model_name: str, sparse_model_name: str):
    """Load all three models: CLIP (image), BGE-M3 (text), BM25 (sparse)"""
    clip = SentenceTransformer(clip_model_name)
    bgem3 = BGEM3FlagModel(bgem3_model_name, use_fp16=True)
    sparse = SparseTextEmbedding(model_name=sparse_model_name)
    return clip, bgem3, sparse


def build_sparse_vector(sparse_model: SparseTextEmbedding, text: str) -> models.SparseVector:
    emb = next(sparse_model.embed([text]))
    return models.SparseVector(indices=list(emb.indices), values=list(emb.values))


def build_dense_vector_from_image(clip_model: SentenceTransformer, image_bytes: bytes) -> list[float]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    vec = clip_model.encode(img)
    return vec.tolist()


def build_dense_vector_from_path_or_url(clip_model: SentenceTransformer, path_or_url: str) -> list[float]:
    img = ImageWD.open(path_or_url).convert("RGB")
    vec = clip_model.encode(img)
    return vec.tolist()


def build_dense_vector_from_text(clip_model: SentenceTransformer, text: str) -> list[float]:
    vec = clip_model.encode(text)
    return vec.tolist()


def pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


st.set_page_config(page_title="Vector Search UI (Qdrant)", layout="wide")

st.title("Qdrant 벡터 검색/적재 UI (Streamlit)")
st.caption("이미지(CLIP) + 텍스트(BGE-M3) dense 벡터, BM25 sparse 벡터 검색")

with st.sidebar:
    st.header("연결 설정")
    qdrant_host = st.text_input("Qdrant Host", value=os.getenv("QDRANT_CLIENT_IP", "http://localhost:6333"))
    collection_name = st.text_input("Collection", value=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME))

    st.divider()
    st.subheader("모델 설정")
    clip_model_name = st.text_input("CLIP 모델 (이미지)", value=os.getenv("CLIP_MODEL", CLIP_MODEL_NAME))
    bgem3_model_name = st.text_input("BGE-M3 모델 (텍스트)", value=os.getenv("BGEM3_MODEL", BGEM3_MODEL_NAME))
    sparse_model_name = st.text_input("Sparse 모델 (BM25)", value=os.getenv("TEXT_EMBEDDING_MODEL", TEXT_EMBEDDING_MODEL))

    st.divider()
    st.subheader("표시")
    limit = st.slider("검색 결과 개수", 1, 100, 20)
    show_payload = st.checkbox("payload 표시", value=True)

clip_model, bgem3_model, sparse_text_model = get_models(clip_model_name, bgem3_model_name, sparse_model_name)
qclient = get_qdrant_client(qdrant_host)

tabs = st.tabs(["🔎 검색", "ℹ️ 컬렉션/헬스"])

# -------------------------
# Search tab
# -------------------------
with tabs[0]:
    st.subheader("검색")

    colA, colB = st.columns([1.2, 1])
    with colA:
        algo = st.radio("검색 알고리즘", ["dense_image", "dense_text", "sparse", "hybrid"], horizontal=True, index=0)

        # sparse/hybrid에서만 텍스트 쿼리 표시
        query_text = ""
        if algo in ("sparse", "hybrid", "dense_text"):
            query_text = st.text_input("텍스트 쿼리", value="루이보스차")

        # dense_image/hybrid에서만 이미지 입력 표시
        uploaded = None
        dense_path_or_url = ""
        if algo in ("dense_image", "hybrid"):
            dense_source = st.selectbox("이미지 입력", ["업로드 이미지", "이미지 URL/경로"], index=0)
            if dense_source == "업로드 이미지":
                uploaded = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg", "webp"])
            else:
                dense_path_or_url = st.text_input("이미지 URL 또는 로컬 경로", value="")
        else:
            dense_source = None

        run = st.button("검색 실행", type="primary")

    with colB:
        st.markdown("#### 팁")
        st.markdown(
            """
- **dense_image**: CLIP 이미지 벡터로 유사도 검색 (image_dense)
- **dense_text**: BGE-M3 텍스트 벡터로 유사도 검색 (text_property_dense)
- **sparse**: BM25 기반 키워드 검색 (text_title_sparse)
- **hybrid**: sparse로 1차 후보 → image dense로 재정렬
            """.strip()
        )

    if run:
        # Validate inputs based on algorithm
        if algo in ("sparse", "hybrid") and not query_text.strip():
            st.error("sparse/hybrid 검색은 텍스트 쿼리가 필요합니다.")
            st.stop()

        if algo == "dense_text" and not query_text.strip():
            st.error("dense_text 검색은 텍스트 쿼리가 필요합니다.")
            st.stop()

        # Build image dense vector if needed
        image_dense_vector = None
        if algo in ("dense_image", "hybrid"):
            try:
                if dense_source == "업로드 이미지":
                    if uploaded is None:
                        st.error("이미지를 업로드하세요.")
                        st.stop()
                    image_bytes = uploaded.getvalue()
                    st.image(image_bytes, caption="Query Image", use_container_width=True)
                    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    image_dense_vector = clip_model.encode(img).tolist()
                else:  # 이미지 URL/경로
                    if not dense_path_or_url.strip():
                        st.error("이미지 URL/경로를 입력하세요.")
                        st.stop()
                    image_dense_vector = build_dense_vector_from_path_or_url(clip_model, dense_path_or_url.strip())
                    try:
                        st.image(dense_path_or_url.strip(), caption="Query Image", use_container_width=True)
                    except Exception:
                        pass
            except Exception as e:
                st.exception(e)
                st.stop()

        # Build text dense vector if needed
        text_dense_vector = None
        if algo == "dense_text":
            try:
                # BGE-M3 returns dict with 'dense_vecs'
                result_dict = bgem3_model.encode([query_text])
                if 'dense_vecs' in result_dict:
                    text_dense_vector = result_dict['dense_vecs'][0].tolist()
                else:
                    st.error("BGE-M3 모델 출력 형식이 예상과 다릅니다.")
                    st.stop()
            except Exception as e:
                st.exception(e)
                st.stop()

        try:
            with st.spinner("Qdrant 조회 중..."):
                if algo == "dense_image":
                    result = qclient.query_points(
                        collection_name=collection_name,
                        query=image_dense_vector,
                        using="image_dense",
                        limit=limit,
                        with_payload=True
                    )
                elif algo == "dense_text":
                    result = qclient.query_points(
                        collection_name=collection_name,
                        query=text_dense_vector,
                        using="text_property_dense",
                        limit=limit,
                        with_payload=True
                    )
                elif algo == "sparse":
                    sparse_vec = build_sparse_vector(sparse_text_model, query_text)
                    result = qclient.query_points(
                        collection_name=collection_name,
                        query=sparse_vec,
                        using="text_title_sparse",
                        limit=limit,
                        with_payload=True
                    )
                else:  # hybrid
                    sparse_vec = build_sparse_vector(sparse_text_model, query_text)
                    result = qclient.query_points(
                        collection_name=collection_name,
                        prefetch=[
                            models.Prefetch(
                                query=sparse_vec,
                                using="text_title_sparse",
                                limit=max(limit * 5, 50),
                            )
                        ],
                        query=image_dense_vector,
                        using="image_dense",
                        limit=limit,
                        with_payload=True
                    )
        except Exception as e:
            st.exception(e)
            st.stop()

        points = getattr(result, "points", []) or []
        if not points:
            st.warning("결과가 없습니다.")
        else:
            rows = []
            for p in points:
                payload = p.payload or {}
                rows.append({
                    "score": float(p.score) if p.score is not None else None,
                    "id": p.id,
                    "product_id": payload.get("product_id"),
                    "title": payload.get("title"),
                    "url": payload.get("url") or payload.get("image_url") or payload.get("image"),
                    "payload": payload if show_payload else None,
                })

            df = pd.DataFrame(rows)
            st.dataframe(df.drop(columns=["payload"]) if not show_payload else df, use_container_width=True, height=360)

            st.markdown("#### 이미지 미리보기")
            cols = st.columns(5)
            for i, r in enumerate(rows[:25]):
                url = r.get("url")
                if not url:
                    continue
                with cols[i % 5]:
                    try:
                        st.image(url, caption=f"{r.get('score'):.4f} / {r.get('product_id')}", use_container_width=True)
                    except Exception:
                        st.caption("이미지 로드 실패")
                        st.text(url)

            if show_payload:
                st.markdown("#### payload 상세")
                for r in rows[:min(10, len(rows))]:
                    with st.expander(f"{r.get('product_id')} · {r.get('title')} · score={r.get('score'):.5f}"):
                        st.code(pretty_json(r["payload"]), language="json")

# -------------------------
# Info tab
# -------------------------
with tabs[1]:
    st.subheader("Qdrant 상태/컬렉션 정보")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("헬스 체크"):
            try:
                st.json(qclient.get_collections().model_dump() if hasattr(qclient.get_collections(), "model_dump") else qclient.get_collections())
            except Exception as e:
                st.exception(e)
    with col2:
        if st.button("컬렉션 상세"):
            try:
                info = qclient.get_collection(collection_name)
                st.json(info.model_dump() if hasattr(info, "model_dump") else info)
            except Exception as e:
                st.exception(e)
