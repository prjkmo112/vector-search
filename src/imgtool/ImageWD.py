import logging
import os
import io
from datetime import timedelta
from pathlib import Path
from typing import IO, Literal, Union
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import PIL.Image
import requests_cache
from requests.adapters import HTTPAdapter
from PIL.Image import Image
from PIL.ImageFile import ImageFile


logger = logging.getLogger(__name__)

StrOrBytesPath = Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]

_cache_dir = Path.home() / ".cache" / "requests-cache"
_cache_dir.mkdir(parents=True, exist_ok=True)

session = requests_cache.CachedSession(
    cache_name=str(_cache_dir / "image"),
    backend='filesystem',
    expire_after=timedelta(days=30),
    allowable_methods=['GET'],
    allowable_codes=[200],
    stale_if_error=True
)

# 연결 풀 크기 증가 (병렬 요청 지원)
adapter = HTTPAdapter(
    pool_connections=20,
    pool_maxsize=20,
    max_retries=3,
    pool_block=False
)
session.mount('http://', adapter)
session.mount('https://', adapter)

def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)

def download_image(url: str) -> bytes:
    response = session.get(
        url,
        headers={
            "accept": 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            "accept-encoding": 'gzip, deflate, br, zstd',
            "accept-language": 'ko-KR,ko;q=0.6',
            "priority": 'u=0, i',
            "sec-ch-ua": '"Chromium";v="142", "Brave";v="142", "Not_A Brand";v="99"',
            "user-agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36'
        },
        timeout=5
    )
    response.raise_for_status()

    if response.from_cache:
        logger.debug(f"[CACHE HIT] {url}")
    else:
        logger.debug(f"[HTTP REQUEST] {url}")

    return response.content

class ImageWD(Image):
    """
    PIL Image class with auto download functionality
    """
    def __init__(self, image: Image):
        super().__init__(image)


def open(
    fp: StrOrBytesPath | IO[bytes],
    mode: Literal["r"] = "r",
    formats: list[str] | tuple[str, ...] | None = None,
    download_implicit_path: Path | None = None
) -> ImageFile:
    parsed_url = urlparse(fp)
    is_http = parsed_url.scheme in ("http", "https") and bool(parsed_url.netloc)

    if is_http:
        image_bytes = download_image(fp)
        image = PIL.Image.open(io.BytesIO(image_bytes), mode, formats)
    else:
        image = PIL.Image.open(fp, mode, formats)

    return image


def open_batch(
    fps: list[StrOrBytesPath | IO[bytes]],
    mode: Literal["r"] = "r",
    formats: list[str] | tuple[str, ...] | None = None,
    max_workers: int = 10
) -> list[ImageFile | None]:
    """
    여러 이미지를 병렬로 다운로드/로드 (기존 open() 함수 활용)

    Args:
        fps: 이미지 경로 또는 URL 리스트
        mode: 파일 모드
        formats: 허용할 이미지 포맷
        max_workers: 병렬 처리 워커 수

    Returns:
        이미지 리스트 (실패 시 해당 인덱스는 None)
    """
    results = [None] * len(fps)

    def load_single(idx: int, fp: StrOrBytesPath | IO[bytes]) -> tuple[int, ImageFile | None]:
        try:
            image = open(fp, mode, formats)
            return idx, image
        except Exception as e:
            logger.error(f"Error loading image {fp}: {e}")
            return idx, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single, idx, fp): idx for idx, fp in enumerate(fps)}

        for future in as_completed(futures):
            idx, image = future.result()
            results[idx] = image

    return results


__all__ = ["open", "open_batch", "ImageWD"]