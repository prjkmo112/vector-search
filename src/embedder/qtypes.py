from dataclasses import dataclass
from typing import Any

QPointPayload = dict[str, Any] | None
DataItemRaw = dict[str, Any]

@dataclass
class DataItem:
    id: int | None
    payload: QPointPayload
    raw: DataItemRaw

@dataclass
class QPointSingleBatchItem:
    id: int | None
    text: str
    image_path: str | None
    payload: QPointPayload