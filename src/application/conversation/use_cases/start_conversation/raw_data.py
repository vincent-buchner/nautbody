from abc import ABC
from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np

SourceType = Literal["vad", "llm_stream"]


class _BasePayload: ...


TPayload = TypeVar("TPayload", bound=_BasePayload)


class RawDataSchema[TPayload: _BasePayload](ABC):
    _Payload = _BasePayload

    source_type: SourceType
    payload: TPayload


@dataclass
class _VADPayload(_BasePayload):
    vad_data: dict[str, float] | None
    audio_bytes: np.ndarray


class VADData(RawDataSchema[_VADPayload]):
    _Payload = _VADPayload

    source_type: SourceType = "vad"
    payload: _Payload


@dataclass
class _LLMPayload(_BasePayload):
    is_true: bool


class LLMData(RawDataSchema[_LLMPayload]):
    _Payload = _LLMPayload

    source_type: SourceType = "llm_stream"
    payload: _Payload


RawData = VADData | LLMData
TRawData = TypeVar("TRawData", bound=RawData, contravariant=True)
