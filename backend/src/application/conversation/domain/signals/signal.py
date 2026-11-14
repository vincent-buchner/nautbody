from abc import ABC
from typing import Any, TypeVar


class Signal(ABC):
    source_type: str
    payload: Any


TSignal = TypeVar("TSignal", bound=Signal, contravariant=True)
