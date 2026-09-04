from abc import ABC
from dataclasses import dataclass
from datetime import datetime

import numpy as np


class ConversationEvent(ABC):
    """
    Man, Python abstract classes are ugly looking.
    """

    @property
    def timestamp(self) -> datetime:
        return datetime.now()


class UserStartSpeakingEvent(ConversationEvent): ...


@dataclass
class UserDeltaSpeakingEvent(ConversationEvent):
    audio_bytes: np.ndarray


class UserStopSpeakingEvent(ConversationEvent): ...
