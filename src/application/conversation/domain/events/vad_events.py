from dataclasses import dataclass

import numpy as np

from application.conversation.domain.events.event import (
    ConversationEvent,
)


class UserStartSpeakingEvent(ConversationEvent): ...


@dataclass
class UserDeltaSpeakingEvent(ConversationEvent):
    audio_bytes: np.ndarray


class UserStopSpeakingEvent(ConversationEvent): ...
