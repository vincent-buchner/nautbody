from abc import ABC
from datetime import datetime


class ConversationEvent(ABC):
    """
    Man, Python abstract classes are ugly looking.
    """

    @property
    def timestamp(self) -> datetime:
        return datetime.now()
