from dataclasses import dataclass


@dataclass
class ConversationContext:
    is_user_speaking: bool = False
    is_assistant_processing: bool = False
