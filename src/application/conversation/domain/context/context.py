from dataclasses import dataclass


@dataclass
class ConversationContext:
    is_user_speaking: bool
    is_assistant_speaking: bool = False
