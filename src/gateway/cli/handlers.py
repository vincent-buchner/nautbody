import asyncio
from collections.abc import Awaitable, Callable

from application.conversation.domain.events.llm_events import (
    AssistantResponseStartedEvent,
    AssistantResponseStoppedEvent,
)
from application.conversation.domain.events.stt_events import (
    AssistantTranscriptionFinishedEvent,
    AssistantTranscriptionStartedEvent,
)
from application.conversation.domain.events.tts_events import (
    AssistantSpeakingCancelledEvent,
    AssistantSpeakingFinishedEvent,
    AssistantSpeakingStartedEvent,
)
from application.conversation.domain.events.vad_events import (
    UserDeltaSpeakingEvent,
    UserInterruptedAssistantEvent,
    UserStartSpeakingEvent,
    UserStopSpeakingEvent,
)
from application.conversation.use_cases.start_conversation.start_conversation import (
    StartConversation,
)


def handle_user_start_speaking(event: UserStartSpeakingEvent) -> None:
    print(event.__class__.__name__)


def handle_user_stop_speaking(event: UserStopSpeakingEvent) -> None:
    print(event.__class__.__name__)


def handle_user_delta_speaking(event: UserDeltaSpeakingEvent) -> None:
    print(event.__class__.__name__)


def handle_user_interrupted_assistant(event: UserInterruptedAssistantEvent) -> None:
    print(event.__class__.__name__)


def handle_assistant_response_started(event: AssistantResponseStartedEvent) -> None:
    print(event.__class__.__name__)


def handle_assistant_response_stopped(event: AssistantResponseStoppedEvent) -> None:
    print(event.__class__.__name__)
    print(event.llm_response_text)


def handle_assistant_transcription_started(
    event: AssistantTranscriptionStartedEvent,
) -> None:
    print(event.__class__.__name__)


def handle_assistant_transcription_finished(
    event: AssistantTranscriptionFinishedEvent,
) -> None:
    print(event.__class__.__name__)
    print(f"transcription: {event.transcription}")


def handle_assistant_speaking_started(event: AssistantSpeakingStartedEvent) -> None:
    print(event.__class__.__name__)


def make_handle_assistant_speaking_finished(
    audio_out_buffer_queue: asyncio.Queue[bytes],
) -> Callable[[AssistantSpeakingFinishedEvent], Awaitable[None]]:
    async def handle_assistant_speaking_finished(
        event: AssistantSpeakingFinishedEvent,
    ) -> None:
        print(event.__class__.__name__)
        await audio_out_buffer_queue.put(event.audio_response)

    return handle_assistant_speaking_finished


def make_handle_assistant_speaking_cancelled(
    audio_out_buffer_queue: asyncio.Queue[bytes],
) -> Callable[[AssistantSpeakingCancelledEvent], None]:
    def handle_assistant_speaking_cancelled(
        event: AssistantSpeakingCancelledEvent,
    ) -> None:
        print(event.__class__.__name__)
        while not audio_out_buffer_queue.empty():
            audio_out_buffer_queue.get_nowait()

    return handle_assistant_speaking_cancelled


def register_cli_handlers(
    conversation: StartConversation,
    audio_out_buffer_queue: asyncio.Queue[bytes],
) -> None:
    """Attach the CLI's presentation-only listeners to a conversation.

    Everything registered here is specific to running as a local CLI
    (stdout logging, pushing finished audio onto the local speaker queue).
    Core orchestration logic lives in StartConversation itself; a future
    API gateway would register its own websocket-flavored handlers here
    instead of reusing these.
    """
    conversation.on(UserStartSpeakingEvent, handle_user_start_speaking)
    conversation.on(UserStopSpeakingEvent, handle_user_stop_speaking)
    conversation.on(UserDeltaSpeakingEvent, handle_user_delta_speaking)
    conversation.on(UserInterruptedAssistantEvent, handle_user_interrupted_assistant)
    conversation.on(AssistantResponseStartedEvent, handle_assistant_response_started)
    conversation.on(AssistantResponseStoppedEvent, handle_assistant_response_stopped)
    conversation.on(
        AssistantTranscriptionStartedEvent, handle_assistant_transcription_started
    )
    conversation.on(
        AssistantTranscriptionFinishedEvent,
        handle_assistant_transcription_finished,
    )
    conversation.on(AssistantSpeakingStartedEvent, handle_assistant_speaking_started)
    conversation.on(
        AssistantSpeakingFinishedEvent,
        make_handle_assistant_speaking_finished(audio_out_buffer_queue),
    )
    conversation.on(
        AssistantSpeakingCancelledEvent,
        make_handle_assistant_speaking_cancelled(audio_out_buffer_queue),
    )
