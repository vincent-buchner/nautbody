from collections.abc import AsyncIterator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from gateway.api.src.conversation.dependencies import make_start_conversation_use_case
from gateway.api.src.conversation.service import (
    register_ws_handlers,
)

router = APIRouter(prefix="/conversation")


@router.websocket("/connect")
async def conversation_websocket(websocket: WebSocket) -> None:

    async def audio_chunks_from_ws() -> AsyncIterator[bytes]:
        while True:
            input_data = await websocket.receive_bytes()
            yield input_data

    await websocket.accept()

    try:
        # FIXME: We should not make a new conversation with every connect,
        # we should have a singleton for initializing
        # the conversation and then an other method for attaching an audio source

        # async for audio_chunk in audio_chunks_from_ws():
        #     print(audio_chunk)
        conversation = make_start_conversation_use_case(audio_chunks_from_ws())
        register_ws_handlers(websocket, conversation)
        await conversation.execute()
    except WebSocketDisconnect:
        pass
