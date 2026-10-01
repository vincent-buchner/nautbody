from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .src.conversation.router import router as conversation_router

app = FastAPI()
app.include_router(conversation_router)


@app.get("/health")
def check_health() -> JSONResponse:
    return JSONResponse(
        {"message": "ok"},
    )
