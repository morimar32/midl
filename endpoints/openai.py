import json
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from enrichr import enrich_request
from models import ChatMessageInput

router = APIRouter()


# --- OpenAI-specific Pydantic Models ---

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessageInput]
    model: Optional[str] = None
    stream: bool = False


class ChatMessageOutput(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessageOutput
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default-model-v1"
    choices: List[ChatChoice]


# --- Endpoint ---

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """OpenAI-compatible chat completions endpoint."""
    config = raw_request.app.state.config

    print("Received messages:")
    for message in request.messages:
        print(f"- Role: {message.role}, Content: {message.content}")

    body = await raw_request.json()
    print(f"Raw request body: {body}")

    try:
        if request.stream or body.get('stream', False):
            async def stream_response():
                response_id = f"chatcmpl-{uuid.uuid4()}"
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": config.get('model', 'default-model-v1'),
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

                try:
                    formatted_messages, response_text, usage = enrich_request(request.messages, config)
                except Exception as e:
                    print(f"Error processing request: {e}")
                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": config.get('model', 'default-model-v1'),
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"Error: {str(e)}"},
                            "finish_reason": "error"
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                chunk["choices"][0]["delta"] = {"content": response_text}
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream"
            )
        else:
            try:
                formatted_messages, response_text, usage = enrich_request(request.messages, config)
                response = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "index": 0,
                        "finish_reason": "stop"
                    }],
                    "model": config.get('model', 'default-model-v1'),
                    "usage": usage,
                }
                return JSONResponse(content=response)
            except Exception as e:
                print(f"Error processing request: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )

    except Exception as e:
        print(f"Error: {e}")
        raise
