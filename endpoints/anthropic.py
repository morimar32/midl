import json
import time
import uuid
from typing import List, Optional, Union

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from enrichr import enrich_request
from models import ChatMessageInput

router = APIRouter()


# --- Anthropic-specific Pydantic Models ---

class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: str


class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[AnthropicContentBlock]]


class AnthropicMessagesRequest(BaseModel):
    model: str
    max_tokens: int
    system: Optional[str] = None
    messages: List[AnthropicMessage]
    stream: bool = False


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponseContentBlock(BaseModel):
    type: str = "text"
    text: str


class AnthropicMessagesResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[AnthropicResponseContentBlock]
    model: str
    stop_reason: str = "end_turn"
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


# --- Translation Functions ---

def anthropic_to_internal(request: AnthropicMessagesRequest) -> List[ChatMessageInput]:
    """Translate Anthropic messages request to internal ChatMessageInput list."""
    internal_messages = []

    if request.system:
        internal_messages.append(ChatMessageInput(role="system", content=request.system))

    for msg in request.messages:
        if isinstance(msg.content, str):
            content = msg.content
        else:
            # Concatenate text blocks
            content = "".join(block.text for block in msg.content if block.type == "text")
        internal_messages.append(ChatMessageInput(role=msg.role, content=content))

    return internal_messages


def internal_to_anthropic(response_text: str, usage: dict, request: AnthropicMessagesRequest) -> AnthropicMessagesResponse:
    """Translate internal response back to Anthropic Messages API format."""
    return AnthropicMessagesResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        content=[AnthropicResponseContentBlock(type="text", text=response_text)],
        model=request.model,
        stop_reason="end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        ),
    )


# --- Endpoint ---

@router.post("/v1/messages")
async def create_message(request: AnthropicMessagesRequest, raw_request: Request):
    """Anthropic Messages API compatible endpoint."""
    config = raw_request.app.state.config

    try:
        internal_messages = anthropic_to_internal(request)

        print("Received Anthropic messages:")
        for msg in internal_messages:
            print(f"- Role: {msg.role}, Content: {msg.content}")

        if request.stream:
            async def stream_response():
                try:
                    formatted_messages, response_text, usage = enrich_request(internal_messages, config)
                except Exception as e:
                    print(f"Error processing request: {e}")
                    error_payload = {
                        "type": "error",
                        "error": {"type": "api_error", "message": str(e)},
                    }
                    yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
                    return

                msg_id = f"msg_{uuid.uuid4().hex[:24]}"

                # 1. message_start
                message_start = {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": request.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": 0,
                        },
                    },
                }
                yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

                # 2. content_block_start
                content_block_start = {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
                yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"

                # 3. content_block_delta
                content_block_delta = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": response_text},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n"

                # 4. content_block_stop
                content_block_stop = {
                    "type": "content_block_stop",
                    "index": 0,
                }
                yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n"

                # 5. message_delta
                message_delta = {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {
                        "output_tokens": usage.get("output_tokens", 0),
                    },
                }
                yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

                # 6. message_stop
                message_stop = {"type": "message_stop"}
                yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
            )
        else:
            formatted_messages, response_text, usage = enrich_request(internal_messages, config)
            response = internal_to_anthropic(response_text, usage, request)
            return JSONResponse(content=response.model_dump())

    except Exception as e:
        print(f"Error in Anthropic endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            },
        )
