import json
import time
import uuid
import yaml # Added for config loading
import sys  # Added for error handling exit
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from enrichr import process_request, initialize

# --- Load Configuration ---

CONFIG_PATH = "config.yaml"
config = {}

try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded successfully from {CONFIG_PATH}")
    # Basic validation (optional, can be expanded)
    if not config or 'model_path' not in config:
         print(f"Warning: 'model_path' not found in {CONFIG_PATH}. Using defaults.")
         # Provide default structure if needed, or rely on .get() later
         config = config or {} # Ensure config is at least an empty dict
except FileNotFoundError:
    print(f"Error: Configuration file not found at {CONFIG_PATH}. Exiting.", file=sys.stderr)
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing configuration file {CONFIG_PATH}: {e}. Exiting.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading configuration: {e}. Exiting.", file=sys.stderr)
    sys.exit(1)


# --- Pydantic Models (Based on PLAN.md Section 4) ---

class ChatMessageInput(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessageInput]
    model: Optional[str] = None # Make model optional for flexibility
    # Add other fields as needed for strict compatibility if required by client

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
    # Use model name from config if available, otherwise use a default
    model: str = config.get('model', 'default-model-v1') # Updated to use config
    choices: List[ChatChoice]
    # usage: Optional[dict] = None # Optional usage stats

# --- FastAPI Application ---

app = FastAPI()

# Initialize the model with config
initialize(config)
print(f"Initializing model with config: {config}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]  # Expose all headers
)

# --- API Endpoint ---

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    Receives messages, prints them, and returns a static response.
    Uses configuration loaded at startup.
    """
    print("Received messages:")
    for message in request.messages:
        print(f"- Role: {message.role}, Content: {message.content}")
    
    # Log raw request for debugging
    body = await raw_request.json()
    print(f"Raw request body: {body}")

    try:
        if body.get('stream', False):
            # Handle streaming response
            async def stream_response():
                response_id = f"chatcmpl-{uuid.uuid4()}"
                # Send the initial chunk
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "default-model-v1",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant"
                        },
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                
                response_text = process_request(request.messages, config)
                # Send the content in a separate chunk
                chunk["choices"][0]["delta"] = {"content": response_text}
                yield f"data: {json.dumps(chunk)}\n\n"
                
                # Send the done message
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response (unchanged)
            response_text = process_request(request.messages, config)
            response = {"id": f"chatcmpl-{uuid.uuid4()}", "object": "chat.completion", "created": int(time.time()), "choices": [{"message": {"role": "assistant", "content": response_text}, "index": 0, "finish_reason": "stop"}], "model": "default-model-v1"}
            return JSONResponse(content=response)

    except Exception as e:
        print(f"Error: {e}")
        raise
        print(f"Error creating response: {e}")
        raise e from None

# --- Server Execution ---

if __name__ == "__main__":
    # You might want to pass config details to uvicorn if needed,
    # but for now, it's loaded globally.
    print(f"Starting server with config: {config}")
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)