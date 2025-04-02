# Plan: Create OpenAI-Compatible Chat Endpoint and Add Configuration

## 1. Goal

Create a simple, lightweight OpenAI-compatible chat endpoint in Python that listens on port 8080, receives a message, prints it, and returns a static "Hello World" response, strictly adhering to the OpenAI `/v1/chat/completions` request/response structure. Subsequently, add a centralized configuration file for model parameters.

## 2. Technology Choice

*   **Language:** Python
*   **Framework:** FastAPI (for API creation, validation)
*   **Server:** Uvicorn (ASGI server)
*   **Configuration Format:** YAML (`PyYAML` library)

## 3. Project Setup

*   Define project dependencies in `requirements.txt`:
    ```
    fastapi
    uvicorn[standard] # Includes pydantic
    PyYAML # For reading config.yaml
    ```
*   *(Implementation Step: User runs `pip install -r requirements.txt`)*

## 4. API Structure Definition (Pydantic Models)

*   **Request Model (`ChatCompletionRequest`):** Mirrors OpenAI structure, requiring `messages` list (with `role`, `content`).
    ```python
    # Example Request Model Snippet (in main.py)
    from pydantic import BaseModel, Field
    from typing import List, Optional
    import time
    import uuid

    class ChatMessageInput(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        messages: List[ChatMessageInput]
        model: Optional[str] = None # Make model optional for flexibility
        # Add other fields as needed for strict compatibility if required by client
    ```
*   **Response Model (`ChatCompletionResponse`):** Mirrors OpenAI structure, including `id`, `object`, `created`, `choices` (with `message`, `finish_reason`).
    ```python
    # Example Response Model Snippet (in main.py)
    class ChatMessageOutput(BaseModel):
        role: str = "assistant"
        content: str

    class ChatChoice(BaseModel):
        index: int = 0
        message: ChatMessageOutput
        finish_reason: str = "stop"

    class ChatCompletionResponse(BaseModel):
        id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
        object: str = "chat.completion"
        created: int = Field(default_factory=lambda: int(time.time()))
        model: str = "static-model-v1" # Provide a static model name
        choices: List[ChatChoice]
        # usage: Optional[dict] = None # Optional usage stats
    ```

## 5. FastAPI Application (`main.py`)

*   Import `FastAPI`, `Uvicorn`, Pydantic models, `List`, `uuid`, `time`.
*   Create `app = FastAPI()` instance.
*   Define `POST /v1/chat/completions` endpoint.
*   Endpoint function:
    *   Accepts `ChatCompletionRequest` body.
    *   Prints content of each message in `request.messages` to console.
    *   Creates `ChatMessageOutput` with `content="Hello World"`.
    *   Creates `ChatChoice` embedding the message output.
    *   Creates `ChatCompletionResponse` embedding the choice.
    *   Returns the `ChatCompletionResponse` object.

## 6. Server Execution (`main.py`)

*   Add `if __name__ == "__main__":` block.
*   Use `uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)` (reload=True is useful for development).

## 7. Diagram (API Flow)

```mermaid
graph LR
    Client -- POST /v1/chat/completions (JSON Request - OpenAI Spec) --> FastAPI_App[FastAPI App (main.py)];
    FastAPI_App -- Validates Request --> Pydantic_Req_Model[Request Model (OpenAI Spec)];
    FastAPI_App -- Extracts & Prints --> Console{Console Output};
    FastAPI_App -- Creates Response --> Pydantic_Res_Model[Response Model (OpenAI Spec)];
    FastAPI_App -- JSON Response (Static "Hello World" - OpenAI Spec) --> Client;
    Uvicorn[Uvicorn Server] -- Runs --> FastAPI_App;
    Uvicorn -- Listens --> Port_8080{Port 8080};
```

## 8. Configuration Management

*   **Goal:** Centralize model and operational parameters.
*   **File:** `config.yaml` in the project root (`/home/morimar/dev/python/modl/config.yaml`).
*   **Format:** YAML.
*   **Content (`config.yaml`):**
    ```yaml
    # Main model configuration
    model_path: "/path/to/your/model.gguf"  # TODO: Specify the path to your main model file
    n_ctx: 2048                           # Example: Maximum context size (e.g., 2048, 4096)
    n_gpu_layers: -1                      # Example: Number of layers to offload to GPU (-1 for all, 0 for none)
    max_tokens: 512                       # Example: Maximum tokens to generate per request
    temperature: 0.8                      # Example: Sampling temperature (e.g., 0.7, 0.8)
    top_p: 0.95                           # Example: Nucleus sampling threshold (e.g., 0.9, 0.95)

    # Optional draft model configuration (leave blank or comment out if not used)
    draft_model_path: ""                  # Path to your draft model .gguf file (optional)
    draft_n_ctx: ""                       # Maximum context size for draft model (optional)
    draft_n_gpu_layers: ""                # Number of layers to offload to GPU for draft model (optional)
    ```
*   **Integration (Future Step - Code Mode):**
    *   Add `PyYAML` to `requirements.txt`.
    *   Modify `main.py` to load settings from `config.yaml`.
    *   Use loaded settings instead of hardcoded values.
    *   Handle optional draft settings.
*   **Diagram (Config Usage):**
    ```mermaid
    graph TD
        A[main.py] -- Reads --> B(config.yaml);
        B -- Contains --> C{model_path};
        B -- Contains --> D{n_ctx};
        B -- Contains --> E{n_gpu_layers};
        B -- Contains --> F{max_tokens};
        B -- Contains --> G{temperature};
        B -- Contains --> H{top_p};
        B -- Contains --> I(Optional Draft Settings);
    ```

## 9. Next Steps (Implementation)

1.  Create `config.yaml` with the specified content.
2.  Update `requirements.txt` to include `PyYAML`.
3.  Modify `main.py` to load and use the configuration.