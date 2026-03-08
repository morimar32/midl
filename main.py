import time
import yaml
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model import initialize, is_initialized
from endpoints.openai import router as openai_router
from endpoints.anthropic import router as anthropic_router

# --- Load Configuration ---

CONFIG_PATH = "config.yaml"
config = {}

try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded successfully from {CONFIG_PATH}")
    if not config or 'model_path' not in config:
         print(f"Warning: 'model_path' not found in {CONFIG_PATH}. Using defaults.")
         config = config or {}
except FileNotFoundError:
    print(f"Error: Configuration file not found at {CONFIG_PATH}. Exiting.", file=sys.stderr)
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing configuration file {CONFIG_PATH}: {e}. Exiting.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading configuration: {e}. Exiting.", file=sys.stderr)
    sys.exit(1)

# --- FastAPI Application ---

app = FastAPI()

# Initialize the model with config
print(f"Initializing model with config: {config}")
initialize(config)

# Wait for model to fully initialize
while not is_initialized():
    print("Waiting for model initialization to complete...")
    time.sleep(1)
print("Model initialization complete")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Store config on app state for routers to access
app.state.config = config

# Include routers
app.include_router(openai_router)
app.include_router(anthropic_router)

# --- Server Execution ---

if __name__ == "__main__":
    print(f"Starting server with config: {config}")
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
