from llama_cpp import Llama
import os
import sys

# Global model instance
llm = None

def initialize(config: dict) -> None:
    """Initialize the LLM model with configuration."""
    global llm
    
    # Validate required config parameters
    required_keys = ['model_path', 'n_ctx', 'n_gpu_layers']
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required config key '{key}'. Exiting.", file=sys.stderr)
            sys.exit(1)
    
    # Verify model file exists
    if not os.path.exists(config['model_path']):
        print(f"Error: Model file not found at '{config['model_path']}'. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    try:
        print(f"Initializing model with path: {config['model_path']}")
        llm = Llama(
            model_path=config['model_path'],
            n_ctx=config['n_ctx'],
            n_gpu_layers=config['n_gpu_layers']
        )
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)

def is_initialized() -> bool:
    """Check if the model has been initialized."""
    global llm
    return llm is not None

def get_llm():
    """Get the initialized LLM instance."""
    global llm
    if not is_initialized():
        raise RuntimeError("Model not initialized")
    return llm