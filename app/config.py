import os
from dotenv import load_dotenv
from langfuse.openai import AsyncOpenAI
from langfuse import Langfuse

load_dotenv()

# Qdrant and Embedding Model Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "liv-scenarios-2"
EMBEDDING_MODEL_NAME = "static-retrieval-mrl-en-v1"

# TecDoc API Configuration
TECDOC_API_KEY = os.getenv("TECDOC_API_KEY")
TECDOC_HOST = "tecdoc-web-services.p.rapidapi.com"

# Initialize clients
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_client = AsyncOpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
langfuse = Langfuse()

# Model pricing configuration
MODEL_PRICES = {
    "default": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000},
    "gpt-4.1-mini": {"prompt": 0.0004 / 1000, "completion": 0.0016 / 1000},
    "llama-3.3-70b-versatile": {"prompt": 0.00059 / 1000, "completion": 0.00079 / 1000},
    # Add more models and their pricing here
} 