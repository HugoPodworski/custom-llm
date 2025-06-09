import os
from contextlib import asynccontextmanager

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer
from app.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, EMBEDDING_MODEL_NAME
from app.routers import chat, search, tools

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Lifespan: Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    app.state.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Lifespan: Embedding model initialized.")

    app.state.qdrant_client = None
    if QDRANT_URL:
        qdrant_client_init_args = {
            "url": QDRANT_URL,
            "api_key": QDRANT_API_KEY,
            "prefer_grpc": True,
            "timeout": 10,
            "grpc_options": {
                'grpc.keepalive_time_ms': 30000,
                'grpc.keepalive_timeout_ms': 10000,
                'grpc.keepalive_permit_without_calls': 1,
                'grpc.http2.min_time_between_pings_ms': 10000,
                'grpc.http2.max_pings_without_data': 0,
            }
        }
        try:
            print(f"Lifespan: Initializing AsyncQdrantClient with args: {qdrant_client_init_args}")
            app.state.qdrant_client = AsyncQdrantClient(**qdrant_client_init_args)
            print("Lifespan: AsyncQdrantClient object created. Attempting warm-up call...")
            # Warm-up call
            await app.state.qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            print(f"Lifespan: Successfully connected to collection '{QDRANT_COLLECTION_NAME}' (warm-up successful).")
        except Exception as e:
            print(f"Lifespan: Error initializing AsyncQdrantClient or during warm-up: {e}")
            print("Lifespan: Please ensure Qdrant is running, accessible, the collection exists, and credentials are correct.")
            app.state.qdrant_client = None 
    else:
        print("Lifespan: QDRANT_URL environment variable not set. Qdrant client will not be initialized.")
    
    yield
    # Shutdown
    if app.state.qdrant_client:
        print("Lifespan: Closing Qdrant client...")
        await app.state.qdrant_client.close()
        print("Lifespan: Qdrant client closed.")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, tags=["Chat"])
app.include_router(search.router, tags=["Search"])
app.include_router(tools.router, tags=["Tools"])

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"} 