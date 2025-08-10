import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Default to modern, generally available models
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    # New-style config
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")  # e.g., "aws" or "gcp"
    PINECONE_REGION = os.getenv("PINECONE_REGION")  # e.g., "us-west1"
    # Backward compatibility: allow legacy PINECONE_ENVIRONMENT like "us-west1-gcp"
    _LEGACY_ENV = os.getenv("PINECONE_ENVIRONMENT")
    if (not PINECONE_CLOUD or not PINECONE_REGION) and _LEGACY_ENV:
        # Try to parse format: "<region>-<cloud>", e.g., "us-west1-gcp"
        parts = _LEGACY_ENV.split("-")
        if len(parts) >= 2:
            # cloud is last token; region is the rest re-joined
            PINECONE_CLOUD = PINECONE_CLOUD or parts[-1]
            PINECONE_REGION = PINECONE_REGION or "-".join(parts[:-1])
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")
    
    # Application Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K = int(os.getenv("TOP_K", "5"))
    
    # FastAPI Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))