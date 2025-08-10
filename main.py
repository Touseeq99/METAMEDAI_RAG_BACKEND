from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import tempfile

from ingestion.ingestion_service import IngestionService
from retrieval.rag_service import RAGService
from config import Config

# Initialize FastAPI app
app = FastAPI(
    title="RAG Application",
    description="A Retrieval-Augmented Generation application using LangChain, Pinecone, and OpenAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
config = Config()
ingestion_service = IngestionService()
rag_service = RAGService()

# Pydantic models for request/response
class TextIngestionRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
    namespace: str = "default"

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    alpha: float = 0.5
    include_sources: bool = False
    custom_prompt: Optional[str] = None
    namespace: str = "default"

class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5
    namespace: str = "default"

class GenerationRequest(BaseModel):
    question: str
    context: str
    custom_prompt: Optional[str] = None

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "RAG Application is running",
        "version": "1.0.0",
        "status": "healthy"
    }

# Serve the simple HTML upload form
@app.get("/upload")
async def upload_page():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(here, "upload.html")
        if not os.path.exists(html_path):
            raise HTTPException(status_code=404, detail="upload.html not found")
        return FileResponse(html_path, media_type="text/html")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ingestion endpoints
@app.post("/ingest/text")
async def ingest_text(request: TextIngestionRequest):
    """Ingest raw text into the vector store"""
    try:
        # Auto-fill metadata/namespace if not provided
        metadata = {"source": "api:text"}
        namespace = "default"
        namespace = request.namespace or "default"

        result = ingestion_service.ingest_text(
            text=request.text,
            metadata=metadata,
            namespace=namespace
        )
        
        if result["status"] == "success":
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
):
    """Ingest a file into the vector store (.txt, .md, .pdf, .docx)"""
    try:
        # Validate file type
        if not file.filename.endswith(('.txt', '.md', '.pdf', '.docx')):
            raise HTTPException(status_code=400, detail="Only .txt, .md, .pdf and .docx files are supported")
        
        # Parse metadata if provided
        # Auto-generate minimal metadata and default namespace
        parsed_metadata = {"title": os.path.basename(file.filename)}
        namespace = "default"
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process the file
            result = ingestion_service.ingest_file(
                file_path=temp_file_path,
                metadata=parsed_metadata,
                namespace=namespace
            )
            
            if result["status"] == "success":
                return result
            else:
                raise HTTPException(status_code=400, detail=result["message"])
                
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/directory")
async def ingest_directory(
    directory_path: str = Form(...),
    file_extensions: str = Form(".txt,.md,.pdf,.docx"),
    namespace: str = Form("default")
):
    """Ingest all text files in a directory"""
    try:
        # Parse file extensions
        extensions = [ext.strip() for ext in file_extensions.split(",")]
        
        if not os.path.exists(directory_path):
            raise HTTPException(status_code=400, detail=f"Directory not found: {directory_path}")
        
        result = ingestion_service.ingest_directory(
            directory_path=directory_path,
            file_extensions=extensions,
            namespace=namespace
        )
        
        if result["status"] == "success":
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Retrieval and generation endpoints
@app.post("/query")
async def query(
    question: str = Form(...),
):
    """Complete RAG pipeline: retrieve and generate (frontend sends only 'question')."""
    try:
        # Server-side defaults
        namespace = "default"
        top_k = 5
        alpha = 0.5
        include_sources = False
        custom_prompt = None

        # Ensure namespace on service
        if namespace != rag_service.namespace:
            rag_service.change_namespace(namespace)

        result = rag_service.query(
            question=question,
            top_k=top_k,
            alpha=alpha,
            include_sources=include_sources,
            custom_prompt=custom_prompt,
        )

        if result["status"] in ["success", "warning"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
async def retrieve(
    query: str = Form(...),
):
    """Only retrieve documents without generation (frontend sends only 'query')."""
    try:
        # Server-side defaults
        namespace = "default"
        top_k = 5
        alpha = 0.5

        if namespace != rag_service.namespace:
            rag_service.change_namespace(namespace)

        result = rag_service.retrieve_only(
            query=query,
            top_k=top_k,
            alpha=alpha,
        )

        if result["status"] == "success":
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate(request: GenerationRequest):
    """Generate response with provided context (no retrieval)"""
    try:
        result = rag_service.generate_only(
            question=request.question,
            context_text=request.context,
            custom_prompt=request.custom_prompt
        )
        
        if result["status"] == "success":
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Management endpoints
@app.get("/stats")
async def get_stats():
    """Get vector store statistics"""
    try:
        result = ingestion_service.get_vector_store_stats()
        
        if result["status"] == "success":
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/namespace/{namespace}")
async def delete_namespace(namespace: str):
    """Delete all vectors in a specific namespace"""
    try:
        result = ingestion_service.delete_namespace(namespace)
        
        if result["status"] == "success":
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if required environment variables are set
        missing_vars = []
        if not getattr(config, "OPENAI_API_KEY", None):
            missing_vars.append("OPENAI_API_KEY")
        if not getattr(config, "PINECONE_API_KEY", None):
            missing_vars.append("PINECONE_API_KEY")

        # Pinecone location: either new CLOUD+REGION or legacy ENV parsed by Config
        cloud_ok = bool(getattr(config, "PINECONE_CLOUD", None))
        region_ok = bool(getattr(config, "PINECONE_REGION", None))
        if not (cloud_ok and region_ok):
            # If legacy couldn't be parsed, Config would leave these empty
            missing_vars.extend([v for v in ["PINECONE_CLOUD", "PINECONE_REGION"] if not getattr(config, v, None)])
        
        if missing_vars:
            return {
                "status": "unhealthy",
                "message": f"Missing environment variables: {', '.join(sorted(set(missing_vars)))}"
            }
        
        return {
            "status": "healthy",
            "message": "All services are operational"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    ) 