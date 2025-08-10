from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any
import os

class IngestionService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = VectorStore()
    
    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None, namespace: str = "default") -> Dict[str, Any]:
        """
        Ingest raw text into the vector store
        """
        try:
            # Process text into chunks
            documents = self.document_processor.process_text(text, metadata)
            
            # Add to vector store
            self.vector_store.add_documents(documents, namespace)
            
            return {
                "status": "success",
                "message": f"Successfully ingested {len(documents)} chunks",
                "chunks_processed": len(documents),
                "namespace": namespace
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "chunks_processed": 0,
                "namespace": namespace
            }
    
    def ingest_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None, namespace: str = "default") -> Dict[str, Any]:
        """
        Ingest a text file into the vector store
        """
        try:
            # Process file into chunks
            documents = self.document_processor.process_file(file_path, metadata)
            
            # Add to vector store
            self.vector_store.add_documents(documents, namespace)
            
            return {
                "status": "success",
                "message": f"Successfully ingested file: {file_path}",
                "chunks_processed": len(documents),
                "file_path": file_path,
                "namespace": namespace
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "chunks_processed": 0,
                "file_path": file_path,
                "namespace": namespace
            }
    
    def ingest_directory(self, directory_path: str, file_extensions: List[str] = [".txt", ".md"], namespace: str = "default") -> Dict[str, Any]:
        """
        Ingest all text files in a directory
        """
        results = []
        total_chunks = 0
        
        try:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if any(file.endswith(ext) for ext in file_extensions):
                        file_path = os.path.join(root, file)
                        result = self.ingest_file(file_path, namespace=namespace)
                        results.append(result)
                        if result["status"] == "success":
                            total_chunks += result["chunks_processed"]
            
            return {
                "status": "success",
                "message": f"Processed {len(results)} files",
                "total_chunks": total_chunks,
                "file_results": results,
                "namespace": namespace
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "total_chunks": 0,
                "file_results": results,
                "namespace": namespace
            }
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        """
        try:
            stats = self.vector_store.get_stats()
            return {
                "status": "success",
                "stats": stats
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def delete_namespace(self, namespace: str) -> Dict[str, Any]:
        """
        Delete all vectors in a specific namespace
        """
        try:
            self.vector_store.delete_namespace(namespace)
            return {
                "status": "success",
                "message": f"Successfully deleted namespace: {namespace}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            } 