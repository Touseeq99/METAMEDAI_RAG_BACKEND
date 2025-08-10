from .retriever import HybridRetriever
from .generator import RAGGenerator
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional

class RAGService:
    def __init__(self, namespace: str = "default"):
        self.retriever = HybridRetriever(namespace)
        self.generator = RAGGenerator()
        self.namespace = namespace
    
    def query(self, question: str, top_k: int = 5, alpha: float = 0.5, 
              include_sources: bool = False, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve relevant documents and generate response
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            alpha: Hybrid search weight (0.0 = sparse only, 1.0 = dense only)
            include_sources: Whether to include source documents in response
            custom_prompt: Optional custom prompt template
        """
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(question, top_k, alpha)
            
            if not retrieved_docs:
                return {
                    "status": "warning",
                    "message": "No relevant documents found",
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "context_used": 0,
                    "sources": []
                }
            
            # Step 2: Generate response
            if include_sources:
                result = self.generator.generate_with_sources(question, retrieved_docs, custom_prompt)
            else:
                result = self.generator.generate_response(question, retrieved_docs, custom_prompt)
            
            # Add retrieval information
            result["retrieved_docs_count"] = len(retrieved_docs)
            result["search_alpha"] = alpha
            result["namespace"] = self.namespace
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "answer": None,
                "context_used": 0,
                "retrieved_docs_count": 0
            }
    
    def retrieve_only(self, query: str, top_k: int = 5, alpha: float = 0.5) -> Dict[str, Any]:
        """
        Only retrieve documents without generation
        """
        try:
            documents = self.retriever.retrieve_with_metadata(query, top_k, alpha)
            
            return {
                "status": "success",
                "documents": documents,
                "count": len(documents),
                "query": query,
                "search_alpha": alpha,
                "namespace": self.namespace
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "documents": [],
                "count": 0
            }
    
    def generate_only(self, question: str, context_text: str, 
                     custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate response with provided context (no retrieval)
        """
        try:
            # Create a document from the provided context
            context_doc = Document(page_content=context_text, metadata={})
            
            result = self.generator.generate_response(question, [context_doc], custom_prompt)
            result["context_provided"] = True
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "answer": None,
                "context_provided": True
            }
    
    def change_namespace(self, namespace: str):
        """Change the namespace for both retriever and service"""
        self.namespace = namespace
        self.retriever.change_namespace(namespace)
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the current namespace"""
        try:
            # This would need to be implemented based on your specific needs
            return {
                "status": "success",
                "namespace": self.namespace,
                "message": "Retrieval service is active"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            } 