from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from config import Config
import time

# Hybrid retriever (optional): requires pinecone_text which depends on mmh3 build tools on Windows
HAS_HYBRID = False
try:
    # Check both the retriever class and the pinecone_text dependency
    from langchain_community.retrievers.pinecone_hybrid_search import (
        PineconeHybridSearchRetriever,  # type: ignore
    )
    try:
        from pinecone_text.hybrid import hybrid_convex_scale  # type: ignore
        HAS_HYBRID = True
    except Exception:
        HAS_HYBRID = False
except Exception:
    HAS_HYBRID = False

class HybridRetriever:
    def __init__(self, namespace: str = "default"):
        self.config = Config()
        self.namespace = namespace
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize Pinecone and create hybrid retriever"""
        # Initialize and cache clients/embeddings once
        self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
        self.index = self.pc.Index(self.config.PINECONE_INDEX_NAME)

        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.OPENAI_API_KEY,
            model=self.config.OPENAI_EMBEDDING_MODEL,
        )

        # Create retriever (hybrid if available, else dense-only)
        if HAS_HYBRID:
            self.retriever = PineconeHybridSearchRetriever(  # type: ignore
                embeddings=self.embeddings,
                index=self.index,
                top_k=5,
                alpha=0.5,
                namespace=self.namespace,
            )
            self.vectorstore = None
        else:
            # Use dense vectorstore for similarity search
            self.vectorstore = LC_Pinecone.from_existing_index(
                index_name=self.config.PINECONE_INDEX_NAME,
                embedding=self.embeddings,
                namespace=self.namespace,
            )
            self.retriever = None
    
    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5, fetch_k: int = 20) -> List[Document]:
        """
        Retrieve relevant documents using hybrid search
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            alpha: Weight for hybrid search (0.0 = sparse only, 1.0 = dense only)
            fetch_k: Initial candidates to fetch before MMR re-ranking (dense-only)
        """
        t0 = time.perf_counter()
        try:
            if HAS_HYBRID and self.retriever is not None:
                # Update dynamic parameters without re-instantiation
                try:
                    self.retriever.top_k = top_k  # type: ignore[attr-defined]
                    self.retriever.alpha = alpha  # type: ignore[attr-defined]
                except Exception:
                    pass
                documents = self.retriever.get_relevant_documents(query)  # type: ignore[union-attr]
            elif self.vectorstore is not None:
                # Use MMR for diversity and often better quality with small k
                try:
                    documents = self.vectorstore.max_marginal_relevance_search(query, k=top_k, fetch_k=fetch_k)
                except Exception:
                    documents = self.vectorstore.similarity_search(query, k=top_k)
            else:
                # Fallback: reinitialize if something went wrong
                self._initialize_retriever()
                return self.retrieve(query, top_k, alpha, fetch_k)

            t1 = time.perf_counter()
            print(f"[Retrieval] query='{query[:40]}...' top_k={top_k} alpha={alpha} fetch_k={fetch_k} took {(t1 - t0)*1000:.1f} ms")
            return documents
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def retrieve_with_metadata(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Retrieve documents with additional metadata
        """
        documents = self.retrieve(query, top_k, alpha)
        
        results = []
        for doc in documents:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, 'score', None)
            }
            results.append(result)
        
        return results
    
    def change_namespace(self, namespace: str):
        """Change the namespace for retrieval"""
        self.namespace = namespace
        self._initialize_retriever()