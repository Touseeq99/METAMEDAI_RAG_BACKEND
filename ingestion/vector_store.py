from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import os
from config import Config

class VectorStore:
    def __init__(self):
        self.config = Config()
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.OPENAI_API_KEY,
            model=self.config.OPENAI_EMBEDDING_MODEL
        )
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index (serverless)."""
        self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)

        # Determine embedding dimension from selected model
        # text-embedding-3-small: 1536, text-embedding-3-large: 3072
        dim = 1536 if "3-small" in (self.config.OPENAI_EMBEDDING_MODEL or "") else 3072

        existing = {ix.name for ix in self.pc.list_indexes()}
        if self.config.PINECONE_INDEX_NAME not in existing:
            # Prefer new-style cloud/region; raise if unavailable
            cloud = self.config.PINECONE_CLOUD
            region = self.config.PINECONE_REGION
            if not cloud or not region:
                raise ValueError("Pinecone cloud/region not configured. Set PINECONE_CLOUD and PINECONE_REGION in .env or provide legacy PINECONE_ENVIRONMENT that can be parsed.")

            self.pc.create_index(
                name=self.config.PINECONE_INDEX_NAME,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self.index = self.pc.Index(self.config.PINECONE_INDEX_NAME)
    
    def add_documents(self, documents: List[Document], namespace: str = "default") -> None:
        """
        Add documents to the vector store
        """
        vectorstore = LC_Pinecone.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.config.PINECONE_INDEX_NAME,
            namespace=namespace
        )
        return vectorstore
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, namespace: str = "default") -> None:
        """
        Add raw texts to the vector store
        """
        vectorstore = LC_Pinecone.from_texts(
            texts=texts,
            embedding=self.embeddings,
            index_name=self.config.PINECONE_INDEX_NAME,
            metadatas=metadatas,
            namespace=namespace
        )
        return vectorstore
    
    def get_vectorstore(self, namespace: str = "default") -> LC_Pinecone:
        """
        Get existing vector store instance
        """
        return LC_Pinecone.from_existing_index(
            index_name=self.config.PINECONE_INDEX_NAME,
            embedding=self.embeddings,
            namespace=namespace
        )
    
    def delete_namespace(self, namespace: str) -> None:
        """
        Delete all vectors in a specific namespace
        """
        self.index.delete(namespace=namespace)
    
    def get_stats(self) -> dict:
        """
        Get index statistics
        """
        return self.index.describe_index_stats() 