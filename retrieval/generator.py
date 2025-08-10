from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
from config import Config
import time

class RAGGenerator:
    def __init__(self):
        self.config = Config()
        self.llm = ChatOpenAI(
            openai_api_key=self.config.OPENAI_API_KEY,
            model=self.config.OPENAI_MODEL,
            temperature=0.3,
            max_tokens=512,
            timeout=20,
            max_retries=2,
        )
        self._setup_prompt_template()
    
    def _setup_prompt_template(self):
        """Setup the prompt template for RAG generation"""
        self.prompt_template = ChatPromptTemplate.from_template("""
        ROLE: You are an academic medical teacher. Explain clearly for educational purposes only.

        SAFETY AND GROUNDING (must follow):
        - Use ONLY the information in the Context below. Do not use outside knowledge.
        - If the context is insufficient, say: "I don't have enough information in the provided context to answer that." Then list what additional context would help.
        - Do NOT give medical advice, diagnosis, treatment recommendations, or actionable clinical guidance.

        TEACHING STYLE:
        - Be concise, structured, and neutral.
        - Prefer definitions first for key terms/acronyms.
        - Explain mechanisms or reasoning step-by-step.
        - Include a simple example or brief analogy when helpful.
        - Cite where statements come from using the provided context labels (e.g., "Document 1").
        - End with 1 short guiding question to check understanding or invite a follow-up.

        FORMAT:
        1) Summary (1–2 sentences)
        2) Key definitions/terms (if relevant)
        3) Explanation/steps
        4) Notes/limitations from context (and cite Document N)
        5) Guiding question

        Context:
        {context}

        Question:
        {question}

        Provide your educational explanation based solely on the context, following the structure above.""")
    
    def generate_response(self, question: str, context_documents: List[Document], 
                         custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response using retrieved context
        
        Args:
            question: User's question
            context_documents: Retrieved documents for context
            custom_prompt: Optional custom prompt template
        """
        try:
            # Prepare context from documents
            context = self._prepare_context(context_documents)
            
            # Use custom prompt if provided
            if custom_prompt:
                prompt = ChatPromptTemplate.from_template(custom_prompt)
            else:
                prompt = self.prompt_template
            
            # Create the chain
            chain = prompt | self.llm

            # Generate response
            t0 = time.perf_counter()
            response = chain.invoke({
                "context": context,
                "question": question
            })
            t1 = time.perf_counter()
            print(f"[Generation] tokens≈? took {(t1 - t0)*1000:.1f} ms")
            
            return {
                "status": "success",
                "answer": response.content,
                "context_used": len(context_documents),
                "question": question
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "answer": None,
                "context_used": 0,
                "question": question
            }
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context string from documents with total size cap to reduce latency"""
        if not documents:
            return "No relevant context found."

        max_total_chars = 4000  # rough cap to control prompt size
        per_doc_max = 1200      # avoid any single doc dominating
        used = 0
        parts: List[str] = []

        for i, doc in enumerate(documents, 1):
            if used >= max_total_chars:
                break
            content = (doc.page_content or "").strip()
            if len(content) > per_doc_max:
                content = content[:per_doc_max] + "..."

            metadata_str = ""
            if getattr(doc, "metadata", None):
                metadata_parts = []
                for key, value in doc.metadata.items():
                    if key not in ['source', 'file_name']:
                        metadata_parts.append(f"{key}: {value}")
                if metadata_parts:
                    metadata_str = f" (Metadata: {', '.join(metadata_parts)})"

            chunk = f"Document {i}{metadata_str}:\n{content}\n"
            # Trim if exceeding total cap
            remaining = max_total_chars - used
            if len(chunk) > remaining:
                chunk = chunk[:max(0, remaining)]
                if not chunk.endswith("..."):
                    chunk = chunk.rstrip() + "..."

            parts.append(chunk)
            used += len(chunk)

        return "\n".join(parts)
    
    def generate_with_sources(self, question: str, context_documents: List[Document],
                             custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate response with source citations
        """
        try:
            # Generate response
            result = self.generate_response(question, context_documents, custom_prompt)
            
            if result["status"] == "success":
                # Add source information
                sources = []
                for doc in context_documents:
                    source_info = {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)
                
                result["sources"] = sources
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "answer": None,
                "sources": []
            } 