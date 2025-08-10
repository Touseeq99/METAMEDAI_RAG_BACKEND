#!/usr/bin/env python3
"""
Test script to verify the RAG application setup
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import fastapi
        print("✓ FastAPI imported successfully")
    except ImportError as e:
        print(f"✗ FastAPI import failed: {e}")
        return False
    
    try:
        import langchain
        print("✓ LangChain imported successfully")
    except ImportError as e:
        print(f"✗ LangChain import failed: {e}")
        return False
    
    try:
        import langchain_openai
        print("✓ LangChain OpenAI imported successfully")
    except ImportError as e:
        print(f"✗ LangChain OpenAI import failed: {e}")
        return False
    
    try:
        import langchain_community
        print("✓ LangChain Community imported successfully")
    except ImportError as e:
        print(f"✗ LangChain Community import failed: {e}")
        return False
    
    try:
        import pinecone
        print("✓ Pinecone imported successfully")
    except ImportError as e:
        print(f"✗ Pinecone import failed: {e}")
        return False
    
    try:
        import openai
        print("✓ OpenAI imported successfully")
    except ImportError as e:
        print(f"✗ OpenAI import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config import Config
        config = Config()
        print("✓ Configuration loaded successfully")
        
        # Check required environment variables
        required_vars = [
            "OPENAI_API_KEY",
            "PINECONE_API_KEY",
            "PINECONE_ENVIRONMENT"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(config, var, None):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"⚠ Missing environment variables: {', '.join(missing_vars)}")
            print("   Please set these in your .env file")
            return False
        else:
            print("✓ All required environment variables are set")
            return True
            
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_modules():
    """Test if application modules can be imported"""
    print("\nTesting application modules...")
    
    try:
        from ingestion.ingestion_service import IngestionService
        print("✓ IngestionService imported successfully")
    except Exception as e:
        print(f"✗ IngestionService import failed: {e}")
        return False
    
    try:
        from retrieval.rag_service import RAGService
        print("✓ RAGService imported successfully")
    except Exception as e:
        print(f"✗ RAGService import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("RAG Application Setup Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing dependencies:")
        print("   uv pip install -r requirements.txt")
        sys.exit(1)
    
    # Test configuration
    if not test_config():
        print("\n❌ Configuration test failed. Please check your .env file.")
        sys.exit(1)
    
    # Test modules
    if not test_modules():
        print("\n❌ Module test failed. Please check the application structure.")
        sys.exit(1)
    
    print("\n✅ All tests passed! Your RAG application is ready to run.")
    print("\nNext steps:")
    print("1. Start the application: python main.py")
    print("2. Visit http://localhost:8000/docs for API documentation")
    print("3. Try the health endpoint: http://localhost:8000/health")

if __name__ == "__main__":
    main() 