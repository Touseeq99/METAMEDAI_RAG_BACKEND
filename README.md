# RAG Application

A modular Retrieval-Augmented Generation (RAG) application built with FastAPI, LangChain, Pinecone, and OpenAI. This application provides both document ingestion and intelligent querying capabilities using hybrid search.

## Features

- **Modular Architecture**: Separate modules for ingestion and retrieval
- **Hybrid Search**: Uses PineconeHybridRetriever for both semantic and keyword search
- **Multiple Ingestion Methods**: Support for text, files, and directory ingestion
- **Flexible Querying**: Complete RAG pipeline with customizable parameters
- **Namespace Support**: Organize data into different namespaces
- **RESTful API**: Full FastAPI implementation with automatic documentation

## Project Structure

```
├── ingestion/
│   ├── __init__.py
│   ├── document_processor.py    # Text chunking and preprocessing
│   ├── vector_store.py          # Pinecone vector store operations
│   └── ingestion_service.py     # Main ingestion orchestration
├── retrieval/
│   ├── __init__.py
│   ├── retriever.py             # Hybrid retrieval with Pinecone
│   ├── generator.py             # OpenAI-based text generation
│   └── rag_service.py           # Main RAG orchestration
├── main.py                      # FastAPI application
├── config.py                    # Configuration management
├── requirements.txt             # Python dependencies
├── env.example                  # Environment variables template
└── README.md                    # This file
```

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key and environment
- UV package manager (for installation)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-app
   ```

2. **Install dependencies using UV**:
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env with your actual API keys and configuration
   ```

4. **Required Environment Variables**:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PINECONE_API_KEY`: Your Pinecone API key
   - `PINECONE_CLOUD`: Cloud provider (e.g., `gcp` or `aws`)
   - `PINECONE_REGION`: Region (e.g., `us-west1`)

## Usage

### Starting the Application

```bash
python main.py
```

The application will start on `http://localhost:8000` by default.

### API Documentation

Once running, visit:
- **Interactive API docs**: `http://localhost:8000/docs`
- **Alternative API docs**: `http://localhost:8000/redoc`

### API Endpoints

#### Ingestion Endpoints

1. **Ingest Text** (`POST /ingest/text`)
   ```json
   {
     "text": "Your document text here...",
     "metadata": {"source": "manual", "category": "example"},
     "namespace": "default"
   }
   ```

2. **Ingest File** (`POST /ingest/file`)
   - Upload a `.txt` or `.md` file
   - Optional metadata as JSON string
   - Specify namespace

3. **Ingest Directory** (`POST /ingest/directory`)
   - Process all text files in a directory
   - Configurable file extensions
   - Specify namespace

#### Query Endpoints

1. **Complete RAG Query** (`POST /query`)
   ```json
   {
     "question": "What is the main topic?",
     "top_k": 5,
     "alpha": 0.5,
     "include_sources": true,
     "namespace": "default"
   }
   ```

2. **Retrieve Only** (`POST /retrieve`)
   ```json
   {
     "query": "search term",
     "top_k": 5,
     "alpha": 0.5,
     "namespace": "default"
   }
   ```

3. **Generate Only** (`POST /generate`)
   ```json
   {
     "question": "What does this mean?",
     "context": "Your context text here..."
   }
   ```

#### Management Endpoints

- `GET /health` - Health check
- `GET /stats` - Vector store statistics
- `DELETE /namespace/{namespace}` - Delete namespace

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Size of text chunks for processing |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K` | 5 | Number of documents to retrieve |
| `OPENAI_MODEL` | gpt-4o-mini | OpenAI model for generation |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-3-small | OpenAI model for embeddings |

### Example Usage

#### 1. Ingest a Document

```python
import requests

# Ingest text
response = requests.post("http://localhost:8000/ingest/text", json={
    "text": "This is a sample document about artificial intelligence...",
    "metadata": {"topic": "AI", "author": "John Doe"},
    "namespace": "ai-docs"
})
print(response.json())
```

#### 2. Query the Knowledge Base

```python
# Ask a question
response = requests.post("http://localhost:8000/query", json={
    "question": "What is artificial intelligence?",
    "top_k": 3,
    "alpha": 0.7,
    "include_sources": True,
    "namespace": "ai-docs"
})
print(response.json())
```

#### 3. Upload and Process Files

```python
# Upload a file
with open("document.txt", "rb") as f:
    files = {"file": f}
    data = {"namespace": "my-docs"}
    response = requests.post("http://localhost:8000/ingest/file", files=files, data=data)
print(response.json())
```

## Advanced Features

### Hybrid Search

The application uses PineconeHybridRetriever which combines:
- **Dense search**: Semantic similarity using embeddings
- **Sparse search**: Keyword matching using BM25

The `alpha` parameter controls the balance:
- `alpha = 0.0`: Pure sparse search (keywords)
- `alpha = 1.0`: Pure dense search (semantic)
- `alpha = 0.5`: Balanced hybrid search (default)

### Namespaces

Organize your data into different namespaces:
- Separate different document collections
- Query specific namespaces
- Manage data independently

### Custom Prompts

You can provide custom prompt templates for generation:
```json
{
  "question": "Summarize this",
  "custom_prompt": "You are a helpful assistant. Based on the context, provide a concise summary: {context}\n\nQuestion: {question}\n\nSummary:"
}
```

## Error Handling

The application includes comprehensive error handling:
- Input validation
- API key verification
- File processing errors
- Vector store operation errors

All endpoints return structured responses with status codes and error messages.

## Development

### Running Tests

```bash
# Install test dependencies
uv pip install pytest pytest-asyncio

# Run tests
pytest
```

### Code Structure

- **Modular Design**: Each component is self-contained
- **Type Hints**: Full type annotation for better development experience
- **Error Handling**: Comprehensive exception handling
- **Configuration**: Centralized configuration management

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required environment variables are set
2. **Pinecone Index**: The application will create the index if it doesn't exist
3. **File Upload**: Only `.txt` and `.md` files are supported
4. **Memory Issues**: Adjust `CHUNK_SIZE` for large documents

### Health Check

Use the health endpoint to verify your setup:
```bash
curl http://localhost:8000/health
```

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request 