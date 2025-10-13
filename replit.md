# RAG Chatbot with Pinecone and OpenAI

## Overview

This is a Retrieval Augmented Generation (RAG) chatbot application that combines vector search with large language models to provide accurate, context-aware responses. The system ingests documents, converts them to embeddings, stores them in Pinecone vector database, and uses OpenAI's models to generate responses based on retrieved context.

The application supports both Flask and FastAPI frameworks, processes documents through chunking and embedding pipelines, and provides a web-based chat interface for user interactions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Web Interface**: The application uses a server-side rendered approach with Flask templates and vanilla JavaScript for the chat interface. The UI is built with Bootstrap 5 (dark theme) and provides a real-time chat experience.

- **Template Engine**: Jinja2 templates with a base layout pattern for consistent styling
- **Styling**: Custom CSS with CSS variables for theming, dark mode optimized
- **JavaScript**: Vanilla JS for AJAX requests to the chat API endpoint
- **UI Components**: Chat message display, loading indicators, input forms with Font Awesome icons

### Backend Architecture

**Dual Framework Support**: The codebase supports both Flask and FastAPI frameworks, likely for flexibility in deployment scenarios.

- **Flask Implementation** (`flask_routes.py`, `main.py`): Traditional request/response handling with blueprint-style route registration
- **FastAPI Implementation** (`routes.py`): Modern async-capable API with Pydantic models for request/response validation
- **Route Pattern**: Single `/api/chat` POST endpoint for processing user queries through the RAG pipeline

**RAG Pipeline Components**:

1. **Embedding Generation** (`src/openai_handler.py`): Converts text to vector embeddings using OpenAI's embedding models
2. **Vector Search** (`src/pinecone_manager.py`): Retrieves semantically similar document chunks from Pinecone
3. **Response Generation** (`src/openai_handler.py`): Uses OpenAI's chat completion API with retrieved context to generate answers

**Text Processing** (`src/text_utils.py`): Document chunking with token-based splitting using tiktoken, supports overlapping chunks to maintain context continuity.

**Data Ingestion** (`ingest_data.py`): Batch processing pipeline for loading documents, chunking text, generating embeddings, and upserting to Pinecone with metadata.

### Data Storage Solutions

**Vector Database**: Pinecone Serverless for scalable vector storage and similarity search

- **Index Configuration**: Cosine similarity metric for semantic matching
- **Serverless Spec**: Cloud-agnostic deployment (AWS/GCP/Azure support)
- **Dimension**: Configurable based on embedding model (default 1536 for text-embedding-3-small)
- **Metadata Storage**: Document text stored alongside vectors for retrieval

**Knowledge Base**: File-based storage in `data/` directory, currently supporting text files with RAG system documentation.

### Configuration Management

**Environment-based Configuration** (`src/config.py`): 
- Centralized configuration using environment variables with `.env` file support
- Validation of required API keys at startup
- Model configuration with sensible defaults (GPT-4o for generation, text-embedding-3-small for embeddings)
- Temperature control for deterministic responses (default 0.0)

### Design Patterns

**Data Transfer Objects**: Pydantic models (`dto/query_dto.py`) for type-safe request/response handling, ensuring validation and serialization consistency.

**Separation of Concerns**: Clear module boundaries:
- Configuration layer (`src/config.py`)
- Service layer (`src/pinecone_manager.py`, `src/openai_handler.py`)
- Utility layer (`src/text_utils.py`)
- Presentation layer (templates, static assets)

**Error Handling**: Logging throughout the application with try-catch blocks for graceful degradation, particularly in API interactions and file operations.

## External Dependencies

### Third-Party APIs

**OpenAI API**:
- **Embedding Model**: text-embedding-3-small (1536 dimensions) for converting text to vectors
- **Generation Model**: GPT-4o for chat completions with context
- **Purpose**: Core AI functionality for both understanding queries and generating responses

**Pinecone Vector Database**:
- **Index Type**: Serverless for scalable deployment
- **Configuration**: Region-specific deployment (default us-east-1 on AWS)
- **Purpose**: Efficient similarity search across embedded document chunks

### Python Dependencies

**Web Frameworks**:
- Flask: Traditional web framework with template rendering
- FastAPI: Modern async API framework with automatic API documentation
- Pydantic: Data validation and serialization

**AI/ML Libraries**:
- openai: Official OpenAI Python client
- pinecone-client: Pinecone vector database SDK
- tiktoken: OpenAI's tokenization library for accurate token counting

**Utilities**:
- python-dotenv: Environment variable management
- tqdm: Progress bars for batch processing
- logging: Built-in Python logging for observability

### Environment Variables Required

- `OPENAI_API_KEY`: Authentication for OpenAI services
- `PINECONE_API_KEY`: Authentication for Pinecone vector database
- `PINECONE_INDEX_NAME`: Name of the Pinecone index (default: "rag-chatbot")
- `PINECONE_CLOUD`: Cloud provider (aws/gcp/azure)
- `PINECONE_REGION`: Deployment region
- `EMBEDDING_MODEL`: OpenAI embedding model identifier
- `GENERATION_MODEL`: OpenAI chat model identifier
- `EMBEDDING_MODEL_DIMENSION`: Vector dimension for embeddings