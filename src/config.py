import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_MODEL_DIMENSION = int(os.getenv("EMBEDDING_MODEL_DIMENSION", 1536))
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o")

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Validate required environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your environment variables.")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set. Please add it to your environment variables.")

# App configuration
MAX_TOKENS_PER_CHUNK = 500
CHUNK_OVERLAP = 50
TEMPERATURE = 0.0  # Lower value for more deterministic responses
