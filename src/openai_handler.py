from openai import OpenAI
import logging
from typing import List

from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, GENERATION_MODEL, TEMPERATURE

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def get_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """
    Generate embeddings for a list of text inputs.
    
    Args:
        texts: List of text strings to generate embeddings for
        model: OpenAI embedding model to use
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
        
    if not isinstance(texts, list):
        texts = [texts]
    
    try:
        response = client.embeddings.create(
            input=texts,
            model=model
        )
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def get_chat_completion(query: str, context: List[str], model: str = GENERATION_MODEL) -> str:
    """
    Generate a response for the user query based on retrieved context.
    
    Args:
        query: User question
        context: List of document chunks to use as context
        model: OpenAI model to use
        
    Returns:
        Generated response text
    """
    # Join context elements with clear separators
    context_str = "\n\n---\n\n".join(context)
    
    system_prompt = f"""
    You are a helpful AI assistant using a Retrieval Augmented Generation (RAG) system.
    Answer the user's question based ONLY on the provided context. 
    If the context doesn't contain enough information to answer the question, 
    say "I don't have enough information to answer that question."
    Don't make up information or use knowledge outside the provided context.
    Always cite your sources from the context if possible.
    
    Context:
    {context_str}
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=TEMPERATURE
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating chat completion: {str(e)}")
        raise
