from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict, Any
import logging

from dto.query_dto import ChatQuery, ChatResponse
from src.pinecone_manager import get_index, get_similar_documents
from src.openai_handler import get_embeddings, get_chat_completion

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/api/chat", response_model=ChatResponse)
async def chat(query: ChatQuery):
    """
    Process a chat message using RAG pattern.
    
    1. Convert question to embedding
    2. Search for similar documents in Pinecone
    3. Use retrieved context to generate an answer with OpenAI
    """
    try:
        logger.debug(f"Received chat query: {query.question}")
        
        # Get embedding for the question
        embeddings = get_embeddings([query.question])
        query_embedding = embeddings[0]
        
        # Retrieve similar documents from Pinecone
        similar_docs = get_similar_documents(query_embedding, top_k=5)
        
        if not similar_docs:
            return ChatResponse(
                answer="I couldn't find any relevant information to answer your question.",
                retrieved_context=[]
            )
        
        # Extract text from documents
        context_texts = [doc['metadata']['text'] for doc in similar_docs]
        
        # Generate answer using retrieved context
        answer = get_chat_completion(query.question, context_texts)
        
        return ChatResponse(
            answer=answer,
            retrieved_context=context_texts
        )
    
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if Pinecone index is accessible
        index = get_index()
        if not index:
            return {"status": "error", "message": "Pinecone index not available"}
        
        return {"status": "ok", "message": "Service is healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Service health check failed: {str(e)}"}
