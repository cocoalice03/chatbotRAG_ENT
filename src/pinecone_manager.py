from pinecone import Pinecone, ServerlessSpec
import logging
import time

from src.config import (
    PINECONE_API_KEY, 
    PINECONE_INDEX_NAME, 
    PINECONE_CLOUD, 
    PINECONE_REGION,
    EMBEDDING_MODEL_DIMENSION
)

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def init_pinecone():
    """
    Initialize and return the Pinecone index.
    Creates the index if it doesn't exist.
    
    Returns:
        Pinecone index instance
    """
    # Get list of index names
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        # Create a Serverless index for better scalability
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_MODEL_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
        
        # Wait for index to be ready
        logger.info("Waiting for index to be ready...")
        time.sleep(10)  # Give some time for the index to initialize
    
    # Get the index
    index = pc.Index(PINECONE_INDEX_NAME)
    logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' is ready")
    return index

def get_index():
    """
    Get the Pinecone index.
    
    Returns:
        Pinecone index instance
    """
    try:
        return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        logger.error(f"Error getting Pinecone index: {str(e)}")
        return None

def get_index_stats():
    """
    Get statistics about the Pinecone index.
    
    Returns:
        Dictionary with index statistics
    """
    index = get_index()
    if not index:
        return None
    
    try:
        stats = index.describe_index_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting index stats: {str(e)}")
        return None

def get_similar_documents(query_embedding, top_k=5, namespace=""):
    """
    Retrieve similar documents from Pinecone based on the query embedding.
    
    Args:
        query_embedding: The embedding vector for the query
        top_k: Number of results to return
        namespace: Pinecone namespace to query
        
    Returns:
        List of documents with similarity scores
    """
    index = get_index()
    if not index:
        return []
    
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        
        return results.matches
    except Exception as e:
        logger.error(f"Error querying Pinecone: {str(e)}")
        return []

def delete_index():
    """
    Delete the Pinecone index.
    """
    try:
        # Get list of index names
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if PINECONE_INDEX_NAME in existing_indexes:
            pc.delete_index(PINECONE_INDEX_NAME)
            logger.info(f"Deleted Pinecone index '{PINECONE_INDEX_NAME}'")
    except Exception as e:
        logger.error(f"Error deleting Pinecone index: {str(e)}")
