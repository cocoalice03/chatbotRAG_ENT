import os
import argparse
import logging
from tqdm import tqdm
import time

from src.config import PINECONE_INDEX_NAME
from src.text_utils import split_text_into_chunks
from src.pinecone_manager import init_pinecone, delete_index, get_index
from src.openai_handler import get_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_file(file_path):
    """Read text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

def ingest_text(text, chunk_size=500, overlap=50, batch_size=10):
    """
    Process the text, split into chunks, create embeddings, and store in Pinecone.
    
    Args:
        text: The document text to process
        chunk_size: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks
        batch_size: Number of vectors to upsert in one batch
    """
    # 1. Split text into chunks
    logger.info(f"Splitting text into chunks (size={chunk_size}, overlap={overlap})...")
    chunks = split_text_into_chunks(text, max_tokens_per_chunk=chunk_size, overlap_tokens=overlap)
    logger.info(f"Created {len(chunks)} chunks from the document")
    
    # 2. Initialize Pinecone
    logger.info("Initializing Pinecone...")
    index = init_pinecone()
    
    # 3. Process chunks in batches to avoid rate limits
    logger.info("Processing chunks and uploading to Pinecone...")
    vectors = []
    
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch_chunks = chunks[i:i+batch_size]
        
        # Create embeddings for this batch
        embeddings = get_embeddings(batch_chunks)
        
        # Create vectors with metadata
        batch_vectors = [
            {
                "id": f"chunk_{i+j}",
                "values": embeddings[j],
                "metadata": {"text": batch_chunks[j], "chunk_id": i+j}
            }
            for j in range(len(batch_chunks))
        ]
        
        vectors.extend(batch_vectors)
        
        # Upsert vectors to Pinecone
        logger.debug(f"Upserting batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
        # Use upsert_items instead of upsert for newer Pinecone client
        index.upsert(vectors=batch_vectors)
        
        # Sleep to avoid rate limiting
        time.sleep(0.5)
    
    logger.info(f"Successfully ingested {len(vectors)} vectors into Pinecone index '{PINECONE_INDEX_NAME}'")
    return len(vectors)

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone for RAG")
    parser.add_argument("--file", type=str, default="data/knowledge_base.txt", 
                        help="Path to the text file to ingest")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Maximum number of tokens per chunk")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Number of tokens to overlap between chunks")
    parser.add_argument("--reset", action="store_true", 
                        help="Delete the existing index before ingestion")
    
    args = parser.parse_args()
    
    try:
        # Check if file exists
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return
        
        # Reset index if requested
        if args.reset:
            logger.info(f"Deleting existing index '{PINECONE_INDEX_NAME}'...")
            delete_index()
            logger.info("Waiting for index deletion to complete...")
            time.sleep(10)  # Wait for deletion to complete
        
        # Read the document
        logger.info(f"Reading document from {args.file}...")
        text = read_file(args.file)
        
        # Ingest the document
        count = ingest_text(text, args.chunk_size, args.overlap)
        logger.info(f"Successfully ingested document with {count} chunks")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
