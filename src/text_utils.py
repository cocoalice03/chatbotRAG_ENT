import tiktoken
import logging

logger = logging.getLogger(__name__)

def get_tokenizer(model_name="text-embedding-3-small"):
    """
    Get an appropriate tokenizer for the given model.
    
    Args:
        model_name: Name of the model to get tokenizer for
        
    Returns:
        A tokenizer instance
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to cl100k_base for newer models
        logger.warning(f"No specific tokenizer found for {model_name}, using cl100k_base")
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text, model_name="text-embedding-3-small"):
    """
    Count the number of tokens in the text.
    
    Args:
        text: Text to count tokens for
        model_name: Name of the model to use for counting
        
    Returns:
        Number of tokens
    """
    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer.encode(text)
    return len(tokens)

def split_text_into_chunks(text, max_tokens_per_chunk=500, overlap_tokens=50):
    """
    Split text into chunks with a specified overlap.
    
    Args:
        text: Text to split
        max_tokens_per_chunk: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    
    chunks = []
    current_pos = 0
    
    while current_pos < len(tokens):
        # Calculate end position for current chunk
        end_pos = min(current_pos + max_tokens_per_chunk, len(tokens))
        chunk_tokens = tokens[current_pos:end_pos]
        
        # Decode tokens back to text
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move position for next chunk, accounting for overlap
        if end_pos == len(tokens):
            break
            
        current_pos += (max_tokens_per_chunk - overlap_tokens)
        
        # Avoid infinite loops if overlap is too large
        if current_pos >= len(tokens):
            break
    
    return chunks

def split_text_by_separator(text, separator="\n\n", min_length=50):
    """
    Split text by a separator, ensuring each chunk meets a minimum length.
    
    Args:
        text: Text to split
        separator: Separator to split by
        min_length: Minimum length for a chunk
        
    Returns:
        List of text chunks
    """
    parts = text.split(separator)
    chunks = []
    current_chunk = ""
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # If adding this part would make the chunk too long, save current chunk and start a new one
        if current_chunk and count_tokens(current_chunk + separator + part) > max_tokens_per_chunk:
            chunks.append(current_chunk)
            current_chunk = part
        else:
            # Add separator only if we already have content
            if current_chunk:
                current_chunk += separator
            current_chunk += part
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
