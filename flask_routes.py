from flask import request, jsonify
import logging
from typing import List, Dict, Any

from dto.query_dto import ChatQuery, ChatResponse
from src.pinecone_manager import get_index, get_similar_documents
from src.openai_handler import get_embeddings, get_chat_completion

# Configure logging
logger = logging.getLogger(__name__)

def register_routes(app):
    @app.route('/api/chat', methods=['POST'])
    def chat():
        """
        Process a chat message using RAG pattern.
        
        1. Convert question to embedding
        2. Search for similar documents in Pinecone
        3. Use retrieved context to generate an answer with OpenAI
        """
        try:
            data = request.get_json()
            if not data or 'question' not in data:
                return jsonify({"error": "Missing question in request"}), 400
                
            question = data['question']
            logger.debug(f"Received chat query: {question}")
            
            # Get embedding for the question
            embeddings = get_embeddings([question])
            query_embedding = embeddings[0]
            
            # Retrieve similar documents from Pinecone
            similar_docs = get_similar_documents(query_embedding, top_k=5)
            
            if not similar_docs:
                return jsonify({
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "retrieved_context": []
                })
            
            # Extract text from documents
            context_texts = [doc.metadata.get('text', '') for doc in similar_docs]
            
            # Generate answer using retrieved context
            answer = get_chat_completion(question, context_texts)
            
            return jsonify({
                "answer": answer,
                "retrieved_context": context_texts
            })
        
        except Exception as e:
            logger.error(f"Error processing chat query: {str(e)}", exc_info=True)
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        try:
            # Check if Pinecone index is accessible
            index = get_index()
            if not index:
                return jsonify({"status": "error", "message": "Pinecone index not available"})
            
            return jsonify({"status": "ok", "message": "Service is healthy"})
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}", exc_info=True)
            return jsonify({"status": "error", "message": f"Service health check failed: {str(e)}"})