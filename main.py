from flask import Flask, render_template, jsonify, request
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Import routes after the app is created to avoid circular dependencies
from flask_routes import register_routes

# Register routes
register_routes(app)

@app.route('/')
def index():
    """Serve the chat interface."""
    return render_template('index.html')

if __name__ == "__main__":
    logger.info("Starting RAG Chatbot server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
