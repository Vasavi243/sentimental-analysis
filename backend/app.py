"""
Flask Backend API for Sentiment Analysis
Provides endpoints for sentiment prediction
"""

import os
import sys
import pickle
import re
import string
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS


def preprocess_text(text):
    """
    Preprocess text for sentiment analysis.
    Must match the function used during training.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags symbol
    text = re.sub(r'@\w+|#', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


# Make preprocess_text available for pickle
import __main__
__main__.preprocess_text = preprocess_text

# Add model directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
    template_folder='../frontend/templates',
    static_folder='../frontend/static'
)
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'sentiment_model.pkl')
HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'prediction_history.txt')

# Global model variable
model = None


def load_model():
    """Load the trained sentiment analysis model."""
    global model
    if model is None:
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except FileNotFoundError:
            print("Error: Model file not found. Please train the model first.")
            raise
    return model


def predict_sentiment(text):
    """
    Predict sentiment of a given text.
    Returns: dict with sentiment, confidence, and probabilities
    """
    m = load_model()
    
    # Get prediction probabilities
    proba = m.predict_proba([text])[0]
    prediction = m.predict([text])[0]
    
    # Get confidence (probability of predicted class)
    confidence = proba[prediction]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return {
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "negative": round(proba[0] * 100, 2),
            "positive": round(proba[1] * 100, 2)
        }
    }


def save_to_history(text, result):
    """Save prediction to history file."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp}|{result['sentiment']}|{result['confidence']}|{text}\n")
    except Exception as e:
        print(f"Error saving to history: {e}")


def get_history():
    """Get prediction history from file."""
    history = []
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('|', 3)
                        if len(parts) == 4:
                            history.append({
                                "timestamp": parts[0],
                                "sentiment": parts[1],
                                "confidence": float(parts[2]),
                                "text": parts[3][:100] + "..." if len(parts[3]) > 100 else parts[3]
                            })
            # Return most recent 20 predictions
            return history[-20:][::-1]
    except Exception as e:
        print(f"Error reading history: {e}")
    return []


# API Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict sentiment of text.
    
    Request body: {"text": "tweet text here"}
    Response: {"sentiment": "Positive/Negative", "confidence": 85.5, "probabilities": {...}}
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No data provided",
                "success": False
            }), 400
        
        text = data.get('text', '').strip()
        
        # Validate input
        if not text:
            return jsonify({
                "error": "Please enter some text to analyze",
                "success": False
            }), 400
        
        if len(text) > 500:
            return jsonify({
                "error": "Text is too long. Maximum 500 characters allowed.",
                "success": False
            }), 400
        
        # Make prediction
        result = predict_sentiment(text)
        result['success'] = True
        
        # Save to history
        save_to_history(text, result)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "success": False
        }), 500


@app.route('/api/history', methods=['GET'])
def history():
    """Get prediction history."""
    try:
        history_data = get_history()
        return jsonify({
            "success": True,
            "history": history_data
        })
    except Exception as e:
        return jsonify({
            "error": f"Failed to get history: {str(e)}",
            "success": False
        }), 500


@app.route('/api/examples', methods=['GET'])
def examples():
    """Get example tweets for testing."""
    return jsonify({
        "success": True,
        "examples": [
            {
                "text": "I absolutely love this new phone! The camera is incredible! 📱",
                "category": "positive"
            },
            {
                "text": "This is the worst service I've ever experienced. So disappointed! 😠",
                "category": "negative"
            },
            {
                "text": "Just had the best vacation ever! The beach was beautiful! 🏖️",
                "category": "positive"
            },
            {
                "text": "Can't believe how terrible this product is. Complete waste of money!",
                "category": "negative"
            },
            {
                "text": "Feeling so happy and grateful today! Life is good! 😊",
                "category": "positive"
            },
            {
                "text": "This traffic is unbearable! Going to be late for my meeting!",
                "category": "negative"
            }
        ]
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "success": False
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "success": False
    }), 500


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please run train_model.py first.")
        sys.exit(1)

    load_model()

    print("Starting Sentiment Analysis API...")

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)