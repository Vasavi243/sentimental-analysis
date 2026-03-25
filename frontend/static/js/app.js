/**
 * Sentiment Analysis App - Frontend JavaScript
 * Handles UI interactions and API communication
 */

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
const tweetInput = document.getElementById('tweet-input');
const charCount = document.getElementById('char-count');
const predictBtn = document.getElementById('predict-btn');
const btnText = predictBtn.querySelector('.btn-text');
const btnLoader = predictBtn.querySelector('.btn-loader');
const errorMessage = document.getElementById('error-message');
const resultSection = document.getElementById('result-section');
const sentimentIcon = document.getElementById('sentiment-icon');
const sentimentLabel = document.getElementById('sentiment-label');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceText = document.getElementById('confidence-text');
const negProbBar = document.getElementById('neg-prob-bar');
const posProbBar = document.getElementById('pos-prob-bar');
const negProbValue = document.getElementById('neg-prob-value');
const posProbValue = document.getElementById('pos-prob-value');
const examplesContainer = document.getElementById('examples-container');
const historyContainer = document.getElementById('history-container');
const clearHistoryBtn = document.getElementById('clear-history-btn');

// State
let isLoading = false;
let predictionHistory = JSON.parse(localStorage.getItem('sentimentHistory')) || [];

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadExamples();
    renderHistory();
});

/**
 * Initialize event listeners
 */
function initEventListeners() {
    // Character count update
    tweetInput.addEventListener('input', updateCharCount);
    
    // Predict button click
    predictBtn.addEventListener('click', handlePredict);
    
    // Enter key to submit (Ctrl+Enter or Cmd+Enter)
    tweetInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            handlePredict();
        }
    });
    
    // Clear history button
    clearHistoryBtn.addEventListener('click', clearHistory);
}

/**
 * Update character count
 */
function updateCharCount() {
    const count = tweetInput.value.length;
    charCount.textContent = count;
    
    // Visual feedback when approaching limit
    if (count > 450) {
        charCount.style.color = '#dc3545';
    } else if (count > 400) {
        charCount.style.color = '#ffc107';
    } else {
        charCount.style.color = '#657786';
    }
}

/**
 * Handle predict button click
 */
async function handlePredict() {
    const text = tweetInput.value.trim();
    
    // Validation
    if (!text) {
        showError('Please enter some text to analyze');
        tweetInput.focus();
        return;
    }
    
    if (text.length > 500) {
        showError('Text is too long. Maximum 500 characters allowed.');
        return;
    }
    
    // Clear previous errors
    hideError();
    
    // Set loading state
    setLoading(true);
    
    try {
        const result = await predictSentiment(text);
        
        if (result.success) {
            displayResult(result);
            addToHistory(text, result);
            hideError();
        } else {
            showError(result.error || 'Prediction failed. Please try again.');
        }
    } catch (error) {
        showError('Failed to connect to the server. Please make sure the backend is running.');
        console.error('Prediction error:', error);
    } finally {
        setLoading(false);
    }
}

/**
 * Call the API to predict sentiment
 */
async function predictSentiment(text) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction request failed');
    }
    
    return await response.json();
}

/**
 * Display the prediction result
 */
function displayResult(result) {
    const isPositive = result.sentiment === 'Positive';
    const confidence = result.confidence;
    
    // Update sentiment icon and label
    sentimentIcon.textContent = isPositive ? '😊' : '😠';
    sentimentLabel.textContent = result.sentiment;
    sentimentLabel.className = `sentiment-label ${isPositive ? 'positive' : 'negative'}`;
    
    // Update confidence bar
    const sentimentDisplay = document.querySelector('.sentiment-display');
    sentimentDisplay.className = `sentiment-display ${isPositive ? 'positive' : 'negative'}`;
    
    confidenceBar.style.width = `${confidence}%`;
    confidenceBar.className = `confidence-bar ${isPositive ? 'positive' : 'negative'}`;
    confidenceText.textContent = `${confidence.toFixed(1)}% confidence`;
    
    // Update probability bars
    const negProb = result.probabilities.negative;
    const posProb = result.probabilities.positive;
    
    negProbBar.style.width = `${negProb}%`;
    posProbBar.style.width = `${posProb}%`;
    negProbValue.textContent = `${negProb.toFixed(1)}%`;
    posProbValue.textContent = `${posProb.toFixed(1)}%`;
    
    // Show result section with animation
    resultSection.classList.remove('hidden');
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Set loading state
 */
function setLoading(loading) {
    isLoading = loading;
    predictBtn.disabled = loading;
    
    if (loading) {
        btnText.textContent = 'Analyzing...';
        btnLoader.classList.remove('hidden');
    } else {
        btnText.textContent = 'Predict Sentiment';
        btnLoader.classList.add('hidden');
    }
}

/**
 * Show error message
 */
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
}

/**
 * Hide error message
 */
function hideError() {
    errorMessage.classList.add('hidden');
}

/**
 * Load example tweets from API
 */
async function loadExamples() {
    try {
        const response = await fetch(`${API_BASE_URL}/examples`);
        const data = await response.json();
        
        if (data.success) {
            renderExamples(data.examples);
        }
    } catch (error) {
        console.error('Failed to load examples:', error);
        examplesContainer.innerHTML = '<div class="example-loading">Failed to load examples</div>';
    }
}

/**
 * Render example tweets
 */
function renderExamples(examples) {
    examplesContainer.innerHTML = '';
    
    examples.forEach(example => {
        const card = document.createElement('div');
        card.className = 'example-card';
        card.innerHTML = `
            <div class="example-text">${escapeHtml(example.text)}</div>
            <span class="example-tag ${example.category}">${example.category}</span>
        `;
        
        card.addEventListener('click', () => {
            tweetInput.value = example.text;
            updateCharCount();
            tweetInput.focus();
            // Scroll to input
            document.querySelector('.input-section').scrollIntoView({ behavior: 'smooth' });
        });
        
        examplesContainer.appendChild(card);
    });
}

/**
 * Add prediction to history
 */
function addToHistory(text, result) {
    const historyItem = {
        id: Date.now(),
        text: text,
        sentiment: result.sentiment,
        confidence: result.confidence,
        timestamp: new Date().toISOString(),
    };
    
    // Add to beginning of array
    predictionHistory.unshift(historyItem);
    
    // Keep only last 20 items
    if (predictionHistory.length > 20) {
        predictionHistory = predictionHistory.slice(0, 20);
    }
    
    // Save to localStorage
    localStorage.setItem('sentimentHistory', JSON.stringify(predictionHistory));
    
    // Update UI
    renderHistory();
}

/**
 * Render prediction history
 */
function renderHistory() {
    if (predictionHistory.length === 0) {
        historyContainer.innerHTML = '<div class="empty-history">No predictions yet. Try analyzing a tweet!</div>';
        clearHistoryBtn.classList.add('hidden');
        return;
    }
    
    clearHistoryBtn.classList.remove('hidden');
    historyContainer.innerHTML = '';
    
    predictionHistory.forEach(item => {
        const isPositive = item.sentiment === 'Positive';
        const date = new Date(item.timestamp);
        const timeStr = date.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        const dateStr = date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric' 
        });
        
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.innerHTML = `
            <div class="history-sentiment-icon ${isPositive ? 'positive' : 'negative'}">
                ${isPositive ? '😊' : '😠'}
            </div>
            <div class="history-content">
                <div class="history-text">${escapeHtml(item.text)}</div>
                <div class="history-meta">
                    <span>${dateStr} at ${timeStr}</span>
                    <span class="history-confidence ${isPositive ? 'positive' : 'negative'}">
                        ${item.confidence.toFixed(1)}% ${item.sentiment}
                    </span>
                </div>
            </div>
        `;
        
        historyContainer.appendChild(historyItem);
    });
}

/**
 * Clear prediction history
 */
function clearHistory() {
    if (confirm('Are you sure you want to clear your prediction history?')) {
        predictionHistory = [];
        localStorage.removeItem('sentimentHistory');
        renderHistory();
    }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Health check - verify backend is running
 */
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        return data.status === 'healthy';
    } catch (error) {
        return false;
    }
}

// Check backend health on load
checkHealth().then(isHealthy => {
    if (!isHealthy) {
        console.warn('Backend server is not running. Please start the Flask server.');
    }
});
