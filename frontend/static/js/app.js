/**
 * Sentiment Analysis App - Frontend JavaScript
 * Handles UI interactions and API communication
 */

// API Configuration
const API_BASE_URL = 'https://sentimental-analysis-1.onrender.com';

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

// Event listeners
function initEventListeners() {
    tweetInput.addEventListener('input', updateCharCount);
    predictBtn.addEventListener('click', handlePredict);

    tweetInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            handlePredict();
        }
    });

    clearHistoryBtn.addEventListener('click', clearHistory);
}

// Character count
function updateCharCount() {
    const count = tweetInput.value.length;
    charCount.textContent = count;

    if (count > 450) {
        charCount.style.color = '#dc3545';
    } else if (count > 400) {
        charCount.style.color = '#ffc107';
    } else {
        charCount.style.color = '#657786';
    }
}

// Predict
async function handlePredict() {
    const text = tweetInput.value.trim();

    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }

    if (text.length > 500) {
        showError('Text too long (max 500 chars)');
        return;
    }

    hideError();
    setLoading(true);

    try {
        const result = await predictSentiment(text);

        if (result.success) {
            displayResult(result);
            addToHistory(text, result);
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Failed to connect to server (check Render backend)');
    } finally {
        setLoading(false);
    }
}

// 🔥 FIXED API CALL
async function predictSentiment(text) {
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
    });

    if (!response.ok) {
        throw new Error('API error');
    }

    return await response.json();
}

// Display result
function displayResult(result) {
    const isPositive = result.sentiment === 'Positive';
    const confidence = result.confidence;

    sentimentIcon.textContent = isPositive ? '😊' : '😠';
    sentimentLabel.textContent = result.sentiment;
    sentimentLabel.className = `sentiment-label ${isPositive ? 'positive' : 'negative'}`;

    confidenceBar.style.width = `${confidence}%`;
    confidenceText.textContent = `${confidence.toFixed(1)}% confidence`;

    const neg = result.probabilities.negative;
    const pos = result.probabilities.positive;

    negProbBar.style.width = `${neg}%`;
    posProbBar.style.width = `${pos}%`;

    negProbValue.textContent = `${neg.toFixed(1)}%`;
    posProbValue.textContent = `${pos.toFixed(1)}%`;

    resultSection.classList.remove('hidden');
}

// Loading state
function setLoading(loading) {
    predictBtn.disabled = loading;

    if (loading) {
        btnText.textContent = 'Analyzing...';
        btnLoader.classList.remove('hidden');
    } else {
        btnText.textContent = 'Predict Sentiment';
        btnLoader.classList.add('hidden');
    }
}

// Error handling
function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}

// 🔥 FIXED examples API
async function loadExamples() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/examples`);
        const data = await res.json();

        if (data.success) {
            renderExamples(data.examples);
        }
    } catch {
        examplesContainer.innerHTML = 'Failed to load examples';
    }
}

// Render examples
function renderExamples(examples) {
    examplesContainer.innerHTML = '';

    examples.forEach(ex => {
        const div = document.createElement('div');
        div.className = 'example-card';
        div.innerText = ex.text;

        div.onclick = () => {
            tweetInput.value = ex.text;
            updateCharCount();
        };

        examplesContainer.appendChild(div);
    });
}

// History
function addToHistory(text, result) {
    predictionHistory.unshift({
        text,
        sentiment: result.sentiment,
        confidence: result.confidence,
    });

    localStorage.setItem('sentimentHistory', JSON.stringify(predictionHistory));
    renderHistory();
}

function renderHistory() {
    if (!predictionHistory.length) {
        historyContainer.innerHTML = 'No history yet';
        return;
    }

    historyContainer.innerHTML = '';

    predictionHistory.forEach(item => {
        const div = document.createElement('div');
        div.innerText = `${item.sentiment} (${item.confidence}%) - ${item.text}`;
        historyContainer.appendChild(div);
    });
}

function clearHistory() {
    predictionHistory = [];
    localStorage.removeItem('sentimentHistory');
    renderHistory();
}

// 🔥 FIXED health API
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/health`);
        const data = await res.json();
        return data.status === 'healthy';
    } catch {
        return false;
    }
}

checkHealth().then(ok => {
    if (!ok) {
        console.warn('Backend not reachable');
    }
});