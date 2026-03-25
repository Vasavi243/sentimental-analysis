"""
Sentiment Analysis Model Training Script
Uses Naive Bayes with TF-IDF vectorization
"""

import pickle
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sample Twitter sentiment dataset
# Format: (text, sentiment) where sentiment is 1 (positive) or 0 (negative)
SAMPLE_TWEETS = [
    # Positive tweets (1)
    ("I love this new phone! It's amazing! 😊", 1),
    ("Just had the best day ever! So happy!", 1),
    ("This restaurant is fantastic! Great food and service!", 1),
    ("Feeling wonderful today! Life is good!", 1),
    ("The movie was absolutely incredible! Highly recommend!", 1),
    ("So grateful for my friends and family! ❤️", 1),
    ("Just got promoted at work! Dreams do come true!", 1),
    ("Beautiful weather today! Perfect for a walk!", 1),
    ("This app is so helpful! Makes my life easier!", 1),
    ("Best vacation ever! Can't wait to come back!", 1),
    ("So excited for the weekend! Going to be fun!", 1),
    ("Just finished a great workout! Feeling strong!", 1),
    ("The concert was amazing! Best night of my life!", 1),
    ("Love my new job! Great team and atmosphere!", 1),
    ("This book is so good! Can't put it down!", 1),
    ("Had a delicious dinner! Everything was perfect!", 1),
    ("So proud of my daughter! She did amazing!", 1),
    ("The sunrise this morning was breathtaking!", 1),
    ("Just adopted a puppy! My heart is full! 🐶", 1),
    ("This coffee shop has the best latte in town!", 1),
    ("Feeling motivated and ready to conquer the day!", 1),
    ("The new update is fantastic! So many great features!", 1),
    ("Just reached my fitness goal! Hard work pays off!", 1),
    ("My team won the championship! So proud of them!", 1),
    ("This gift made my day! Thank you so much!", 1),
    ("The beach was beautiful today! Perfect relaxation!", 1),
    ("Just got engaged! I'm the happiest person alive!", 1),
    ("This course is excellent! Learning so much!", 1),
    ("The customer service here is outstanding!", 1),
    ("So happy with my purchase! Exceeded expectations!", 1),
    ("Just ran my first marathon! What an achievement!", 1),
    ("The flowers in my garden are blooming beautifully!", 1),
    ("This song always puts me in a great mood!", 1),
    ("Had the best sleep last night! Feeling refreshed!", 1),
    ("My cat is so cute! She always makes me smile!", 1),
    ("The presentation went perfectly! So relieved!", 1),
    ("Just booked my dream vacation! Can't wait!", 1),
    ("This skincare product is a game changer!", 1),
    ("So thankful for all the birthday wishes!", 1),
    ("The new restaurant in town is incredible!", 1),
    ("Just finished my degree! Officially graduated!", 1),
    ("This place has such positive vibes! Love it!", 1),
    ("My garden is thriving! So rewarding to see!", 1),
    ("The kids did amazing in their recital! So proud!", 1),
    ("Just got a new bike! Ready for adventures!", 1),
    ("This podcast is so inspiring! Highly recommend!", 1),
    ("The sunset tonight was absolutely gorgeous!", 1),
    ("So happy to be home! Missed my family!", 1),
    ("This workout program is changing my life!", 1),
    ("Just made new friends at the meetup! Great people!", 1),
    ("The museum exhibit was fascinating! Learned so much!", 1),
    
    # Negative tweets (0)
    ("This is the worst day ever. Everything is going wrong. 😠", 0),
    ("I hate this weather. It's so depressing.", 0),
    ("Terrible service at this restaurant. Never coming back!", 0),
    ("Feeling so sad and lonely today.", 0),
    ("The movie was horrible. Complete waste of time!", 0),
    ("So frustrated with this app. It keeps crashing!", 0),
    ("Just got fired from my job. Life is unfair.", 0),
    ("This traffic is unbearable! Going to be late again!", 0),
    ("The food was cold and tasteless. Very disappointed.", 0),
    ("Worst customer service experience ever! So angry!", 0),
    ("My phone broke again! So tired of this!", 0),
    ("Feeling sick and miserable today.", 0),
    ("The hotel room was dirty and disgusting!", 0),
    ("So stressed about this deadline! Can't handle it!", 0),
    ("This book is so boring! Can't finish it!", 0),
    ("The flight was delayed for 5 hours! Unacceptable!", 0),
    ("Just lost my wallet. What a nightmare!", 0),
    ("The wifi here is terrible! Can't get any work done!", 0),
    ("So disappointed with my purchase. Total ripoff!", 0),
    ("This headache won't go away! So painful!", 0),
    ("The concert was a disaster! Sound was awful!", 0),
    ("Feeling so anxious about tomorrow. Can't sleep.", 0),
    ("The repair shop damaged my car even more!", 0),
    ("So tired of all this rain! When will it stop!", 0),
    ("This laptop is so slow! Can't stand it anymore!", 0),
    ("The doctor's office made me wait 2 hours! Rude staff!", 0),
    ("Just got scammed online! Lost so much money!", 0),
    ("The gym is always packed! Can't get a machine!", 0),
    ("So angry at my neighbor! Loud music all night!", 0),
    ("This product broke after one day! Poor quality!", 0),
    ("The exam was so hard! Think I failed!", 0),
    ("Feeling so overwhelmed with everything right now.", 0),
    ("The delivery was late and the food was cold!", 0),
    ("So annoyed by all these spam calls!", 0),
    ("This software is so buggy! Can't use it!", 0),
    ("Just got a parking ticket! So unfair!", 0),
    ("The air conditioning is broken! It's so hot!", 0),
    ("So sad to hear about the accident. Terrible news.", 0),
    ("This coffee tastes burnt! Can't drink this!", 0),
    ("The meeting was a complete waste of time!", 0),
    ("Feeling so exhausted! Need a break from everything!", 0),
    ("The store was out of stock! Drove all this way!", 0),
    ("So frustrated with my internet connection!", 0),
    ("This movie theater is filthy! Never coming back!", 0),
    ("Just got rejected from my dream job. Heartbroken.", 0),
    ("The noise from construction is driving me crazy!", 0),
    ("So disappointed in myself today. Failed again.", 0),
    ("This airline lost my luggage! What a mess!", 0),
    ("Feeling so isolated lately. Need friends.", 0),
    ("The battery on this phone dies so fast!", 0),
    ("So angry about the price increase! Greedy company!", 0),
]


def preprocess_text(text):
    """
    Preprocess text for sentiment analysis:
    - Convert to lowercase
    - Remove URLs
    - Remove mentions (@username)
    - Remove hashtags symbol (keep text)
    - Remove punctuation
    - Remove numbers
    - Remove extra whitespace
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


# For pickling compatibility
import __main__
__main__.preprocess_text = preprocess_text


def train_and_save_model():
    """Train the sentiment analysis model and save it to disk."""
    print("Preparing dataset...")
    
    # Separate texts and labels
    texts = [tweet[0] for tweet in SAMPLE_TWEETS]
    labels = [tweet[1] for tweet in SAMPLE_TWEETS]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create a pipeline with TF-IDF and Naive Bayes
    # Using English stop words
    stop_words = set(stopwords.words('english'))
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=preprocess_text,
            stop_words=list(stop_words),
            max_features=5000,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=1,
            max_df=0.95
        )),
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    
    print("Training model...")
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Save the model
    model_path = 'sentiment_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"\nModel saved to {model_path}")
    
    return pipeline


def load_model():
    """Load the trained model from disk."""
    model_path = 'sentiment_model.pkl'
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print("Model not found. Training new model...")
        return train_and_save_model()


def predict_sentiment(text, model=None):
    """
    Predict sentiment of a given text.
    Returns: (sentiment_label, confidence)
    """
    if model is None:
        model = load_model()
    
    # Get prediction probabilities
    proba = model.predict_proba([text])[0]
    prediction = model.predict([text])[0]
    
    # Get confidence (probability of predicted class)
    confidence = proba[prediction]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return sentiment, confidence, proba.tolist()


if __name__ == "__main__":
    # Train and save the model
    model = train_and_save_model()
    
    # Test with some examples
    print("\n" + "="*50)
    print("Testing the model with sample tweets:")
    print("="*50)
    
    test_tweets = [
        "I absolutely love this product! Best purchase ever!",
        "This is terrible. I want my money back!",
        "Just had an amazing dinner with friends!",
        "Worst experience ever. Never again!"
    ]
    
    for tweet in test_tweets:
        sentiment, confidence, proba = predict_sentiment(tweet, model)
        print(f"\nTweet: {tweet}")
        print(f"Sentiment: {sentiment} ({confidence:.1%} confidence)")
        print(f"Probabilities - Negative: {proba[0]:.3f}, Positive: {proba[1]:.3f}")
