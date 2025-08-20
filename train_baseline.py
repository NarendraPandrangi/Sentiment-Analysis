"""
Script for training the sentiment analysis model using Logistic Regression.
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def load_and_prepare_data():
    """Load the processed tweets and prepare for training"""
    # Load the processed dataset
    processed_file = os.path.join('data', 'processed', 'processed_tweets.csv')
    
    if not os.path.exists(processed_file):
        raise FileNotFoundError(
            f"Processed data not found at {processed_file}. "
            "Please run preprocess.py first."
        )
    
    # Load data
    print("Loading processed data...")
    df = pd.read_csv(processed_file)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print("\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    
    return df

def save_model(model, filename):
    """Save a model or vectorizer to disk"""
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {filename} to models directory")

def train_model(X_text, y, vectorizer):
    """Train and evaluate the Logistic Regression model"""
    # Transform text data to TF-IDF features
    print("\nCreating TF-IDF features...")
    X = vectorizer.fit_transform(X_text)
    print(f"Feature matrix shape: {X.shape}")
    
    # Split the data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Create and train the model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        n_jobs=-1,
        random_state=42  # For reproducibility
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    return model, vectorizer

def save_model(model, filename):
    """Save a model or vectorizer to disk"""
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {filename} to models directory")

def main():
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Prepare features and target
    X = df['processed_text'].fillna('')  # Replace NaN with empty string
    y = df['sentiment']
    
    # Create TF-IDF vectorizer
    print("\nInitializing TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,    # Limit features to top 5000 terms
        min_df=5,            # Ignore terms that appear in less than 5 documents
        max_df=0.95,         # Ignore terms that appear in more than 95% of documents
        ngram_range=(1, 2),  # Include both unigrams and bigrams
        stop_words='english', # Remove English stop words
        strip_accents='unicode',  # Remove accents
        lowercase=True  # Convert to lowercase
    )
    
    # Train model and get the trained vectorizer back
    print("\nTraining and evaluating model...")
    model, vectorizer = train_model(X, y, vectorizer)
    
    # Save model and vectorizer
    print("\nSaving model and vectorizer...")
    save_model(model, 'sentiment_model.pkl')
    save_model(vectorizer, 'tfidf_vectorizer.pkl')

if __name__ == "__main__":
    main()
