"""
Script for preprocessing and cleaning the raw tweets data.
"""
import re
import os
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
try:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK data downloaded successfully!")
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")

# Download required NLTK data
try:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK data downloaded successfully!")
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")

def clean_text(text):
    """Clean and preprocess text data"""
    try:
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions (@user)
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        
        # Remove hashtags while keeping the text
        text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
        
        # Remove RT (retweet) indicator
        text = re.sub(r'^rt[\s]+', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
    except Exception as e:
        print(f"Error in clean_text: {str(e)}")
        return text

def tokenize_and_lemmatize(text):
    """Tokenize and lemmatize text"""
    try:
        # Simple tokenization (split by whitespace)
        tokens = text.split()
        
        # Remove stopwords
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                     'to', 'was', 'were', 'will', 'with', 'amp', 'rt', 'via'}
        
        # Filter tokens
        tokens = [
            token.lower() for token in tokens 
            if token.lower() not in stop_words 
            and len(token) > 2  # Remove very short words
            and not token.startswith(('@', '#', 'http'))  # Remove any remaining mentions/hashtags/urls
        ]
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in tokenize_and_lemmatize: {str(e)}")
        return text

def preprocess_tweets(df):
    """Clean and preprocess tweets"""
    print("Starting text preprocessing...")
    
    # Create a copy of the dataframe to avoid modifying the original
    processed_df = df.copy()
    
    # Clean text
    print("Cleaning text...")
    processed_df['cleaned_text'] = processed_df['text'].apply(clean_text)
    
    # Tokenize and lemmatize
    print("Tokenizing and lemmatizing...")
    processed_df['processed_text'] = processed_df['cleaned_text'].apply(tokenize_and_lemmatize)
    
    return processed_df

def main():
    # Load the dataset from raw data directory
    input_file = os.path.join('data', 'raw', 'twitter_training.csv')
    
    if not os.path.exists(input_file):
        print(f"Error: Dataset not found at {input_file}")
        print("Please make sure the twitter_training.csv file is in the data/raw directory.")
        return
    
    try:
        print("Reading the dataset...")
        # Read the CSV file with proper column names
        df = pd.read_csv(input_file, names=['id', 'category', 'sentiment', 'text'])
        print(f"Successfully loaded {len(df)} tweets")
        
        # Process tweets
        processed_df = preprocess_tweets(df)
        
        # Create processed directory if it doesn't exist
        os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
        
        # Save processed data
        output_file = os.path.join('data', 'processed', 'processed_tweets.csv')
        processed_df.to_csv(output_file, index=False)
        print(f'\nSaved processed tweets to {output_file}')
        
        # Print statistics and preview
        print("\nProcessed Data Preview:")
        print("-" * 80)
        print("\nOriginal text:")
        print(processed_df['text'].head(2))
        print("\nCleaned text:")
        print(processed_df['cleaned_text'].head(2))
        print("\nProcessed text:")
        print(processed_df['processed_text'].head(2))
        
        print("\nDataset Statistics:")
        print("-" * 80)
        print(f"Total tweets processed: {len(processed_df)}")
        print("\nSentiment distribution:")
        print(processed_df['sentiment'].value_counts())
        print("\nCategory distribution:")
        print(processed_df['category'].value_counts())
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
