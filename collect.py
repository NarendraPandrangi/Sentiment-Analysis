"""
Script for loading tweets from the training dataset
"""
import os
import pandas as pd

def load_dataset():
    """
    Load tweets from the training dataset CSV file
    """
    try:
        # Path to the dataset
        dataset_path = os.path.join('data', 'raw', 'twitter_training.csv')
        
        # Read the CSV file with proper column names
        df = pd.read_csv(dataset_path, names=['id', 'category', 'sentiment', 'text'])
        
        print("\nDataset Statistics:")
        print(f"Total tweets: {len(df)}")
        print("\nSentiment distribution:")
        print(df['sentiment'].value_counts())
        print("\nCategories distribution:")
        print(df['category'].value_counts())
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def main():
    # Load the dataset
    df = load_dataset()
    
    if df is not None:
        print("\nDataset preview:")
        print(df.head())
        
        # Optional: Save a processed version if needed
        processed_path = os.path.join('data', 'raw', 'processed_training.csv')
        df.to_csv(processed_path, index=False)
        print(f"\nDataset processed and saved to: {processed_path}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
