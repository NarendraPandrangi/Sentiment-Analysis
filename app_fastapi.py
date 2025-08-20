"""
FastAPI application for serving the sentiment analysis model.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

from infer import load_model, predict_sentiment

# Initialize FastAPI app
app = FastAPI(title="Twitter Sentiment Analysis API")

# Load model at startup
try:
    vectorizer, model = load_model()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    vectorizer, model = None, None

class Tweet(BaseModel):
    text: str

class TweetBatch(BaseModel):
    tweets: List[str]

@app.get("/")
async def root():
    return {"message": "Welcome to Twitter Sentiment Analysis API"}

@app.post("/analyze")
async def analyze_sentiment(tweet: Tweet):
    if not vectorizer or not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    sentiment = predict_sentiment(tweet.text, vectorizer, model)
    return {"text": tweet.text, "sentiment": sentiment}

@app.post("/analyze-batch")
async def analyze_batch(tweets: TweetBatch):
    if not vectorizer or not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    for text in tweets.tweets:
        sentiment = predict_sentiment(text, vectorizer, model)
        results.append({"text": text, "sentiment": sentiment})
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
