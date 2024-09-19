from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import uvicorn
import re

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize the FastAPI app
app = FastAPI()

# Define the review input model
class Review(BaseModel):
    product_id: str
    review_id: str
    review_text: str
    expected_sentiment: str = None  # Optional field for testing purposes

# Define the response model
class SentimentResponse(BaseModel):
    review_text: str
    sentiment: str
    confidence: float

# Utility function to split text into segments
def split_text(review_text: str) -> List[str]:
    # Split text based on periods and commas
    segments = re.split(r'[.,]', review_text)
    # Remove leading/trailing whitespace
    segments = [seg.strip() for seg in segments if seg.strip()]
    return segments

# Utility function for sentiment analysis
def analyze_sentiment(review_text: str) -> Dict[str, float]:
    sentiment_scores = sia.polarity_scores(review_text)
    compound_score = sentiment_scores['compound']
    
    # Classify sentiment based on compound score
    if compound_score >= 0.05:
        return {"sentiment": "positive", "confidence": sentiment_scores['pos']}
    elif compound_score <= -0.05:
        return {"sentiment": "negative", "confidence": sentiment_scores['neg']}
    else:
        return {"sentiment": "neutral", "confidence": sentiment_scores['neu']}

# Function to determine overall sentiment from a list of results
def determine_overall_sentiment(results: List[Dict[str, str]]) -> str:
    positive_count = sum(1 for res in results if res["sentiment"] == "positive")
    negative_count = sum(1 for res in results if res["sentiment"] == "negative")
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

# Function to calculate overall confidence from a list of results
def calculate_overall_confidence(results: List[Dict[str, float]]) -> float:
    if not results:
        return 0.0

    # Calculate the average confidence score
    total_confidence = sum(res["confidence"] for res in results)
    average_confidence = total_confidence / len(results)
    
    return average_confidence

# API Endpoint to analyze a single review
@app.post("/analyze_review", response_model=SentimentResponse)
async def analyze_review(review: Review):
    try:
        segments = split_text(review.review_text)
        results = []
        for segment in segments:
            sentiment = analyze_sentiment(segment)
            results.append({
                "segment": segment,
                "sentiment": sentiment["sentiment"],
                "confidence": sentiment["confidence"]
            })
        overall_sentiment = determine_overall_sentiment(results)
        overall_confidence = calculate_overall_confidence(results)
        return {
            "review_text": review.review_text,
            "sentiment": overall_sentiment,
            "confidence": overall_confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API Endpoint to analyze a list of reviews
@app.post("/analyze_reviews", response_model=List[SentimentResponse])
async def analyze_reviews(reviews: List[Review]):
    results = []
    for review in reviews:
        sentiment = analyze_sentiment(review.review_text)
        results.append({
            "review_text": review.review_text,
            "sentiment": sentiment["sentiment"],
            "confidence": sentiment["confidence"]
        })
    return results

# Run the FastAPI application using Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
