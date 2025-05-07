from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import spacy
import textdescriptives as td
import os
from pathlib import Path
import pandas as pd
import ast

# Initialize FastAPI app
app = FastAPI(title="Prompt Quality Assessment API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and components
MODEL_DIR = Path("../models")
model = joblib.load(MODEL_DIR / "xgb_model.joblib")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")

# Load spaCy model and add textdescriptives pipeline
try:
    nlp = spacy.load("en_core_web_sm")
    # Add textdescriptives pipeline components
    if "textdescriptives/all" not in nlp.pipe_names:
        nlp.add_pipe("textdescriptives/all")
except OSError:
    print("Downloading spacy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textdescriptives/all")

class PromptRequest(BaseModel):
    text: str

def extract_features(text: str) -> np.ndarray:
    """Extract features from the input text using spaCy and textdescriptives."""
    doc = nlp(text)
    
    # Extract features as a dictionary
    features_dict = {
        "readability": doc._.readability,
        "token_length": doc._.token_length,
        "sentence_length": doc._.sentence_length,
        "coherence": doc._.coherence,
        "information_theory": doc._.information_theory,
        "entropy": doc._.entropy,
        "perplexity": doc._.perplexity,
        "per_word_perplexity": doc._.per_word_perplexity
    }
    
    # Create a single-row DataFrame
    df = pd.DataFrame([features_dict])
    
    # Flatten dictionary columns
    features_list = []
    for col in df.columns:
        col_data = df[col]
        first_value = col_data.iloc[0]
        
        if isinstance(first_value, dict):
            # Normalize dictionary into separate columns
            normalized_df = pd.json_normalize(col_data)
            normalized_df.columns = [f"{col}_{sub_col}" for sub_col in normalized_df.columns]
            features_list.append(normalized_df)
        elif isinstance(first_value, (int, float)) or pd.isna(first_value):
            # Keep numeric columns as is
            features_list.append(col_data.to_frame())
    
    # Combine all features
    X = pd.concat(features_list, axis=1)
    print("Extracted features:", X.columns.tolist())
    print("Expected columns:", feature_columns)
    
    # Create feature vector matching training data
    feature_vector = np.zeros(len(feature_columns))
    for i, col in enumerate(feature_columns):
        if col in X.columns:
            feature_vector[i] = X[col].iloc[0]
        else:
            print(f"Missing feature: {col}")
    
    return feature_vector.reshape(1, -1)

def get_quality_assessment(probabilities) -> dict:
    """Convert model probabilities into quality assessment."""
    prob_diff = abs(probabilities[0] - probabilities[1])
    
    if prob_diff < 0.2:  # Small probability differential
        quality = "okay"
        confidence = 0.5
    else:
        quality = "good" if probabilities[1] > probabilities[0] else "bad"
        confidence = max(probabilities)
    
    return {
        "quality": quality,
        "confidence": float(confidence),
        "probabilities": {
            "bad": float(probabilities[0]),
            "good": float(probabilities[1])
        }
    }

@app.post("/assess")
async def assess_prompt(request: PromptRequest):
    try:
        # Extract features
        features = extract_features(request.text)
        
        # Get model probabilities
        probabilities = model.predict_proba(features)[0]
        
        # Get quality assessment
        assessment = get_quality_assessment(probabilities)
        
        return {
            "text": request.text,
            "assessment": assessment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 