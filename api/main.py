from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import spacy
import textdescriptives as td
import os
from pathlib import Path
import pandas as pd
import ast
from google.generativeai import GenerativeModel, configure
from typing import Optional, List, Dict

# Initialize FastAPI app
app = FastAPI(title="Prompt Quality Assessment API")

# Get port from environment variable or use default
PORT = int(os.getenv('PORT', 8000))

#configure gemini
configure(api_key=os.getenv('GOOGLE_API_KEY')) # set this env variable elsewhere
gemini_model = GenerativeModel('gemini-2.0-flash')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

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

class GeminiPromptRequest(BaseModel):
    text: str
    system_prompt: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

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
    
    if prob_diff < 0.3:  # Small probability differential
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

@app.get("/config")
async def get_config():
    """Endpoint to provide configuration to the frontend"""
    return JSONResponse({
        "apiPort": PORT
    })

@app.get("/")
async def root():
    return FileResponse('static/index.html') 

@app.post("/assess-gemini")
async def assess_prompt_gemini(request: GeminiPromptRequest = Body(...)):
    try:
        # Build message history for Gemini
        messages = []
        if request.history:
            for msg in request.history:
                if msg['role'] == 'user':
                    messages.append({"role": "user", "parts": [msg['text']]})
                elif msg['role'] == 'gemini':
                    messages.append({"role": "model", "parts": [msg['text']]})
        else:
            messages = [
                {"role": "user", "parts": [request.text]}
            ]
        # Always add the latest user message if not already present
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "parts": [request.text]})
        response = gemini_model.generate_content(messages)
        return {"gemini_assessment": response.text}
    except Exception as e:
        return {"gemini_assessment": f"Error: {str(e)}"}
