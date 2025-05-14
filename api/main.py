from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
from pathlib import Path
from google.generativeai import GenerativeModel, configure
from typing import Optional, List, Dict

# Initialize FastAPI app
app = FastAPI(title="Gemini Chat Interface")

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

class GeminiPromptRequest(BaseModel):
    text: str
    system_prompt: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

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
