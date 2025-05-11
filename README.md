# Prompt Quality Assessment Tool

A web-based tool that assesses the quality of prompts using machine learning and natural language processing techniques. The tool provides real-time feedback on prompt quality with a visual indicator and detailed statistics.

## Features

- Real-time prompt quality assessment
- Visual quality indicator bar
- Confidence scores and probabilities
- Support for three quality levels: Good, Okay, and Bad
- REST API for programmatic access
- **Multi-turn chat with Google Gemini (Gemini 2.0 Flash) integration**
- Markdown rendering for Gemini responses

## Technical Stack

### Backend (FastAPI)
- Python 3.11
- FastAPI for API endpoints
- XGBoost for machine learning
- spaCy and TextDescriptives for NLP features:
  - Readability metrics (Flesch Reading Ease, Flesch-Kincaid Grade, SMOG, Gunning-Fog, etc.)
  - Information theory metrics (entropy, perplexity, per-word perplexity)
  - Token and sentence length statistics
  - Text coherence measures

### Frontend
- Pure HTML/CSS/JavaScript
- Real-time assessment with debouncing
- Responsive design
- Modern UI with quality indicator bar

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/renwah/cs568_final_project.git
   cd cs568_final_project
   ```

2. Create and activate a Conda environment:
   ```bash
   conda create -n prompt-quality python=3.11
   conda activate prompt-quality
   ```

   Alternative setup without Conda (not recommended):
   ```bash
   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   # Upgrade pip (recommended)
   python -m pip install --upgrade pip
   ```

3. Install dependencies:
   ```bash
   cd api
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

6. **Set up your Gemini API key:**
   - Sign up for access to Google Gemini (https://aistudio.google.com/app/apikey) and obtain your API key.
   - In the `api` directory, create a `.env` file (you can copy from the provided `.env.example`):
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and add your Gemini API key:
     ```env
     GOOGLE_API_KEY=your-gemini-api-key-here
     ```
   - (Optional) You can also set the `PORT` variable in `.env` if you want to run the server on a different port.

7. **Run the app and chat with Gemini!**
   - The web interface now supports multi-turn chat with Gemini, with context-aware responses and markdown rendering.
   - Your prompt quality will still be assessed in real time as you type.

## API Endpoints

### POST /assess
Assesses the quality of a prompt.

Request body:
```json
{
    "text": "Your prompt text here"
}
```

Response:
```json
{
    "text": "Your prompt text here",
    "assessment": {
        "quality": "good|okay|bad",
        "confidence": 0.95,
        "probabilities": {
            "good": 0.95,
            "bad": 0.05
        }
    }
}
```

### GET /health
Health check endpoint.

Response:
```json
{
    "status": "healthy"
}
```

## Model Details

The quality assessment is performed using an XGBoost classifier trained on a dataset of labeled prompts. The model uses various text features including:
- Readability metrics
- Information theory metrics
- Token and sentence statistics
- Coherence measures

The model classifies prompts as:
- "Good" - Well-structured, clear prompts
- "Okay" - Prompts with room for improvement
- "Bad" - Unclear or poorly structured prompts

## Authors
[Fill in author names here]

## License
[Add license information]
        