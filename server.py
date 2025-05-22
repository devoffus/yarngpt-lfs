from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torchaudio
from transformers import AutoModelForCausalLM
from yarngpt.audiotokenizer import AudioTokenizerV2
import io
import base64
import torch
from typing import Dict, List, Optional
import requests
import subprocess
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YarnGPT2b Speech API",
    description="Generate synthetic speech from text using YarnGPT2b and AudioTokenizerV2.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SpeechRequest(BaseModel):
    text: str = Field(..., example="Hello, this is a test.", min_length=1, max_length=1000,
                     description="Input text to convert to speech")
    lang: str = Field(default="english", example="english", 
                     description="Language of the text (english/yoruba/igbo/hausa)")
    speaker: str = Field(default="jude", example="jude",
                        description="Speaker identity used for voice synthesis")

class VoiceInfo(BaseModel):
    id: str
    description: str

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

# Available voices organized by language
AVAILABLE_VOICES = {
    "english": [
        {"id": "idera", "description": "Female Nigerian English voice"},
        {"id": "jude", "description": "Male Nigerian English voice"},
        {"id": "zainab", "description": "Female Nigerian English voice"},
        {"id": "chinenye", "description": "Female Nigerian English voice"},
        {"id": "emma", "description": "Male Nigerian English voice"},
    ],
    "yoruba": [
        {"id": "yoruba_male1", "description": "Male Yoruba voice"},
        {"id": "yoruba_male2", "description": "Male Yoruba voice"},
        {"id": "yoruba_female1", "description": "Female Yoruba voice"},
        {"id": "yoruba_female2", "description": "Female Yoruba voice"},
    ],
    "igbo": [
        {"id": "igbo_male1", "description": "Male Igbo voice"},
        {"id": "igbo_male2", "description": "Male Igbo voice"},
        {"id": "igbo_female1", "description": "Female Igbo voice"},
        {"id": "igbo_female2", "description": "Female Igbo voice"},
    ],
    "hausa": [
        {"id": "hausa_male1", "description": "Male Hausa voice"},
        {"id": "hausa_male2", "description": "Male Hausa voice"},
        {"id": "hausa_female1", "description": "Female Hausa voice"},
        {"id": "hausa_female2", "description": "Female Hausa voice"},
    ]
}
# Global model variables
model = None
audio_tokenizer = None

def download_file(url: str, path: str):
    """Download a file from a URL with progress tracking"""
    try:
        if "drive.google.com" in url:
            try:
                import gdown
                gdown.download(url, path, quiet=False)
            except ImportError:
                subprocess.run(["wget", "-O", path, url], check=True)
        else:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        logger.info(f"Downloaded {path} successfully")
    except Exception as e:
        logger.error(f"Failed to download {path}: {str(e)}")
        raise

def initialize_model():
    """Initialize the model and tokenizer"""
    global model, audio_tokenizer
    
    # Download required files if missing
    required_files = {
        "config": {
            "url": "https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
            "path": "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        },
        "model": {
            "url": "https://drive.google.com/uc?id=1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt",
            "path": "wavtokenizer_large_speech_320_24k.ckpt"
        }
    }

    for file_info in required_files.values():
        if not os.path.exists(file_info["path"]):
            download_file(file_info["url"], file_info["path"])

    # Initialize components
    try:
        logger.info("Initializing audio tokenizer...")
        audio_tokenizer = AudioTokenizerV2(
            "saheedniyi/YarnGPT2b",
            required_files["model"]["path"],
            required_files["config"]["path"]
        )

        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "saheedniyi/YarnGPT2b",
            torch_dtype="auto"
        ).to(audio_tokenizer.device)

        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the application starts"""
    try:
        initialize_model()
    except Exception as e:
        logger.critical(f"Failed to initialize model: {str(e)}")
        # Consider exiting if model fails to load
        # import sys; sys.exit(1)

@app.post(
    "/generate-speech",
    response_model=Dict[str, str],
    summary="Generate speech from text",
    response_description="Base64 encoded WAV audio",
    tags=["Speech Generation"]
)
async def generate_speech(request: SpeechRequest):
    """Convert text to speech using the specified voice"""
    try:
        # Validate input
        if request.lang not in AVAILABLE_VOICES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language. Available: {list(AVAILABLE_VOICES.keys())}"
            )
            
        if not any(v.id == request.speaker for v in AVAILABLE_VOICES[request.lang]):
            available = [v.id for v in AVAILABLE_VOICES[request.lang]]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid speaker for {request.lang}. Available: {available}"
            )

        # Generate speech
        prompt = audio_tokenizer.create_prompt(
            request.text,
            lang=request.lang,
            speaker_name=request.speaker
        )
        
        input_ids = audio_tokenizer.tokenize_prompt(prompt)
        
        output = model.generate(
            input_ids=input_ids,
            temperature=0.1,
            repetition_penalty=1.1,
            max_length=4000,
        )
        
        # Process and encode audio
        codes = audio_tokenizer.get_codes(output)
        audio = audio_tokenizer.get_audio(codes)

        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, sample_rate=24000, format="wav")
        
        return {
            "audio": base64.b64encode(buffer.getvalue()).decode('utf-8'),
            "format": "wav",
            "sample_rate": 24000
        }

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/voices",
    response_model=Dict[str, List[VoiceInfo]],
    summary="List available voices",
    tags=["Voice Management"]
)
async def list_voices(language: Optional[str] = None):
    """Get available voices, optionally filtered by language"""
    try:
        if language:
            if language not in AVAILABLE_VOICES:
                raise HTTPException(
                    status_code=404,
                    detail=f"Language not found. Available: {list(AVAILABLE_VOICES.keys())}"
                )
            return {language: AVAILABLE_VOICES[language]}
        return AVAILABLE_VOICES
    except Exception as e:
        logger.error(f"Voice listing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/health",
    response_model=HealthCheck,
    summary="Service health check",
    tags=["Monitoring"]
)
async def health_check():
    """Check if the service is healthy"""
    return {
        "status": "healthy" if model and audio_tokenizer else "degraded",
        "model_loaded": bool(model and audio_tokenizer),
        "timestamp": datetime.now().isoformat()
    }