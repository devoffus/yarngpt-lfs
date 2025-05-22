from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torchaudio
from transformers import AutoModelForCausalLM
from yarngpt.audiotokenizer import AudioTokenizerV2
import io
import base64
import torch
from typing import Dict, List

app = FastAPI(
    title="YarnGPT2b Speech API",
    description="Generate synthetic speech from text using YarnGPT2b and AudioTokenizerV2.",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
class SpeechRequest(BaseModel):
    text: str = Field(..., example="Hello, this is a test.", description="Input text to convert to speech")
    lang: str = Field(default="english", example="english", description="Language of the text")
    speaker: str = Field(default="jude", example="jude", description="Speaker identity used for voice synthesis")

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

# Initialize model
tokenizer_path = "saheedniyi/YarnGPT2b"
wav_tokenizer_config_path = "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
wav_tokenizer_model_path = "wavtokenizer_large_speech_320_24k.ckpt"

try:
    print("Initializing audio tokenizer...")
    audio_tokenizer = AudioTokenizerV2(
        tokenizer_path, 
        wav_tokenizer_model_path, 
        wav_tokenizer_config_path
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        tokenizer_path,
        torch_dtype="auto"
    ).to(audio_tokenizer.device)

    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise e

@app.post(
    "/generate-speech",
    summary="Generate speech from text",
    response_description="Base64 encoded WAV audio",
    tags=["Speech Generation"]
)
async def generate_speech(request: SpeechRequest):
    """
    Converts input text into a synthetic speech waveform and returns it as a Base64-encoded WAV file.
    """
    try:
        print(f"Generating speech for: {request.text[:50]}...")
        
        # Validate voice exists for the requested language
        if request.lang not in AVAILABLE_VOICES:
            raise HTTPException(status_code=400, detail=f"Language '{request.lang}' not supported")
            
        voice_ids = [v["id"] for v in AVAILABLE_VOICES[request.lang]]
        if request.speaker not in voice_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Speaker '{request.speaker}' not available for language '{request.lang}'. Available voices: {voice_ids}"
            )
        
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
        
        codes = audio_tokenizer.get_codes(output)
        audio = audio_tokenizer.get_audio(codes)

        # Save to bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, sample_rate=24000, format="wav")
        buffer.seek(0)

        encoded_audio = base64.b64encode(buffer.read()).decode('utf-8')

        return {"audio": encoded_audio}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/voices",
    summary="List available voices",
    response_description="Dictionary of available voices by language",
    tags=["Voice Management"]
)
async def list_voices(language: str = None):
    """
    Returns a dictionary of available voices organized by language.
    If a language parameter is provided, returns only voices for that language.
    """
    try:
        if language:
            if language not in AVAILABLE_VOICES:
                raise HTTPException(status_code=404, detail=f"Language '{language}' not found")
            return {language: AVAILABLE_VOICES[language]}
        return AVAILABLE_VOICES
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/health",
    summary="Health check",
    response_description="Service status",
    tags=["Monitoring"]
)
async def health_check():
    """Returns the health status of the API."""
    return {"status": "healthy"}