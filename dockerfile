FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install gdown torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# Copy application
COPY . .

# Download model files during build
RUN wget -q https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml && \
    gdown -q 1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt -O wavtokenizer_large_speech_320_24k.ckpt

# Runtime configuration
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cuda
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]