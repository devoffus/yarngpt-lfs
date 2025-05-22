## API Endpoints

### Text-to-Speech
- `POST /generate-speech` - Convert text to speech
- `GET /voices` - List available voices
- `GET /voices?language=english` - List only English voices

<!-- ### Training
- `POST /jobs/train` - Start fine-tuning job
- `GET /jobs/{job_id}` - Check training status -->

### System
- `GET /health` - Service health check

## Quick Start

### Prerequisites
- Python 3.8+
- GPU with CUDA (recommended)
- [Docker](https://docs.docker.com/get-docker/) (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yarngpt-api.git
cd yarngpt-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download model files:
```bash
wget https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml
gdown 1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt -O wavtokenizer_large_speech_320_24k.ckpt
```

### Running the API

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Access the API documentation at: http://localhost:8000/docs

### Docker Deployment

```bash
docker build -t yarngpt-api .
docker run -p 8000:8000 yarngpt-api
```

## Usage Examples

### Generate Speech

```bash
curl -X POST "http://localhost:8000/generate-speech" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world", "lang":"english", "speaker":"jude"}'
```

### Start Fine-Tuning Job

```bash
curl -X POST "http://localhost:8000/jobs/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "./data/training_samples.json",
    "epochs": 3,
    "batch_size": 4
  }'
```

### Check Job Status

```bash
curl "http://localhost:8000/jobs/YOUR_JOB_ID"
```

## Dataset Format

For fine-tuning, provide a JSON file with the following structure:

```json
{
  "text": [
    "First training sample",
    "Second training sample",
    "..."
  ]
}
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Base model path | `saheedniyi/YarnGPT2b` |
| `DEVICE` | Computation device | `cuda` if available, else `cpu` |
| `MAX_AUDIO_LENGTH` | Maximum audio length (tokens) | `4000` |

## License

MIT License

## Support

For issues and feature requests, please [open an issue](https://github.com/yourusername/yarngpt-api/issues).

---

**Note**: For production deployment, consider adding:
- Authentication
- Rate limiting
- Persistent job storage (Redis/Database)
- Load balancing
