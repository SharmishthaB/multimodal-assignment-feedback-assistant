# Setup Guide

## Prerequisites

- **Python 3.10+**
- **CUDA 11.8+** (for GPU support; CPU-only also works but slower)
- **Git**
- **16GB+ RAM** (or 12GB with 4-bit quantization)
- **Disk space**: 30GB+ (for models + vector DB)

---

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/SharmishthaB/genai-multimodal-assignment-feedback-assistant.git
cd genai-multimodal-assignment-feedback-assistant
```

### 2. Create Virtual Environment

**macOS/Linux:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### GPU Support (CUDA 11.8)

If you have NVIDIA GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### macOS with Apple Silicon

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 4. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your settings
nano .env  # or use your editor
```

Key settings:
- `DEVICE=cuda` (for GPU) or `cpu`
- `LOAD_IN_4BIT=true` (saves memory)
- `LLM_MODEL_PATH` (path to Phi-3 model)

### 5. Download Models

The app automatically downloads models from Hugging Face on first run. To pre-download:

```bash
# Set Hugging Face cache
export HF_HOME=./models/hf_cache

# Download Vision Model
python -c "from transformers import AutoModelForVision2Seq; AutoModelForVision2Seq.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')"

# Download Embedding Model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

### 6. Build Vector Database

```bash
python build_chromadb.py
```

This creates ChromaDB index from `data/rubric/*.md`.

### 7. Run Application

```bash
python app.py
```

Visit: **http://localhost:7860**

---

## Troubleshooting

### Out of Memory (OOM)

Enable 4-bit quantization in `.env`:
```bash
LOAD_IN_4BIT=true
DEVICE=cuda
```

Or reduce to CPU (slower):
```bash
DEVICE=cpu
```

### CUDA Not Detected

Check PyTorch installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, reinstall PyTorch with correct CUDA version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U
```

### Model Download Stuck

Set Hugging Face cache and try again:
```bash
export HF_HOME=./models/hf_cache
python build_chromadb.py
```

### ChromaDB Index Error

Rebuild from scratch:
```bash
rm -rf data/chromadb
python build_chromadb.py
```

---

## Docker Setup

### Build Image

```bash
docker build -t mafa:latest .
```

### Run Container (CPU)

```bash
docker run -it -p 7860:7860 mafa:latest
```

### Run Container (GPU)

```bash
docker run -it --gpus all -p 7860:7860 mafa:latest
```

Visit: **http://localhost:7860**

---

## Docker Compose (Multi-Service)

```bash
docker-compose up --build
```

This optionally includes:
- Main MAFA app (port 7860)
- PostgreSQL for persistence (port 5432)
- Redis for caching (port 6379)

See `docker-compose.yml` for configuration.

---

## Development Setup

For contributing to MAFA:

```bash
# Install dev dependencies
pip install -r requirements.txt[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Format code
black src/ app.py

# Lint
flake8 src/ app.py

# Type check
mypy src/ app.py
```

---

## Kaggle Environment

If running on Kaggle notebooks:

```python
!pip install -q gradio langgraph chromadb unsloth "transformers>=4.40" accelerate

import os
os.environ["UNSLOTH_COMPLIED_CACHE"] = "/kaggle/working/unsloth_cache"
os.makedirs("/kaggle/working/unsloth_cache", exist_ok=True)

# Run app
exec(open("app.py").read())
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_PATH` | `./models/phi_3` | Path to fine-tuned Phi-3 model |
| `CHROMADB_PATH` | `./data/chromadb` | Path to ChromaDB index |
| `DEVICE` | `cuda` if available else `cpu` | PyTorch device |
| `LOAD_IN_4BIT` | `true` | Use 4-bit quantization |
| `LOG_LEVEL` | `INFO` | Logging level |
| `GRADIO_SERVER_PORT` | `7860` | Web UI port |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Bind to all interfaces |
| `DEBUG` | `false` | Debug mode |

---

## Next Steps

- Read [API.md](API.md) for component documentation
- Check [DEVELOPMENT.md](DEVELOPMENT.md) for architecture details
- See [DEPLOYMENT.md](DEPLOYMENT.md) for hosting options
