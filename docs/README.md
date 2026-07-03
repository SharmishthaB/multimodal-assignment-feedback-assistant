# MAFA — Multimodal Assignment Feedback Assistant

**AI-powered assignment grading & Socratic coaching for ML courses**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

MAFA is an intelligent assignment feedback system that combines **computer vision, retrieval-augmented generation (RAG), and fine-tuned LLMs** to:

- 📄 **Extract** handwritten and printed assignment text using a Vision-Language Model (Qwen2-VL)
- 🔍 **Retrieve** rubric criteria using semantic search (ChromaDB + embeddings)
- 📊 **Analyze** submissions with structured, rubric-aligned feedback
- 💡 **Coach** students using Socratic questioning to deepen understanding

Built for **ML/AI course instructors** to scale personalized feedback while maintaining pedagogical rigor.

---

## 🏗️ Architecture

```
Assignment (PDF/Image)
    ↓
[Vision Transcriber] → Extract text + handwriting
    ↓
[Rubric Retriever] → Semantic search (ChromaDB)
    ↓
[LLM Analyzer] → Structured grading with rubric
    ↓
[Socratic Coach] → Generate follow-up questions
    ↓
Feedback Report + Coaching Tips
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vision Transcriber** | Qwen2-VL-7B | Handwriting + text extraction from images/PDFs |
| **Embedding Model** | BAAI/bge-small-en-v1.5 | Semantic search for rubric retrieval |
| **Vector Database** | ChromaDB | In-memory vector store for rubric rules |
| **LLM (Fine-tuned)** | Phi-3 + LoRA | Assignment analysis & structured grading |
| **Workflow Engine** | LangGraph | DAG-based pipeline orchestration |
| **UI Framework** | Gradio | Interactive web interface |

---

## 📦 Tech Stack

- **Language**: Python 3.10+
- **ML Framework**: PyTorch 2.0+
- **LLMs**: Hugging Face Transformers
- **Vector DB**: ChromaDB
- **Orchestration**: LangGraph
- **UI**: Gradio 4.x
- **Quantization**: bitsandbytes (4-bit)
- **Fine-tuning**: Unsloth + PEFT + TRL

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (recommended for GPU acceleration)
- 16GB+ RAM (or 8GB with 4-bit quantization)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SharmishthaB/genai-multimodal-assignment-feedback-assistant.git
   cd genai-multimodal-assignment-feedback-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the app**
   ```bash
   python app.py
   ```

   Visit `http://localhost:7860` in your browser.

---

## 📋 Project Structure

```
mafa/
├── src/
│   ├── pipeline.py           # LangGraph workflow orchestration
│   ├── vision/               # Vision transformer components
│   ├── llm/                  # LLM analyzers & coaches
│   ├── rag/                  # Retrieval-Augmented Generation
│   └── ingestion/            # Data ingestion (PDF, rubrics)
├── config/
│   ├── settings.py           # Configuration management
│   └── constants.py          # Constants & challenge templates
├── data/
│   ├── rubric/               # Rubric definitions
│   ├── datasets/             # Training datasets
│   └── chromadb/             # Vector store (created at runtime)
├── tests/                    # Unit & integration tests
├── docs/                     # Documentation
├── app.py                    # Gradio UI entry point
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

---

## 🎓 Features

### ✅ Assignment Analysis
- Multi-format support: PDF, PNG, JPG, handwritten worksheets
- Structured grading aligned with custom rubrics
- Evidence extraction with score justification

### 💬 Socratic Coaching
- Adaptive follow-up questions based on performance
- Topic-specific challenges for perfect scores
- Encourages deeper understanding without spoiling answers

### 🔧 Extensibility
- Pluggable LLM backends
- Custom rubric definitions
- Configurable pipeline stages

### 📊 Debug & Transparency
- Rubric retrieval scoring
- Evidence snippet highlighting
- Full analysis reasoning visible in UI

---

## 🔧 Configuration

Edit `.env` to customize:

```bash
# Model paths
LLM_MODEL_PATH=./models/phi_3
CHROMADB_PATH=./data/chromadb

# Device selection
DEVICE=cuda  # or 'cpu'
LOAD_IN_4BIT=true

# Logging
LOG_LEVEL=INFO
```

See [SETUP.md](docs/SETUP.md) for detailed configuration guide.

---

## 📚 Usage Guide

### Basic Workflow

1. **Upload assignment** (PDF or image)
2. **Select rubric** (or use default ML rubric)
3. **Analyze** (runs through pipeline)
4. **View feedback** with:
   - Rubric-aligned scores
   - Evidence extraction
   - Improvement suggestions
   - Coaching questions

### Advanced: Custom Rubrics

Rubrics are defined as JSON:

```json
{
  "rule_id": "hmm-1",
  "subtopic": "HMM - Likelihood Problem",
  "criteria": "Correctly computes forward probability α(t)",
  "points": 3,
  "common_error": "Confuses forward with backward pass"
}
```

Place in `data/rubric/` and rebuild ChromaDB:
```bash
python build_chromadb.py
```

See [API.md](docs/API.md) for full API reference.

---

## 🤖 Fine-tuning

To fine-tune Phi-3 on your own teaching data:

```bash
python scripts/finetune.py \
  --data_path data/teaching_samples.jsonl \
  --output_dir models/phi_3_custom \
  --epochs 3
```

Dataset format (JSONL):
```json
{
  "instruction": "Grade this ML assignment...",
  "input": "[Assignment text]",
  "output": "[Your feedback]"
}
```

---

## 🧪 Testing

Run tests:
```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

---

## 📖 Documentation

- [**SETUP.md**](docs/SETUP.md) — Detailed installation & configuration
- [**API.md**](docs/API.md) — Component API reference
- [**DEVELOPMENT.md**](docs/DEVELOPMENT.md) — Developer guide & architecture

---

## 🐳 Docker Deployment

Build and run with Docker:

```bash
docker build -t mafa:latest .
docker run -it --gpus all -p 7860:7860 mafa:latest
```

See [Docker setup](docs/SETUP.md#docker) for GPU configuration.

---

## 🚢 Deployment Options

### Recommended: HuggingFace Spaces ⭐
Free tier with GPU support, native Gradio integration:
1. Fork to your GitHub
2. Create Space from GitHub repo
3. Set environment variables in Space settings

### Other Options
- **Render**: `pip install` + start command
- **Railway**: Connect GitHub repo, auto-deploy
- **AWS EC2**: Full control, pay-per-hour

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for step-by-step guides.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add feature'`
4. Push: `git push origin feature/your-feature`
5. Open Pull Request

---

## 📝 License

MIT License — see [LICENSE](LICENSE) file.

---

## 🙋 Support & Questions

- **Issues**: [GitHub Issues](https://github.com/SharmishthaB/genai-multimodal-assignment-feedback-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SharmishthaB/genai-multimodal-assignment-feedback-assistant/discussions)
- **Email**: [your-email@example.com]

---

## 📊 Project Status

- ✅ Core pipeline (Vision → RAG → LLM → Coaching)
- ✅ Gradio UI with debug panel
- ✅ ChromaDB integration
- ⏳ Deployment guides
- ⏳ Comprehensive test suite
- 🔄 Continuous improvements

---

## 🙏 Acknowledgments

- **Qwen2-VL-7B** for handwriting recognition
- **Phi-3** fine-tuned model for assignment analysis
- **Unsloth** team for 2x faster training
- **LangGraph** for workflow orchestration
- **HuggingFace** ecosystem

---

**Made with ❤️ for educators and learners**
