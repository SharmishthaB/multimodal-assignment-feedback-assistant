# MAFA — Multimodal Assignment Feedback Assistant

**AI-powered assignment grading & Socratic coaching using Vision-Language Models, RAG, and fine-tuned LLMs**

## 🎯 Quick Links

📖 **[Full Documentation](docs/README.md)** | 🚀 **[Setup Guide](docs/SETUP.md)** | 📚 **[API Reference](docs/API.md)**

---

## What is MAFA?

MAFA is an intelligent system that helps instructors grade assignments at scale while maintaining personalized, pedagogically-sound feedback. It combines:

- 📷 **Vision-Language Model** (Qwen2-VL) for handwritten/printed text extraction
- 🔍 **Semantic Search** (ChromaDB) for rubric retrieval
- 🧠 **Fine-tuned LLM** (Phi-3) for structured grading
- 💬 **Socratic Coaching** engine for follow-up questions

**Perfect for:** ML/AI course instructors who want to provide detailed, consistent feedback at scale.

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/SharmishthaB/genai-multimodal-assignment-feedback-assistant.git
cd genai-multimodal-assignment-feedback-assistant

# 2. Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your settings (DEVICE=cuda recommended)

# 4. Build vector DB
python build_chromadb.py

# 5. Run!
python app.py
```

Visit **http://localhost:7860**

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Gradio 4.x |
| **Vision** | Qwen2-VL-7B |
| **RAG** | ChromaDB + Embeddings |
| **LLM** | Phi-3 (fine-tuned) |
| **Orchestration** | LangGraph |
| **Quantization** | bitsandbytes 4-bit |
| **Fine-tuning** | Unsloth + LoRA |

---

## 📂 Project Structure

```
├── src/                    # Core modules (vision, LLM, RAG)
├── config/                 # Configuration & constants
├── data/                   # Rubrics, datasets, embeddings
├── tests/                  # Test suite
├── docs/                   # Comprehensive documentation
├── app.py                  # Gradio UI entry point
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Multi-service setup
└── .env.example            # Environment template
```

---

## 🚀 Deployment

### Local (Recommended for Development)
```bash
python app.py
```

### Docker (CPU)
```bash
docker build -t mafa:latest .
docker run -it -p 7860:7860 mafa:latest
```

### Docker with GPU
```bash
docker run -it --gpus all -p 7860:7860 mafa:latest
```

### Docker Compose
```bash
docker-compose up --build
```

### Cloud Deployment
- **HuggingFace Spaces** (recommended, free GPU tier)
- **Render, Railway, AWS EC2** (see [docs/README.md](docs/README.md#-deployment) for guides)

---

## 📖 Documentation

- **[Full README](docs/README.md)** — Features, architecture, overview
- **[Setup Guide](docs/SETUP.md)** — Installation, configuration, troubleshooting
- **[API Reference](docs/API.md)** — Component documentation, code examples

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📋 Features

✅ Multi-format support (PDF, PNG, JPG, handwritten)  
✅ Rubric-aligned grading  
✅ Socratic follow-up questions  
✅ Evidence extraction & highlighting  
✅ Custom rubric definitions  
✅ Fine-tuning support  
✅ Docker ready  
✅ Gradio web UI  

---

## 🙋 Support

- **Issues**: [GitHub Issues](https://github.com/SharmishthaB/genai-multimodal-assignment-feedback-assistant/issues)
- **Questions**: Open a discussion or issue

---

**Made with ❤️ for educators**
