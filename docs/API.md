# API Reference

## Core Components

### VisionTranscriber

Extracts text from images and PDFs using Qwen2-VL vision model.

```python
from src.vision import VisionTranscriber

transcriber = VisionTranscriber()
text = transcriber.transcribe_image("assignment.png")
text = transcriber.transcribe_pdf("assignment.pdf")
```

**Methods:**
- `transcribe_image(path: str) -> str` — Extract text from image
- `transcribe_pdf(path: str) -> str` — Extract text from all PDF pages

---

### ChromaRetriever

Retrieves relevant rubric rules using semantic search.

```python
from src.rag import ChromaRetriever

retriever = ChromaRetriever("./data/chromadb")
rules = retriever.retrieve(query="HMM likelihood", top_k=5)
```

**Methods:**
- `retrieve(query: str, top_k: int = 5) -> List[Dict]` — Retrieve top-k rubric rules
- `add_rules(rules: List[Dict]) -> None` — Add new rules to index
- `clear() -> None` — Reset index

**Rule Schema:**
```json
{
  "rule_id": "hmm-1",
  "subtopic": "HMM - Likelihood Problem",
  "criteria": "Correctly computes forward probability α(t)",
  "points": 3,
  "common_error": "Confuses forward with backward pass"
}
```

---

### StructuredLLMAnalyzer

Analyzes assignments and produces structured rubric-aligned feedback.

```python
from src.llm import StructuredLLMAnalyzer

analyzer = StructuredLLMAnalyzer(model, tokenizer)
feedback = analyzer.analyze(
    student_text="My solution...",
    rubric_rules=[...],
    topic="HMM"
)
```

**Methods:**
- `analyze(student_text: str, rubric_rules: List[Dict], topic: str) -> Dict` — Grade assignment
- Returns:
  ```json
  {
    "score": 18,
    "max_score": 20,
    "violations": ["rule-1", "rule-3"],
    "evidence": "Student correctly computed...",
    "suggestions": ["Consider the termination step..."]
  }
  ```

---

### SocraticCoach

Generates Socratic follow-up questions to deepen student understanding.

```python
from src.llm import SocraticCoach

coach = SocraticCoach(model, tokenizer)
questions = coach.generate_questions(
    topic="HMM",
    student_understanding="partial",
    score=18,
    max_score=20
)
```

**Methods:**
- `generate_questions(topic: str, student_understanding: str, score: int, max_score: int) -> List[str]` — Generate follow-up questions

**Questioning Strategy:**
- Perfect score (18-20): Advanced challenges from `PERFECT_SCORE_CHALLENGES`
- Good score (14-17): Clarifying questions on edge cases
- Partial (10-13): Guided rediscovery questions
- Low (<10): Scaffolded concept questions

---

### Pipeline (LangGraph)

Orchestrates the full analysis workflow.

```python
from src.pipeline import build_pipeline

pipeline = build_pipeline(vlm, analyzer, coach, retriever)

result = pipeline.invoke({
    "assignment_path": "submission.pdf",
    "topic": "HMM",
    "rubric_rules": []
})
```

**Pipeline Stages:**
1. Transcription (Vision)
2. Rubric Retrieval (RAG)
3. Analysis (LLM)
4. Coaching (LLM)
5. Report Generation

---

## Configuration

### Settings

```python
from config import settings

print(settings.LLM_MODEL_PATH)
print(settings.DEVICE)
print(settings.LOG_LEVEL)
```

All settings load from `.env` or defaults. See `.env.example`.

---

## Constants

```python
from config.constants import (
    LEAKAGE_KEYWORDS,
    PERFECT_SCORE_CHALLENGES,
    DEFAULT_CHALLENGE,
    MODEL_MAX_SEQ_LEN,
    ANALYZER_MAX_PROMPT,
    COACH_MAX_PROMPT
)
```

---

## Error Handling

All components raise `MAFAException` for errors:

```python
from src.exceptions import MAFAException

try:
    text = transcriber.transcribe_pdf("file.pdf")
except MAFAException as e:
    print(f"Error: {e.message}")
    print(f"Code: {e.code}")
```

**Exception Codes:**
- `TRANSCRIPTION_ERROR` — Vision model failed
- `RETRIEVAL_ERROR` — ChromaDB query failed
- `ANALYSIS_ERROR` — LLM analysis failed
- `PIPELINE_ERROR` — Workflow failed

---

## Examples

### Complete Analysis Pipeline

```python
from src.pipeline import build_pipeline
from src.vision import VisionTranscriber
from src.rag import ChromaRetriever
from src.llm import StructuredLLMAnalyzer, SocraticCoach
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load models
model = AutoModelForCausalLM.from_pretrained("./models/phi_3")
tokenizer = AutoTokenizer.from_pretrained("./models/phi_3")

# Initialize components
vlm = VisionTranscriber()
retriever = ChromaRetriever("./data/chromadb")
analyzer = StructuredLLMAnalyzer(model, tokenizer)
coach = SocraticCoach(model, tokenizer)

# Build pipeline
pipeline = build_pipeline(vlm, analyzer, coach, retriever)

# Run analysis
result = pipeline.invoke({
    "assignment_path": "student_work.pdf",
    "topic": "Hidden Markov Models",
    "rubric_rules": retriever.retrieve("HMM")
})

print(f"Score: {result['score']}/{result['max_score']}")
print(f"Feedback: {result['feedback']}")
print(f"Questions: {result['coaching_questions']}")
```

### Custom Rubric Loading

```python
from src.ingestion import RubricLoader

loader = RubricLoader()
rules = loader.load_from_markdown("data/rubric/custom_rubric.md")
retriever.add_rules(rules)
```

---

## Testing

Run component tests:

```bash
pytest tests/test_vision.py -v
pytest tests/test_retriever.py -v
pytest tests/test_analyzer.py -v
pytest tests/test_pipeline.py -v
```

---

See [DEVELOPMENT.md](DEVELOPMENT.md) for architecture and contributing guidelines.
