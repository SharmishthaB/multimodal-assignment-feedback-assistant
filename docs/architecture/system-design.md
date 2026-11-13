# System Design & Architecture

## System Overview

The **Multimodal Assignment Feedback Assistant (MAFA)** is an intelligent tutoring system designed to grade and provide pedagogical feedback on undergraduate Machine Learning assignments. Rather than providing direct corrections, MAFA uses Socratic questioning grounded in course rubrics to stimulate student self-discovery.

```
                           ┌──────────────────────────────┐
                           │      Student Submission      │
                           │   (Handwritten / Text Math)  │
                           └──────────────┬───────────────┘
                                          │
                                          ▼
┌───────────────────┐              ┌──────────────┐
│  Lecture Chunks   │              │ Vision Stage │ (Qwen2-VL / EasyOCR)
│   & Rubric MD     │              └──────┬───────┘
└─────────┬─────────┘                     │
          │                               ▼
          ▼                     ┌──────────────────┐
┌───────────────────┐           │ Transcribed Math │
│  ChromaDB Vector  │           └─────────┬────────┘
│    Repository     │◄────────────────────┤
└─────────┬─────────┘                     ▼
          │ Hybrid Search       ┌──────────────────┐
          │ (Semantic + Topic)  │ 1. Retrieval RAG │
          └────────────────────►└─────────┬────────┘
                                          │
                                          ▼
                                ┌──────────────────┐
                                │ 2. Analyzer Node │ (Phi-3-mini Diagnostic)
                                └─────────┬────────┘
                                          │
                                          ▼
                                ┌──────────────────┐
                                │ 3. Socratic Node │ (Phi-3 LoRA Fine-Tuned)
                                └─────────┬────────┘
                                          │
                                          ▼
                                ┌──────────────────┐
                                │  4. Guardrails   │ (Regex & Semantic Filter)
                                └─────────┬────────┘
                                          │
                                          ▼
                                ┌──────────────────┐
                                │  Feedback Card   │
                                └──────────────────┘
```

## Architectural Principles

1. **Pedagogical Integrity (Zero Solution Leakage)**:
   Feedback must guide students through targeted questions, never reveal solutions, final values, or derivations.
2. **Grounding via Rubric RAG**:
   Grading rules and misconception templates are indexed in ChromaDB collections (`rubric_rules` and `lecture_knowledge`) to prevent hallucinated criteria.
3. **Deep Pipeline Module**:
   The entire 5-stage pipeline sits behind a single deep interface: `pipeline.run(submission, topic) -> FeedbackResult`. UI frameworks (Gradio, Streamlit) never touch internal prompts or tokenizers directly.
4. **Adapter Seam for Runtime Flexibility**:
   Decouples inference hardware from pipeline logic. The system runs on Kaggle GPU via `UnslothAdapter`, locally in CI via `MockAdapter`, and on free cloud hosting via `ServerlessAPIAdapter`.
