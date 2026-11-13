# Engineering Design Rationale & Technical Q&A

This document records the architectural decisions, trade-offs, and technical rationale behind the Multimodal Assignment Feedback Assistant (MAFA).

---

## 1. Core Technical Defense (Interview Q&A)

### Q1: Why fine-tune a model with Socratic dialogue instead of using standard few-shot prompt engineering on GPT-4 / Claude?
> **Answer:**  
> Standard commercial LLMs are trained with RLHF to be helpful, which causes them to eagerly provide the full correct answer or derivation when a student makes an error. While prompt engineering can reduce this, general models routinely suffer from "helpfulness leakage" on complex multi-step math problems.  
> We fine-tuned Phi-3-mini (3.8B) with LoRA on synthetic `student_teacher_pairs.jsonl` specifically curated with Socratic questioning techniques. This hardcodes the pedagogical policy into the model weights, ensuring it asks scaffolding questions that lead students to discover their own errors.

### Q2: Why use RAG if the model is already fine-tuned on machine learning?
> **Answer:**  
> Fine-tuning teaches *tone, style, and pedagogy* (how to ask questions without giving answers), but shouldn't be relied upon for exact syllabus criteria and points allocations.  
> Course rubrics frequently update across semesters. By storing professor-approved rubrics in ChromaDB and using hybrid retrieval (combining semantic embeddings with exact topic filtering), we ground evaluations in verified ground truth. This eliminates hallucinated grading criteria and allows syllabus updates without retraining weights.

### Q3: How do you guarantee the model never leaks the solution to a student?
> **Answer:**  
> Through a multi-layered defense-in-depth:
> 1. **Model Weights**: Fine-tuned on student-teacher pairs where teacher turns strictly contain questions, hints, and diagnostic observations.
> 2. **Prompt Boundary**: System prompt explicitly specifies penalties for solution disclosure.
> 3. **Deterministic Guardrail Filter**: A post-generation regex and keyword filter (`LEAKAGE_KEYWORDS` such as `"therefore x ="`, `"the correct formula is"`, `"the solution is"`) flags and suppresses any output attempting to disclose answers.

### Q4: How is the system deployed without incurring expensive GPU hosting costs?
> **Answer:**  
> We applied the **Adapter Pattern** at our inference seam:
> * **Training & High-Throughput Benchmarks**: Runs on Kaggle GPUs using Unsloth 4-bit quantized LoRA adapters (`UnslothCudaAdapter`).
> * **Live Portfolio Deployment ($0 Cost)**: Deployed to Hugging Face Spaces (free 2 vCPU tier) using a hybrid runtime. It performs live ChromaDB vector search over the rubric database and either connects to free-tier serverless inference APIs (e.g., Groq / Gemini Flash) or serves interactive verified test cases in showcase mode without requiring any paid GPU instances.

### Q5: Why choose Phi-3-mini over larger open-weight models like Llama 3 8B or Mistral 7B?
> **Answer:**  
> Latency, memory footprint, and edge deployability:
> 1. With 4-bit quantization via BitsAndBytes/Unsloth, Phi-3-mini consumes under 3.5GB VRAM, allowing it to run concurrently with vision models (like Qwen2-VL) on a single free T4 GPU (16GB VRAM) on Kaggle or Google Colab.
> 2. The 4096-token context window is more than sufficient for long-form student derivations and rubric rules.

---

## 2. Whiteboard Pipeline Walkthrough

When explaining the system flow, walk through the 5 stages:

```
Step 1: Ingestion      Extract math and derivations from student uploads (OCR/Vision).
Step 2: Retrieval      Retrieve top-k rubric rules from ChromaDB matching topic & error patterns.
Step 3: Diagnosis      Structured Analyzer parses student derivation into (Correct Steps, Missing/Violated Rules).
Step 4: Pedagogy       Socratic Coach crafts targeted scaffolding hints based on violated rule seeds.
Step 5: Verification   Anti-Leakage Guardrail screens feedback text before delivering to the UI.
```

---

## 3. Engineering Metrics & Evaluation

* **Character Error Rate (CER)**: Evaluates Vision OCR accuracy against ground truth LaTeX.
* **Rubric Recall @ K**: Measures whether the RAG retriever pulled the correct ground-truth rubric rule for a known student mistake.
* **Solution Leakage Rate**: Percentage of generated feedback cards that inadvertently contain answers or final numeric values (target: `< 1%`).
