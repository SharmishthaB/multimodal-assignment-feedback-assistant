# ADR-0001: Deep Feedback Pipeline and Inference Adapters

## Status
Accepted

## Context
The initial prototype implemented the pipeline across multiple disconnected classes (`VisionTranscriber`, `StructuredLLMAnalyzer`, `SocraticCoach`, `ChromaRetriever`) coordinated directly within the Gradio presentation layer (`app.py`), while model weights were loaded eagerly on import. This tightly coupled the UI to Kaggle GPU environments, made unit testing impossible without CUDA, and spread error-handling and prompt-truncation logic across multiple modules.

## Decision
1. **Collapse Pipeline into a Deep Module**: Unify transcription, rubric retrieval, diagnostic analysis, Socratic coaching, and guardrail verification into a single deep `FeedbackPipeline` module in `src/pipeline.py`.
2. **Inference Adapters**: Place model generation behind an `InferenceAdapter` interface (`UnslothAdapter`, `ServerlessAPIAdapter`, `MockAdapter`) allowing the pipeline to execute identically across Kaggle GPUs, local CPU development, and free portfolio hosting environments.
3. **Lazy Initialization**: Defer all GPU memory allocations and model downloads until explicitly called.

## Consequences
* **Positive**: The UI code in `app.py` shrinks to a thin presentation layer; unit testing can run in headless CI without GPUs; portfolio deployment on free tiers becomes possible.
* **Trade-off**: Requires dependency injection of the adapter when instantiating the pipeline.
