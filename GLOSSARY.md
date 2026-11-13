# Domain Glossary

Canonical vocabulary used across the Multimodal Assignment Feedback Assistant (MAFA).

## Core Domain Terms

### Submission
A student's assignment work provided either as a handwritten image (processed via Vision OCR) or as raw text/LaTeX mathematical derivations.

### Rubric Rule
A verified grading criterion derived from course materials and stored in ChromaDB. Identified by a unique `rule_id` (e.g., `BP_001`), containing the `topic`, `subtopic`, `criteria`, `common_error`, points, and a `socratic_hint` template.

### Diagnostic Analysis
The output of the evaluation phase where the student's submission is compared against retrieved Rubric Rules to identify correct steps, rule violations, and severity (`minor` oversight vs. `major` critical gap).

### Socratic Hint
A guided, reflective question that points out a misconception or asks the student to re-examine a specific step without providing the numerical answer or mathematical solution.

### Anti-Leakage Guardrail
A safety check that inspects generated feedback cards against a list of solution-revealing patterns (e.g., "the answer is", "therefore x =") to ensure strict compliance with Socratic pedagogy.

### Feedback Card
The final structured markdown report presented to the student, containing praise for correct work, targeted Socratic questions for identified gaps, and rubric references.

### Inference Adapter
An abstraction seam separating LLM inference execution from the pipeline logic. Allows running Unsloth 4-bit CUDA models on Kaggle, free-tier serverless APIs on Hugging Face Spaces, or in-memory mock adapters for unit testing.
