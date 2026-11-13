"""
MAFA Configuration & Environment Detection
Provides unified paths and topic constants across Kaggle and Local runtimes.
"""

import os
from pathlib import Path
from typing import List

# Detect execution runtime
IS_KAGGLE = os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent

def _resolve_rubric_path() -> str:
    """Finds ml_rubric.md, prioritizing mounted datasets or falling back to the repo copy."""
    if "RUBRIC_PATH" in os.environ and os.path.exists(os.environ["RUBRIC_PATH"]):
        return os.environ["RUBRIC_PATH"]
    candidates = [
        "/kaggle/input/datasets/sharmishthab/mafa-rubrics/ml_rubric.md",
        "/kaggle/input/mafa-rubrics/ml_rubric.md",
        "/kaggle/working/ml_rubric.md",
        str(ROOT_DIR / "ml_rubric.md"),
        str(ROOT_DIR / "data" / "rubrics" / "ml_rubric.md"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return str(ROOT_DIR / "ml_rubric.md")

def _resolve_model_path() -> str:
    """Finds fine-tuned model path with fallback."""
    if "MODEL_PATH" in os.environ and os.path.exists(os.environ["MODEL_PATH"]):
        return os.environ["MODEL_PATH"]
    candidates = [
        "/kaggle/working/phi_3",
        "/kaggle/input/phi-3",
        "/kaggle/input/phi_3",
        str(ROOT_DIR / "phi_3"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "/kaggle/working/phi_3" if IS_KAGGLE else str(ROOT_DIR / "phi_3")

DEFAULT_CHROMA_PATH = "/kaggle/working/chromadb" if IS_KAGGLE else str(ROOT_DIR / "chromadb")
CHROMA_PATH = os.environ.get("CHROMA_PATH", DEFAULT_CHROMA_PATH)
MODEL_PATH = _resolve_model_path()
RUBRIC_PATH = _resolve_rubric_path()

# Supported ML Curriculum Topics
TOPICS: List[str] = [
    "Decision Trees",
    "Backpropagation",
    "Forward Propagation",
    "SVM - Hard Margin",
    "SVM - Soft Margin",
    "Naive Bayes",
    "HMM - Likelihood Problem",
    "HMM - Decoding Problem",
    "HMM - Learning Problem",
    "EM Algorithm",
    "K-Means Clustering",
    "CNN Trainable Parameters",
    "Random Forest",
    "Bagging & Boosting",
    "Gradient Boosting",
    "Logistic Regression",
    "Performance Metrics",
    "PCA / SVD",
]

# Anti-leakage guardrail phrases (must never appear in Socratic feedback)
LEAKAGE_KEYWORDS: List[str] = [
    "the answer is",
    "therefore x =",
    "the correct formula is",
    "solving this gives",
    "the solution is",
    "= [specific numeric answer]",
    "here's how to solve",
    "the correct answer is",
]

# Pedagogical challenge questions for submissions with 100% score
PERFECT_SCORE_CHALLENGES = {
    "HMM - Likelihood Problem": "can you explain why the termination step sums over all states rather than taking the maximum?",
    "HMM - Decoding Problem": "can you explain why Viterbi uses max instead of sum, and what goes wrong if you use sum?",
    "HMM - Learning Problem": "can you explain what the E-step and M-step are doing geometrically in Baum-Welch?",
    "Backpropagation": "can you explain what happens to gradients in very deep networks, and how does this affect learning in early layers?",
    "Forward Propagation": "can you trace what happens to the output if one weight in the first layer becomes exactly zero?",
    "Naive Bayes": "can you explain why Laplace smoothing adds |V| to the denominator, not just 1?",
    "SVM - Hard Margin": "can you explain geometrically why only the support vectors determine the decision boundary?",
    "SVM - Soft Margin": "can you explain what increasing C does to the margin width and why?",
    "Decision Trees": "can you explain why information gain tends to favour features with many distinct values?",
    "K-Means Clustering": "can you explain what happens when two centroids converge to the same point?",
    "PCA / SVD": "can you explain the geometric meaning of the first principal component?",
    "Logistic Regression": "can you explain why we use log-loss instead of squared error for classification?",
    "Random Forest": "can you explain how feature bagging reduces correlation between trees?",
    "Gradient Boosting": "can you explain what the residuals in each boosting round represent?",
    "CNN Trainable Parameters": "can you calculate the parameter count if you added one more conv layer with the same settings?",
    "EM Algorithm": "can you explain what the Q function represents and why we maximise it?",
    "Bagging & Boosting": "can you explain why bagging reduces variance but not bias?",
    "Performance Metrics": "can you explain when you would prefer recall over precision, and give a real example?",
}

DEFAULT_CHALLENGE = "can you think of an edge case where this algorithm would behave unexpectedly?"
