"""Core constants and challenge definitions for MAFA."""

# Keywords that indicate direct answer leakage
LEAKAGE_KEYWORDS = [
    "the answer is",
    "therefore x =",
    "the correct formula is",
    "solving this gives",
    "the solution is",
    "= [specific numeric answer]",
    "here's how to solve",
]

# Socratic follow-up challenges for perfect scores
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

# Default challenge for unknown topics
DEFAULT_CHALLENGE = "can you think of an edge case where this algorithm would behave unexpectedly?"

# Model configuration
MODEL_MAX_SEQ_LEN = 4096
ANALYZER_MAX_PROMPT = MODEL_MAX_SEQ_LEN - 500
COACH_MAX_PROMPT = MODEL_MAX_SEQ_LEN - 300
