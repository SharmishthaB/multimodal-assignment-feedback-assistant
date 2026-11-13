"""
Quick interactive test script for MAFA FeedbackPipeline.
Demonstrates end-to-end evaluation with zero external dependencies or GPU requirements.
"""

from src.pipeline import FeedbackPipeline

def main():
    print("=" * 60)
    print("🎓 MAFA Interactive Pipeline Test")
    print("=" * 60)

    # Initialize in mock mode (instant local execution)
    pipeline = FeedbackPipeline.from_mock()

    sample_submission = (
        "Forward pass:\n"
        "z2 = W2 * a1 + b2\n"
        "a2 = sigmoid(z2)\n"
        "Loss = 0.5 * (a2 - y)^2\n\n"
        "Backward pass:\n"
        "delta2 = a2 - y\n"
        "delta1 = a1 - y  <-- Applied output error directly to layer 1"
    )
    topic = "Backpropagation"

    print(f"\n[1] Topic: {topic}")
    print(f"[2] Student Submission:\n{sample_submission}\n")
    print("[3] Running Pipeline (Ingest -> RAG -> Analyze -> Coach -> Guardrails)...")

    result = pipeline.run(student_work=sample_submission, topic=topic)

    print("\n" + "=" * 60)
    print("📊 1. DIAGNOSTIC ANALYSIS (Identified Violations & Correct Steps)")
    print("=" * 60)
    print(f"Rubric Score : {result.error_analysis.get('rubric_score')}/3")
    print("Correct Steps:")
    for step in result.error_analysis.get("correct_steps", []):
        print(f"  ✓ {step}")
    print("Violations:")
    for v in result.error_analysis.get("violated_rules", []):
        print(f"  ✗ [{v.get('rule_id')}] ({v.get('severity')}): {v.get('student_error')}")

    print("\n" + "=" * 60)
    print("💡 2. SOCRATIC FEEDBACK CARD (No Answers Leaked)")
    print("=" * 60)
    print(result.feedback_card)

    print("\n" + "=" * 60)
    print("🔍 3. RETRIEVED GROUND-TRUTH RUBRIC RULES (ChromaDB / Markdown RAG)")
    print("=" * 60)
    for r in result.rubric_rules[:3]:
        print(f"- [{r.get('rule_id')}] {r.get('subtopic')}: {r.get('criteria')}")

    print("\n" + "=" * 60)
    print(f"Anti-Leakage Guardrail Passed: {'YES' if not result.leakage_detected else 'FLAGGED'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
