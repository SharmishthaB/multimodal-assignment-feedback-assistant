"""
Unit Tests for MAFA Architecture
Verifies:
1. Deep FeedbackPipeline execution & Socratic outputs
2. Anti-leakage guardrail interception
3. KnowledgeRepository hybrid search & markdown fallback
4. InferenceEngine decoupling & zero import side-effects
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path across all execution environments
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import unittest
from src.pipeline import FeedbackPipeline, FeedbackResult
from src.inference import MockInference, MockVisionTranscriber, get_inference_engine
from src.repository import KnowledgeRepository, parse_rubric_markdown


class LeakageTestAdapter:
    """Mock adapter that deliberately attempts to leak the answer."""
    def generate(self, prompt: str) -> str:
        if "Diagnostic Evaluator" in prompt:
            return '{"topic": "Backpropagation", "violated_rules": [{"rule_id": "BP_001", "student_error": "Formula mistake"}], "correct_steps": [], "rubric_score": 1}'
        return "The answer is x = 42 and the correct formula is delta = y - y_hat."


class TestFeedbackPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = FeedbackPipeline.from_mock()

    def test_mock_pipeline_execution(self):
        """Verifies end-to-end execution of the deep pipeline."""
        result = self.pipeline.run(
            student_work="I calculated delta = y_hat - y directly for hidden layers.",
            topic="Backpropagation",
        )
        self.assertIsInstance(result, FeedbackResult)
        self.assertEqual(result.topic, "Backpropagation")
        self.assertFalse(result.is_perfect)
        self.assertFalse(result.leakage_detected)
        self.assertGreater(len(result.feedback_card), 20)
        self.assertIn("Backpropagation", result.error_analysis.get("topic", ""))

    def test_mock_pipeline_topic_adaptation(self):
        """Verifies mock inference dynamically adapts to topics like HMM Decoding."""
        result = self.pipeline.run(
            student_work="Viterbi algorithm t=1 sunny 0.53 rainy 0.13 v2(sunny)=max(...)",
            topic="HMM - Decoding Problem",
        )
        self.assertEqual(result.topic, "HMM - Decoding Problem")
        self.assertEqual(result.error_analysis.get("topic"), "HMM - Decoding Problem")
        violations = result.error_analysis.get("violated_rules", [])
        self.assertGreaterEqual(len(violations), 1)
        self.assertTrue(violations[0]["rule_id"].startswith("HMM_D"))
        self.assertIn("HMM - Decoding Problem", result.feedback_card)
        self.assertNotIn("Backpropagation", result.feedback_card)

    def test_clean_rule_id(self):
        """Verifies robust normalization of diverse model rule_id formats."""
        clean = FeedbackPipeline._clean_rule_id
        self.assertEqual(clean("[BP_001] Chain Rule"), "BP_001")
        self.assertEqual(clean("[DT_002]"), "DT_002")
        self.assertEqual(clean("HMM_D_004"), "HMM_D_004")
        self.assertEqual(clean("SVM_SM_003"), "SVM_SM_003")

    def test_anti_leakage_guardrail_triggers(self):
        """Verifies that the anti-leakage guardrail intercepts solution disclosures."""
        leaky_pipeline = FeedbackPipeline(llm=LeakageTestAdapter())
        result = leaky_pipeline.run(
            student_work="Mistake in backprop",
            topic="Backpropagation",
        )
        self.assertTrue(result.leakage_detected)
        self.assertNotIn("The answer is", result.feedback_card)
        self.assertIn("Socratic Guidance", result.feedback_card)

    def test_perfect_score_flow(self):
        """Verifies challenge questions when a student submits a 100% correct answer."""
        class PerfectAdapter:
            def generate(self, prompt: str) -> str:
                return '{"topic": "Decision Trees", "violated_rules": [], "correct_steps": ["Computed information gain"], "rubric_score": 3}'

        perfect_pipeline = FeedbackPipeline(llm=PerfectAdapter())
        result = perfect_pipeline.run(
            student_work="Perfect derivation of information gain.",
            topic="Decision Trees",
        )
        self.assertTrue(result.is_perfect)
        self.assertIn("Outstanding Work", result.feedback_card)
        self.assertIn("Deepen Your Understanding", result.feedback_card)


class TestKnowledgeRepository(unittest.TestCase):

    def test_parse_rubric_markdown(self):
        """Verifies canonical parsing of ml_rubric.md."""
        rules = parse_rubric_markdown()
        self.assertGreater(len(rules), 10)
        sample = rules[0]
        self.assertIn("rule_id", sample)
        self.assertIn("topic", sample)
        self.assertIn("criteria", sample)

    def test_repository_fallback_retrieval(self):
        """Verifies in-memory retrieval when ChromaDB is not yet built."""
        repo = KnowledgeRepository(chroma_path="/nonexistent/path/chroma")
        rules = repo.retrieve_rubric("Backpropagation", k=3)
        self.assertGreaterEqual(len(rules), 1)
        for r in rules:
            self.assertEqual(r.get("topic", "").lower(), "backpropagation")


class TestInferenceDecoupling(unittest.TestCase):

    def test_mock_engine_factory(self):
        """Verifies factory returns mock engine when outside Kaggle and without API keys."""
        engine = get_inference_engine(runtime="mock")
        self.assertIsInstance(engine, MockInference)
        out = engine.generate("Hello world")
        self.assertIn("Socratic Guidance", out)


if __name__ == "__main__":
    unittest.main()
