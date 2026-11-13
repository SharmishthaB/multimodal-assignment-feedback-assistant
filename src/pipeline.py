"""
Feedback Pipeline Module
========================
Deep module encapsulating the 5-stage Multimodal Socratic Evaluation:
1. Vision Ingestion (OCR / LaTeX Extraction)
2. Rubric RAG Retrieval (ChromaDB / Direct Rubric Search)
3. Diagnostic Analysis (Misconception & Rule Violation Identification)
4. Socratic Coaching (Guiding Questions & Scaffolding)
5. Anti-Leakage Guardrail (Zero Solution Disclosure)
"""

import os
import re
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .config import (
    LEAKAGE_KEYWORDS,
    PERFECT_SCORE_CHALLENGES,
    DEFAULT_CHALLENGE,
)
from .repository import KnowledgeRepository
from .inference import (
    InferenceEngine,
    VisionTranscriber,
    get_inference_engine,
    get_vision_transcriber,
    MockInference,
    MockVisionTranscriber,
)

logger = logging.getLogger(__name__)


# Backward-compatible alias
InferenceAdapter = InferenceEngine
VisionAdapter = VisionTranscriber

@dataclass
class FeedbackResult:
    """Canonical output returned by the FeedbackPipeline."""
    topic: str
    transcribed_text: str
    feedback_card: str
    error_analysis: Dict[str, Any]
    rubric_rules: List[Dict[str, Any]]
    is_perfect: bool = False
    leakage_detected: bool = False
    raw_response: Optional[str] = None


# ============================================================================
# DEEP FEEDBACK PIPELINE
# ============================================================================

class FeedbackPipeline:
    """
    Deep Module presenting a single, unified interface for the entire
    multimodal assignment evaluation and Socratic coaching workflow.
    """

    def __init__(
        self,
        llm: InferenceEngine,
        repository: Optional[KnowledgeRepository] = None,
        vision: Optional[VisionTranscriber] = None,
    ):
        self.llm = llm
        self.repository = repository or KnowledgeRepository()
        self.vision = vision or MockVisionTranscriber()

    @classmethod
    def create(cls, runtime: Optional[str] = None):
        """Standard factory auto-detecting environment (Kaggle GPU vs Cloud/Mock)."""
        return cls(
            llm=get_inference_engine(runtime),
            repository=KnowledgeRepository(),
            vision=get_vision_transcriber(runtime),
        )

    @classmethod
    def from_kaggle(cls):
        """Factory explicitly configured for Kaggle GPU runtime."""
        return cls.create(runtime="kaggle")

    @classmethod
    def from_mock(cls):
        """Factory explicitly configured for mock/local development."""
        return cls(
            llm=MockInference(),
            repository=KnowledgeRepository(),
            vision=MockVisionTranscriber(),
        )

    # ── Internal Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _clean_rule_id(rid: Any) -> str:
        """Normalises rule IDs (e.g. '[BP_001] Chain Rule' -> 'BP_001', 'SVM_SM_003' -> 'SVM_SM_003')."""
        match = re.search(r"[A-Z]{2,5}(?:_[A-Z]{1,4})*_\d{3}", str(rid))
        return match.group(0) if match else re.sub(r"[\[\]\s]", "", str(rid))

    @staticmethod
    def _format_rubric_for_analyzer(rubric_rules: List[Dict[str, Any]]) -> str:
        if not rubric_rules:
            return "(No rubric rules retrieved — evaluate based on topic knowledge)"
        lines = []
        for r in rubric_rules:
            rid = r.get("rule_id", "?")
            sub = r.get("subtopic", "")
            crit = r.get("criteria", "").strip()
            pts = r.get("points", "1")
            common = r.get("common_error", "").strip()
            line = f"[{rid}] {sub} ({pts} pts)\n  Criteria: {crit}"
            if common:
                line += f"\n  Common Error: {common}"
            lines.append(line)
        return "\n\n".join(lines)

    @staticmethod
    def _build_violation_brief(violated_rules: List[Dict[str, Any]], rubric_rules: List[Dict[str, Any]]) -> str:
        rule_map = {
            FeedbackPipeline._clean_rule_id(r.get("rule_id", "")): r
            for r in rubric_rules
        }
        blocks = []
        for i, v in enumerate(violated_rules, 1):
            rid = FeedbackPipeline._clean_rule_id(v.get("rule_id", ""))
            error_text = v.get("student_error", "").strip()
            severity = v.get("severity", "minor")
            sev_label = "critical gap" if severity == "major" else "minor oversight"

            r = rule_map.get(rid, {})
            subtopic = r.get("subtopic", rid)
            hint_seed = r.get("socratic_hint", "").strip()
            common_error = r.get("common_error", "").strip()

            block = f"Issue {i} [{rid} — {subtopic}] ({sev_label})\n"
            if error_text:
                block += f"  Student wrote: {error_text}\n"
            if common_error:
                block += f"  Known pattern: {common_error}\n"
            if hint_seed:
                block += f"  Hint seed (adapt, never copy verbatim): {hint_seed}\n"
            blocks.append(block)
        return "\n".join(blocks) if blocks else "(none)"

    @staticmethod
    def _normalise_analysis(raw_text: str, rubric_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parses model JSON output and rescues fused keys or malformed fields."""
        parsed = {}
        try:
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
        except Exception:
            pass

        if not isinstance(parsed, dict):
            parsed = {}

        violations_raw = parsed.get("violated_rules", [])
        violations_clean = []
        top_correct_steps = list(parsed.get("correct_steps", []))

        for v in violations_raw:
            if not isinstance(v, dict):
                continue
            raw_id = str(v.get("rule_id") or v.get("rule", ""))
            v["rule_id"] = FeedbackPipeline._clean_rule_id(raw_id)
            v.setdefault("student_error", "")
            v.setdefault("severity", "minor")
            if v["rule_id"] != "UNKNOWN" or v["student_error"]:
                violations_clean.append(v)

        return {
            "topic": parsed.get("topic", ""),
            "violated_rules": violations_clean,
            "correct_steps": list(dict.fromkeys(top_correct_steps)),
            "rubric_score": parsed.get("rubric_score", 1 if violations_clean else 3),
        }

    @staticmethod
    def _clean_generation(text: str) -> str:
        """Removes box-drawing characters and excessive symbol noise."""
        text = re.sub(r"[\u2500-\u257F\u2580-\u259F\u25A0-\u25FF]+", "", text)
        text = re.sub(r"[\u2190-\u21FF\u2200-\u22FF]{2,}", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {3,}", " ", text)
        return text.strip()

    # ── Primary Execution Seam ────────────────────────────────────────────────

    def run(
        self,
        student_work: str = "",
        topic: str = "",
        image_path: Optional[str] = None,
    ) -> FeedbackResult:
        """
        Executes the full multimodal evaluation pipeline.

        Args:
            student_work: Direct text/LaTeX entered by the student.
            topic: Machine Learning curriculum topic.
            image_path: Path to an uploaded handwritten assignment image.

        Returns:
            FeedbackResult with structured analysis and Socratic card.
        """
        # 1. Vision Ingestion
        transcribed = student_work.strip() if student_work else ""
        if image_path and os.path.exists(image_path):
            try:
                ocr_result = self.vision.transcribe(image_path)
                if ocr_result:
                    transcribed = (
                        f"{transcribed}\n{ocr_result}".strip()
                        if transcribed
                        else ocr_result
                    )
            except Exception as e:
                logger.error(f"Vision transcription failed: {e}")

        # 2. Rubric Retrieval (RAG)
        rules = self.repository.retrieve_rubric(topic, transcribed, k=5)

        # 3. Diagnostic Analysis
        analyzer_prompt = (
            f"<|system|>\nYou are an expert Machine Learning Teaching Assistant and Diagnostic Evaluator.\n"
            f"Analyze the student's work strictly against the provided rubric rules.\n"
            f"Respond ONLY with a valid JSON object matching the schema:\n"
            f'{{"topic": "{topic}", "violated_rules": [{{"rule_id": "...", "student_error": "...", "severity": "minor|major"}}], '
            f'"correct_steps": ["..."], "rubric_score": 1-3}}<|end|>\n'
            f"<|user|>\nTopic: {topic}\n\nRubric Rules:\n{self._format_rubric_for_analyzer(rules)}\n\n"
            f"Student Work:\n{transcribed}<|end|>\n<|assistant|>\n"
        )
        raw_analysis = self.llm.generate(analyzer_prompt)
        analysis = self._normalise_analysis(raw_analysis, rules)

        is_perfect = len(analysis.get("violated_rules", [])) == 0

        # 4. Socratic Coaching
        if is_perfect:
            challenge = PERFECT_SCORE_CHALLENGES.get(topic, DEFAULT_CHALLENGE)
            feedback = (
                f"### 🌟 Outstanding Work!\n\n"
                f"Your solution for **{topic}** is completely correct and follows all rubric criteria!\n\n"
                f"**Deepen Your Understanding:**\n"
                f"To push your mastery further: {challenge}"
            )
        else:
            violation_brief = self._build_violation_brief(analysis["violated_rules"], rules)
            coach_prompt = (
                f"<|system|>\nYou are a Socratic teaching coach for Machine Learning.\n"
                f"Guide students toward understanding without revealing answers.\n"
                f"NEVER say 'the answer is', never provide the corrected equation, and never solve the problem for them.\n"
                f"Ask leading questions that help them discover where their derivation went astray.<|end|>\n"
                f"<|user|>\nTopic: {topic}\n\nStudent Work:\n{transcribed}\n\n"
                f"Diagnostic Observations:\n{violation_brief}<|end|>\n<|assistant|>\n"
            )
            raw_feedback = self.llm.generate(coach_prompt)
            feedback = self._clean_generation(raw_feedback)

        # 5. Anti-Leakage Guardrail
        leakage_detected = any(
            kw in feedback.lower() for kw in LEAKAGE_KEYWORDS
        )
        if leakage_detected:
            logger.warning("Guardrail triggered: solution leakage detected in coach output!")
            feedback = (
                f"### 💡 Socratic Guidance\n\n"
                f"Review your derivation for **{topic}** around the steps highlighted in the diagnostic panel.\n\n"
                f"**Guided Question:**\n"
                f"Which underlying mathematical property or rule did you assume when transitioning between the steps? "
                f"Check whether that assumption holds for intermediate variables."
            )

        return FeedbackResult(
            topic=topic,
            transcribed_text=transcribed,
            feedback_card=feedback,
            error_analysis=analysis,
            rubric_rules=rules,
            is_perfect=is_perfect,
            leakage_detected=leakage_detected,
            raw_response=raw_analysis,
        )
