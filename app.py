"""
MAFA (Multimodal Assignment Feedback Assistant) — Gradio UI
============================================================
Lightweight presentation layer that interacts with the deep FeedbackPipeline.
Runs on Kaggle with Unsloth 4-bit CUDA models, and runs locally/on Hugging Face
Spaces with lightweight mock/API adapters for free portfolio demonstrations.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from src.config import TOPICS, IS_KAGGLE
from src.pipeline import FeedbackPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# INITIALIZE DEEP PIPELINE
# ============================================================================
logger.info(f"Initializing FeedbackPipeline (Runtime: {'Kaggle GPU' if IS_KAGGLE else 'Local/Cloud Demo'})...")

if IS_KAGGLE:
    pipeline = FeedbackPipeline.from_kaggle()
else:
    # Graceful fallback for local development, CI testing, and free Hugging Face Spaces
    pipeline = FeedbackPipeline.from_mock()

logger.info("✅ FeedbackPipeline ready")


# ============================================================================
# HANDLERS
# ============================================================================
def analyze_submission(
    uploaded_file,
    input_text: str,
    selected_topic: str,
    show_rubric_reference: bool = True,
    show_timestamps: bool = False,
) -> Tuple[str, Optional[str], Optional[str]]:

    has_file = uploaded_file is not None
    has_text = input_text is not None and input_text.strip() != ""

    if not has_file and not has_text:
        return "❌ **Error:** Please upload a file OR enter text.", None, None
    if not selected_topic:
        return "❌ **Error:** Please select a topic from the dropdown.", None, None

    try:
        logger.info(f"Processing submission for topic: {selected_topic}")
        image_path = uploaded_file.name if has_file else None

        result = pipeline.run(
            student_work=input_text if has_text else "",
            topic=selected_topic,
            image_path=image_path,
        )

        error_json = json.dumps(result.error_analysis, indent=2)
        logger.info("✅ Submission processing complete")
        return result.feedback_card, result.transcribed_text, error_json

    except Exception as e:
        logger.error(f"Error processing submission: {str(e)}", exc_info=True)
        return f"❌ **Error:** {str(e)}", None, None


def reset_form():
    return None, None, False, False, "Your feedback will appear here after submission.", None, None


def format_file_info(uploaded_file) -> str:
    if uploaded_file is None:
        return "No file selected"
    p = Path(uploaded_file.name)
    return f"📄 File: {p.name} ({p.stat().st_size / 1024:.1f} KB)"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================
def build_interface():
    with gr.Blocks(
        title="🎓 MAFA",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    ) as demo:

        gr.Markdown("""
# 🎓 MAFA: Multimodal Assignment Feedback Assistant
**Intelligent Socratic Feedback for Machine Learning Assignments**

Upload your handwritten solution to a machine learning problem. Our system will:
1. **Transcribe** your handwritten math and diagrams
2. **Analyze** your work against a rubric
3. **Generate** personalized Socratic hints (no direct solutions)
4. **Provide** actionable feedback tied to course materials
---
""", elem_id="header")

        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                gr.Markdown("## 📝 Your Submission")

                file_input = gr.File(
                    label="📎 Upload Handwritten Solution",
                    file_count="single",
                    file_types=[".pdf", ".jpg", ".jpeg", ".png", ".webp"],
                    interactive=True,
                )
                text_input = gr.Textbox(
                    label="✍️ Or paste your solution text",
                    placeholder="Type your answer here (instead of uploading image)...",
                    lines=4,
                    interactive=True,
                )
                file_info = gr.Textbox(value="No file selected", label="File Status", interactive=False)
                file_input.change(fn=format_file_info, inputs=file_input, outputs=file_info)

                gr.Markdown("### 🏷️ Problem Topic")
                topic_dropdown = gr.Dropdown(
                    choices=TOPICS,
                    label="Select the ML topic for this assignment:",
                    interactive=True,
                    value=None,
                )

                gr.Markdown("### ⚙️ Options")
                show_rubric_ref = gr.Checkbox(label="Show rubric rule references", value=True)
                show_timestamps = gr.Checkbox(label="Show transcribed text & error analysis (debug)", value=False)

                gr.Markdown("### 🎬 Actions")
                submit_btn = gr.Button("📤 Get Feedback", variant="primary", size="lg")
                reset_btn = gr.Button("🔄 Clear Form", variant="secondary")

                gr.Markdown("""
### ℹ️ Tips for Best Results
- **Clear handwriting**: Write legibly with dark pen
- **Show your work**: Include all derivation steps
- **Label everything**: Mark variables, equations, and steps
- **Include diagrams**: Hand-drawn sketches are okay

**Processing time:** ~30-60 seconds on GPU (instant in mock demo)
""")

            with gr.Column(scale=1.2, min_width=400):
                gr.Markdown("## 📊 Feedback Card")
                feedback_output = gr.Markdown(
                    value="Your feedback will appear here after submission.",
                )

                gr.Markdown("---")
                with gr.Accordion("🔍 Debug Information (Transcription & Error Analysis)", open=False):
                    transcription_output = gr.Textbox(
                        label="Transcribed Content", interactive=False, lines=6, max_lines=12
                    )
                    error_analysis_output = gr.Code(
                        label="Error Analysis (JSON)", language="json", interactive=False
                    )

                gr.Markdown("""---
### 📚 About MAFA
MAFA combines:
- **Qwen2-VL-7B** for multimodal handwritten math transcription
- **ChromaDB RAG** for rubric-aligned error detection
- **Fine-tuned Phi-3-mini (LoRA)** for Socratic hint generation
- **Anti-Leakage Guardrails** for strict zero-answer pedagogical enforcement
""")

        submit_btn.click(
            fn=analyze_submission,
            inputs=[file_input, text_input, topic_dropdown, show_rubric_ref, show_timestamps],
            outputs=[feedback_output, transcription_output, error_analysis_output],
        )
        reset_btn.click(
            fn=reset_form,
            inputs=None,
            outputs=[file_input, topic_dropdown, show_rubric_ref, show_timestamps,
                     feedback_output, transcription_output, error_analysis_output],
        )

    return demo


if __name__ == "__main__":
    logger.info("🚀 Starting MAFA UI...")
    demo = build_interface()
    # On Kaggle, share=True creates a public gradio.live URL accessible from your browser
    demo.launch(share=IS_KAGGLE, server_name="0.0.0.0", server_port=7860, debug=True)
