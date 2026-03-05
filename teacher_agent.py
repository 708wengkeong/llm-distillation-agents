"""
Teacher Agent
-------------
Wraps the large teacher model and generates gold-standard reasoning
traces (chain-of-thought) and final answers that become training data
for the student model.

Usage:
    python teacher_agent.py
"""

import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a Teacher Agent generating training data for a smaller student LLM.
Produce maximally clear, step-by-step reasoning traces. Show every intermediate step.
Your output will be used directly as supervised fine-tuning data, so pedagogical clarity is critical.

Always respond with valid JSON only. No preamble, no markdown fences."""

VERBOSITY = {
    "small":  "Be extremely explicit. Number every step. Define every term. Assume minimal prior knowledge.",
    "medium": "Be clear and moderately detailed. Show all key reasoning steps.",
    "large":  "Be concise but complete. Assume solid background knowledge. Focus on non-obvious steps.",
}


def run_single(prompt: str, student_profile: str = "medium", strategy: str = "cot") -> dict:
    """
    Generate a teacher trace for a single prompt.

    Args:
        prompt:          The training prompt to respond to
        student_profile: Informs verbosity/depth of trace
        strategy:        "cot" | "rationale" | "selfplay"

    Returns:
        dict with keys: prompt, reasoning_trace, final_answer, model, strategy
    """
    verbosity_note = VERBOSITY.get(student_profile, VERBOSITY["medium"])

    if strategy == "cot":
        instruction = "Generate a full chain-of-thought reasoning trace, then a clearly marked final answer."
    elif strategy == "rationale":
        instruction = "Give the final answer first, then append a post-hoc rationale explaining the reasoning."
    else:  # selfplay
        instruction = "Generate a candidate answer with reasoning. This will be scored and may be used as training data."

    user_prompt = f"""Generate a teacher response for this training prompt.

Prompt: {prompt}
Verbosity guidance: {verbosity_note}
Task: {instruction}

Return JSON with this exact structure:
{{
  "prompt": "{prompt}",
  "reasoning_trace": "full step-by-step reasoning here",
  "final_answer": "concise final answer here",
  "model": "claude-sonnet-4-20250514",
  "strategy": "{strategy}"
}}"""

    print(f"[TeacherAgent] Generating trace for: '{prompt[:60]}...'")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    return json.loads(raw)


def run(curriculum_output: dict) -> list[dict]:
    """
    Run the Teacher Agent over all prompts from the Curriculum Agent.

    Args:
        curriculum_output: Output dict from curriculum_agent.run()

    Returns:
        List of teacher trace dicts, one per prompt
    """
    prompts = curriculum_output["prompts"]
    profile = curriculum_output["student_profile"]
    strategy = curriculum_output["strategy"]

    print(f"[TeacherAgent] Generating {len(prompts)} traces — strategy={strategy}")

    results = []
    for prompt in prompts:
        trace = run_single(prompt, student_profile=profile, strategy=strategy)
        results.append(trace)

    print(f"[TeacherAgent] Done — {len(results)} traces generated")
    return results


if __name__ == "__main__":
    # Standalone demo
    sample_curriculum = {
        "topic": "arithmetic reasoning",
        "student_profile": "medium",
        "strategy": "cot",
        "iteration": 1,
        "prompts": [
            "A train travels at 60 km/h. How long does it take to travel 210 km?",
        ],
    }

    output = run(sample_curriculum)
    print("\n--- Teacher Agent Output ---")
    print(json.dumps(output, indent=2))
