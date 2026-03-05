"""
Curriculum Agent
----------------
Selects and prioritizes training prompts that target the student
model's current capability gaps. Applies active learning principles —
prefers examples at the boundary of the student's current ability.

Usage:
    python curriculum_agent.py
"""

import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a Curriculum Agent in an LLM distillation pipeline.
Your job is to select training prompts that maximally target the student model's capability gaps.
Apply active learning principles: prefer examples at the boundary of the student's current ability.

Always respond with valid JSON only. No preamble, no markdown fences."""

STUDENT_PROFILES = {
    "small":  "1-3B parameter model needing very explicit, step-by-step guidance",
    "medium": "7-13B parameter model benefiting from moderate complexity examples",
    "large":  "30B+ parameter model that can handle nuanced, advanced reasoning",
}


def run(
    topic: str,
    student_profile: str = "medium",
    strategy: str = "cot",
    eval_gaps: list[str] | None = None,
    iteration: int = 1,
    seed_prompt: str = "",
) -> dict:
    """
    Run the Curriculum Agent.

    Args:
        topic:           Skill/domain to distill (e.g. "arithmetic reasoning")
        student_profile: "small" | "medium" | "large"
        strategy:        "cot" | "rationale" | "selfplay"
        eval_gaps:       Capability gaps from previous Eval Agent run
        iteration:       Current distillation round
        seed_prompt:     Optional hint to bias prompt selection

    Returns:
        dict with keys: topic, student_profile, strategy, prompts,
                        rationale, iteration
    """
    eval_gaps = eval_gaps or []
    profile_desc = STUDENT_PROFILES.get(student_profile, STUDENT_PROFILES["medium"])

    gap_section = ""
    if eval_gaps:
        gap_section = f"\nKnown capability gaps from previous eval:\n" + "\n".join(
            f"  - {g}" for g in eval_gaps
        )

    seed_section = f"\nSeed prompt hint: {seed_prompt}" if seed_prompt else ""

    user_prompt = f"""Select optimal training prompts for distillation.

Topic: {topic}
Student model: {profile_desc}
Strategy: {strategy}
Iteration: {iteration}{gap_section}{seed_section}

{"On iteration 1, sample broadly across easy → medium → hard difficulty." if iteration == 1 else "Focus narrowly on the known capability gaps listed above."}

Return JSON with this exact structure:
{{
  "topic": "{topic}",
  "student_profile": "{student_profile}",
  "strategy": "{strategy}",
  "iteration": {iteration},
  "prompts": [
    "prompt 1 text",
    "prompt 2 text",
    "prompt 3 text"
  ],
  "rationale": "brief explanation of why these prompts were chosen"
}}"""

    print(f"[CurriculumAgent] Running — topic='{topic}', iteration={iteration}")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    result = json.loads(raw)

    print(f"[CurriculumAgent] Selected {len(result['prompts'])} prompts")
    return result


if __name__ == "__main__":
    output = run(
        topic="chain-of-thought arithmetic reasoning",
        student_profile="medium",
        strategy="cot",
        iteration=1,
    )
    print("\n--- Curriculum Agent Output ---")
    print(json.dumps(output, indent=2))
