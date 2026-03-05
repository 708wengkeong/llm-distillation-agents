"""
Quality / Filter Agent
----------------------
Evaluates teacher model outputs before they become training data.
Scores on correctness, clarity, pedagogical value, and completeness.
Rejects examples that would introduce noise into the student's dataset.

Usage:
    python quality_agent.py
"""

import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a Quality Agent evaluating LLM-generated training data.
Be rigorous and critical. A bad training example is worse than no training example.
Score each example honestly across all dimensions.

Always respond with valid JSON only. No preamble, no markdown fences."""

# Verdict thresholds
PASS_THRESHOLD = 0.75
CONDITIONAL_THRESHOLD = 0.50

# Scoring weights
WEIGHTS = {
    "correctness": 0.40,
    "clarity": 0.25,
    "pedagogical_value": 0.20,
    "completeness": 0.15,
}


def weighted_score(scores: dict) -> float:
    return sum(scores.get(k, 0) * w for k, w in WEIGHTS.items())


def verdict_from_score(score: float) -> str:
    if score >= PASS_THRESHOLD:
        return "pass"
    elif score >= CONDITIONAL_THRESHOLD:
        return "conditional"
    return "reject"


def evaluate_single(example: dict) -> dict:
    """
    Evaluate a single teacher output.

    Args:
        example: dict with keys prompt, reasoning_trace, final_answer

    Returns:
        dict with keys: example, scores, overall_score, verdict,
                        improvement_notes, passed
    """
    user_prompt = f"""Evaluate this LLM-generated training example.

PROMPT: {example.get('prompt', '')}
REASONING TRACE: {example.get('reasoning_trace', '')}
FINAL ANSWER: {example.get('final_answer', '')}

Score each dimension from 0.0 to 1.0:
- correctness: Is the answer factually/logically correct?
- clarity: Is the reasoning easy to follow?
- pedagogical_value: Will this help a student model generalize?
- completeness: Are all reasoning steps shown?

Return JSON with this exact structure:
{{
  "scores": {{
    "correctness": 0.0,
    "clarity": 0.0,
    "pedagogical_value": 0.0,
    "completeness": 0.0
  }},
  "improvement_notes": ["note 1", "note 2"]
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    eval_result = json.loads(raw)

    scores = eval_result["scores"]
    overall = weighted_score(scores)
    verdict = verdict_from_score(overall)

    result = {
        "example": example,
        "scores": scores,
        "overall_score": round(overall, 3),
        "verdict": verdict,
        "improvement_notes": eval_result.get("improvement_notes", []),
        "passed": verdict in ("pass", "conditional"),
    }

    print(
        f"[QualityAgent] Prompt='{example.get('prompt','')[:40]}...' "
        f"score={overall:.2f} verdict={verdict.upper()}"
    )
    return result


def run(teacher_outputs: list[dict]) -> list[dict]:
    """
    Evaluate all teacher outputs.

    Args:
        teacher_outputs: List of dicts from teacher_agent.run()

    Returns:
        List of evaluation result dicts
    """
    print(f"[QualityAgent] Evaluating {len(teacher_outputs)} examples")

    results = [evaluate_single(ex) for ex in teacher_outputs]

    passed = sum(1 for r in results if r["passed"])
    avg_score = sum(r["overall_score"] for r in results) / len(results) if results else 0

    print(
        f"[QualityAgent] Done — {passed}/{len(results)} passed, "
        f"avg_score={avg_score:.2f}"
    )
    return results


if __name__ == "__main__":
    sample_teacher_outputs = [
        {
            "prompt": "A train travels at 60 km/h. How long does it take to travel 210 km?",
            "reasoning_trace": "Step 1: Use time = distance / speed. Step 2: time = 210 / 60 = 3.5 hours.",
            "final_answer": "3.5 hours",
            "model": "claude-sonnet-4-20250514",
            "strategy": "cot",
        }
    ]

    output = run(sample_teacher_outputs)
    print("\n--- Quality Agent Output ---")
    print(json.dumps(output, indent=2))
