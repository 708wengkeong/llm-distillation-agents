"""
Eval Agent
----------
Benchmarks student model improvement after each distillation round.
Identifies remaining capability gaps and generates recommendations
that feed back into the Curriculum Agent for the next iteration.

Usage:
    python eval_agent.py
"""

import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are an Eval Agent analyzing distillation pipeline results.
Think like an ML researcher: be quantitative, identify root causes of remaining weaknesses,
and give actionable recommendations that will guide the next training iteration.

Always respond with valid JSON only. No preamble, no markdown fences."""


def run(
    training_data: list[dict],
    topic: str,
    student_profile: str = "medium",
    iteration: int = 1,
    previous_gaps: list[str] | None = None,
    benchmark_results: dict | None = None,
) -> dict:
    """
    Run the Eval Agent to close the distillation loop.

    Args:
        training_data:     Output from augmentation_agent.run()
        topic:             Skill/domain being distilled
        student_profile:   "small" | "medium" | "large"
        iteration:         Current distillation round number
        previous_gaps:     Capability gaps from previous iteration
        benchmark_results: Optional real student eval scores dict

    Returns:
        dict with benchmark_delta, remaining_gaps, next_focus_areas,
              convergence_estimate, data_quality_summary, recommendation
    """
    previous_gaps = previous_gaps or []

    # Compute data quality summary locally
    total = len(training_data)
    total_with_variants = sum(1 + len(d.get("variants", [])) for d in training_data)
    avg_score = (
        sum(d.get("metadata", {}).get("source_score", 0.0) for d in training_data) / total
        if total else 0.0
    )
    difficulty_counts = {}
    all_tags = []
    for d in training_data:
        meta = d.get("metadata", {})
        diff = meta.get("difficulty", "unknown")
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        all_tags.extend(meta.get("tags", []))

    data_summary = {
        "total_primary_examples": total,
        "total_with_variants": total_with_variants,
        "avg_quality_score": round(avg_score, 3),
        "difficulty_distribution": difficulty_counts,
        "unique_tags": list(set(all_tags)),
    }

    benchmark_section = ""
    if benchmark_results:
        benchmark_section = f"\nActual student benchmark results:\n{json.dumps(benchmark_results, indent=2)}"

    prev_gap_section = ""
    if previous_gaps:
        prev_gap_section = f"\nPrevious iteration gaps:\n" + "\n".join(
            f"  - {g}" for g in previous_gaps
        )

    # Sample training examples for context
    sample_questions = [d.get("question", "") for d in training_data[:3]]
    sample_tags = list(set(all_tags))[:8]

    user_prompt = f"""Analyze this distillation round and generate recommendations.

Topic: {topic}
Student profile: {student_profile}
Iteration: {iteration}
Training data summary: {json.dumps(data_summary, indent=2)}
Sample questions covered: {json.dumps(sample_questions, indent=2)}
Tags covered: {sample_tags}{prev_gap_section}{benchmark_section}

Return JSON with this exact structure:
{{
  "iteration": {iteration},
  "benchmark_delta": 0.0,
  "capability_scores": {{
    "subtopic_name": 0.0
  }},
  "remaining_gaps": ["gap 1", "gap 2", "gap 3"],
  "next_focus_areas": ["focus 1", "focus 2", "focus 3"],
  "convergence_estimate": 3,
  "data_quality_summary": {json.dumps(data_summary)},
  "recommendation": "free-text summary for the operator"
}}

For benchmark_delta: estimate % improvement this round will likely yield.
For capability_scores: score 0.0-1.0 per subtopic covered.
For convergence_estimate: how many more rounds until the student converges on this topic."""

    print(f"[EvalAgent] Analyzing iteration {iteration} — {total} training examples")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    result = json.loads(raw)

    print(
        f"[EvalAgent] Done — delta={result.get('benchmark_delta')}% "
        f"gaps={len(result.get('remaining_gaps', []))} "
        f"est_rounds={result.get('convergence_estimate')}"
    )
    return result


if __name__ == "__main__":
    sample_training_data = [
        {
            "question": "A train travels at 60 km/h. How long to travel 210 km?",
            "scratchpad": "1. Use time = distance / speed\n2. time = 210 / 60\n3. time = 3.5 hours",
            "answer": "3.5 hours",
            "variants": [{"question": "If a train moves at 60 km/h, what is the travel time for 210 km?",
                          "scratchpad": "1. Formula: t = d/s\n2. t = 210/60 = 3.5",
                          "answer": "3.5 hours"}],
            "metadata": {"topic": "arithmetic reasoning", "difficulty": "medium",
                         "tags": ["speed", "distance", "time"], "source_score": 0.93, "strategy": "cot"},
        }
    ]

    output = run(
        training_data=sample_training_data,
        topic="arithmetic reasoning",
        student_profile="medium",
        iteration=1,
    )
    print("\n--- Eval Agent Output ---")
    print(json.dumps(output, indent=2))
