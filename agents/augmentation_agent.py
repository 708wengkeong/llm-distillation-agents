"""
Augmentation Agent
------------------
Transforms validated teacher outputs into structured, training-ready
(question, scratchpad, answer) triples. Generates paraphrase variants
to increase dataset diversity.

Usage:
    python augmentation_agent.py
"""

import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are an Augmentation Agent preparing training data for LLM fine-tuning.
Format examples as clean (question, scratchpad, answer) triples.
The scratchpad should be a numbered, step-by-step reasoning trace.
Generate diverse paraphrase variants to increase dataset coverage.

Always respond with valid JSON only. No preamble, no markdown fences."""


def augment_single(quality_result: dict, topic: str = "") -> dict | None:
    """
    Format a single quality-passed example into training triples.

    Args:
        quality_result: Output dict from quality_agent.evaluate_single()
        topic:          Topic label for metadata

    Returns:
        Structured training triple dict, or None if example didn't pass
    """
    if not quality_result.get("passed"):
        return None

    example = quality_result["example"]

    user_prompt = f"""Convert this teacher output into structured training data.

ORIGINAL PROMPT: {example.get('prompt', '')}
REASONING TRACE: {example.get('reasoning_trace', '')}
FINAL ANSWER: {example.get('final_answer', '')}
QUALITY SCORE: {quality_result.get('overall_score', 0):.2f}
TOPIC: {topic or 'general'}

Produce:
1. A clean primary training triple
2. One surface-level paraphrase variant (same reasoning, different wording)

Return JSON with this exact structure:
{{
  "question": "clean, standalone question",
  "scratchpad": "1. step one\\n2. step two\\n3. step three",
  "answer": "concise final answer",
  "variants": [
    {{
      "question": "paraphrase of the question",
      "scratchpad": "paraphrased reasoning steps",
      "answer": "same final answer"
    }}
  ],
  "metadata": {{
    "topic": "{topic or 'general'}",
    "difficulty": "easy|medium|hard",
    "tags": ["tag1", "tag2"],
    "source_score": {quality_result.get('overall_score', 0):.3f},
    "strategy": "{example.get('strategy', 'cot')}"
  }}
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    result = json.loads(raw)

    print(
        f"[AugmentationAgent] Formatted triple — "
        f"difficulty={result['metadata'].get('difficulty')} "
        f"tags={result['metadata'].get('tags')}"
    )
    return result


def run(quality_outputs: list[dict], topic: str = "") -> list[dict]:
    """
    Augment all quality-passed examples.

    Args:
        quality_outputs: List of dicts from quality_agent.run()
        topic:           Topic label for metadata

    Returns:
        List of training triple dicts (only passed examples)
    """
    passed = [r for r in quality_outputs if r.get("passed")]
    print(
        f"[AugmentationAgent] Processing {len(passed)}/{len(quality_outputs)} "
        f"passed examples"
    )

    results = []
    for qr in passed:
        triple = augment_single(qr, topic=topic)
        if triple:
            results.append(triple)

    total_with_variants = sum(1 + len(r.get("variants", [])) for r in results)
    print(
        f"[AugmentationAgent] Done — {len(results)} primaries, "
        f"{total_with_variants} total examples (incl. variants)"
    )
    return results


if __name__ == "__main__":
    sample_quality_output = [
        {
            "example": {
                "prompt": "A train travels at 60 km/h. How long does it take to travel 210 km?",
                "reasoning_trace": "Step 1: Use time = distance / speed.\nStep 2: time = 210 / 60 = 3.5 hours.",
                "final_answer": "3.5 hours",
                "strategy": "cot",
            },
            "scores": {"correctness": 1.0, "clarity": 0.9, "pedagogical_value": 0.85, "completeness": 0.9},
            "overall_score": 0.93,
            "verdict": "pass",
            "improvement_notes": [],
            "passed": True,
        }
    ]

    output = run(sample_quality_output, topic="arithmetic reasoning")
    print("\n--- Augmentation Agent Output ---")
    print(json.dumps(output, indent=2))
