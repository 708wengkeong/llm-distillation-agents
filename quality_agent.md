# Quality / Filter Agent

## Role

The Quality Agent is the **gatekeeper** of the pipeline. It evaluates every teacher output before it can become training data, scoring it across multiple dimensions and rejecting examples that would introduce noise or errors into the student's fine-tuning dataset.

## Inputs

```python
[
  {
    "prompt": str,
    "reasoning_trace": str,
    "final_answer": str,
    "model": str
  },
  ...
]
```

## Outputs

```python
[
  {
    "example": dict,           # original teacher output
    "scores": {
      "correctness": float,    # 0.0 - 1.0
      "clarity": float,
      "pedagogical_value": float,
      "completeness": float
    },
    "overall_score": float,
    "verdict": str,            # "pass" | "conditional" | "reject"
    "improvement_notes": list[str],
    "passed": bool
  },
  ...
]
```

## Scoring Dimensions

| Dimension | Description | Weight |
|---|---|---|
| Correctness | Is the answer factually/logically correct? | 40% |
| Clarity | Is the reasoning easy to follow? | 25% |
| Pedagogical Value | Will this help the student model generalize? | 20% |
| Completeness | Are all reasoning steps shown? | 15% |

## Verdict Thresholds

- **PASS**: overall_score ≥ 0.75 → included as-is
- **CONDITIONAL**: 0.5 ≤ score < 0.75 → flagged for review, may be included with notes
- **REJECT**: score < 0.5 → dropped from training data

## Design Notes

- Uses a separate LLM call (not the same as teacher) to avoid self-evaluation bias
- Rejection rate is logged and fed back to the Curriculum Agent — high rejection rates signal the curriculum is too hard or the topic is underspecified
- In a production setup, a trained reward model can replace the LLM-based quality check for speed

## System Prompt

```
You are a Quality Agent evaluating LLM-generated training data.
Be rigorous and critical. A bad training example is worse than
no training example — it will corrupt the student model.
Score each example honestly across all dimensions.
Output structured JSON only.
```
