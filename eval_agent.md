# Eval Agent

## Role

The Eval Agent **closes the loop**. After each distillation round, it benchmarks the student model against held-out tasks, measures capability deltas, identifies remaining gaps, and generates recommendations that feed directly back into the Curriculum Agent for the next iteration.

## Inputs

```python
{
  "training_data": list[dict],     # output from Augmentation Agent
  "student_profile": str,
  "topic": str,
  "iteration": int,
  "previous_gaps": list[str],      # gaps from last iteration (if any)
  "benchmark_results": dict        # optional: real student eval scores
}
```

## Outputs

```python
{
  "iteration": int,
  "benchmark_delta": float,        # estimated improvement % this round
  "capability_scores": {
    "subtopic_A": float,           # 0.0 - 1.0 per subtopic
    "subtopic_B": float,
    ...
  },
  "remaining_gaps": list[str],     # capabilities still underperforming
  "next_focus_areas": list[str],   # recommendations for curriculum
  "convergence_estimate": int,     # estimated rounds to convergence
  "data_quality_summary": {
    "total_examples": int,
    "pass_rate": float,
    "avg_quality_score": float
  },
  "recommendation": str            # free-text summary for the operator
}
```

## Evaluation Modes

### Simulated (default)
The Eval Agent uses the LLM to reason about likely student improvement based on the training data produced. Fast, no infrastructure needed. Useful for pipeline development and iteration planning.

### Live Benchmark (production)
Pass actual student model eval results in `benchmark_results`. The agent interprets scores, computes deltas against a baseline, and diagnoses remaining weaknesses. Requires running the student model on a held-out eval set between iterations.

## Design Notes

- **Convergence detection**: if benchmark_delta < 2% for two consecutive rounds on the same subtopic, flag for curriculum shift rather than more of the same
- **Gap prioritization**: ranks remaining gaps by frequency in failed examples, not just by severity
- **Loop closure**: `next_focus_areas` becomes `eval_gaps` input to the Curriculum Agent in the next iteration — this is the mechanism that makes the pipeline adaptive

## System Prompt

```
You are an Eval Agent analyzing distillation pipeline results.
Think like an ML researcher: be quantitative, identify root causes
of remaining weaknesses, and give actionable recommendations.
Your output feeds directly into the next training iteration.
Output structured JSON only.
```
