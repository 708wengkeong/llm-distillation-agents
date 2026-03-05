# Curriculum Agent

## Role

The Curriculum Agent decides **what** to teach the student model. It selects and prioritizes training prompts that target the student's current capability gaps, applying principles from active learning and curriculum learning theory.

## Inputs

| Field | Type | Description |
|---|---|---|
| `topic` | str | Skill or knowledge domain to distill |
| `student_profile` | str | Size/capability tier: `"small"`, `"medium"`, `"large"` |
| `strategy` | str | Distillation strategy: `"cot"`, `"rationale"`, `"selfplay"` |
| `eval_gaps` | list[str] | Capability gaps from previous Eval Agent run (empty on iter 1) |
| `iteration` | int | Current distillation round number |

## Outputs

```python
{
  "topic": str,
  "student_profile": str,
  "strategy": str,
  "prompts": list[str],     # 3-5 selected training prompts
  "rationale": str,         # why these prompts were chosen
  "iteration": int
}
```

## Design Notes

- On iteration 1, samples broadly across the topic difficulty spectrum (easy → hard)
- On subsequent iterations, narrows focus to gaps reported by the Eval Agent
- Applies a "start hard, scale down" philosophy — surfaces hard examples early to maximize information per training step
- Can optionally be seeded with a custom prompt to bias selection

## System Prompt

```
You are a Curriculum Agent in an LLM distillation pipeline.
Your job is to select training prompts that maximally target the
student model's capability gaps. Apply active learning principles:
prefer examples at the boundary of the student's current ability.
Output structured JSON only.
```
