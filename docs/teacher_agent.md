# Teacher Agent

## Role

The Teacher Agent wraps the large, capable teacher model and generates **gold-standard supervision signals** — detailed reasoning traces, chain-of-thought steps, and final answers that will become training data for the student model.

## Inputs

```python
{
  "prompts": list[str],      # from Curriculum Agent
  "strategy": str,           # "cot" | "rationale" | "selfplay"
  "topic": str,
  "student_profile": str     # informs verbosity/depth of traces
}
```

## Outputs

```python
[
  {
    "prompt": str,
    "reasoning_trace": str,   # full step-by-step CoT
    "final_answer": str,
    "model": str,             # which teacher model was used
    "strategy": str
  },
  ...  # one per input prompt
]
```

## Design Notes

- **Verbosity scales with student size**: small student → maximally explicit traces; large student → concise, nuanced reasoning
- **Self-consistency sampling**: optionally generates multiple traces per prompt and selects the majority-vote answer to improve reliability
- **Multi-teacher routing**: different prompts can be routed to different teacher models (e.g. code → DeepSeek, math → o3-mini)
- Teacher outputs are not used directly as training data — they pass through the Quality Agent first

## System Prompt

```
You are a Teacher Agent generating training data for a student LLM.
Produce maximally clear, step-by-step reasoning traces.
Show every intermediate step. Your output will be used directly
as supervised fine-tuning data, so pedagogical clarity is critical.
```
