# Augmentation Agent

## Role

The Augmentation Agent transforms validated teacher outputs into **structured, training-ready data**. It formats examples into `(question, scratchpad, answer)` triples — the standard format for SFT (supervised fine-tuning) — and generates paraphrase variants to increase dataset diversity.

## Inputs

```python
[
  {
    "example": {
      "prompt": str,
      "reasoning_trace": str,
      "final_answer": str
    },
    "scores": dict,
    "verdict": str,
    "passed": bool
  },
  ...
]
# Only passed=True examples are processed
```

## Outputs

```python
[
  {
    "question": str,           # clean, standalone question
    "scratchpad": str,         # formatted reasoning steps
    "answer": str,             # final answer only
    "variants": [
      {
        "question": str,       # surface paraphrase
        "scratchpad": str,
        "answer": str
      },
      ...                      # 1-2 variants per example
    ],
    "metadata": {
      "topic": str,
      "difficulty": str,       # "easy" | "medium" | "hard"
      "tags": list[str],
      "source_score": float,
      "strategy": str
    }
  },
  ...
]
```

## Augmentation Techniques

- **Surface paraphrase**: Reword the question without changing the underlying reasoning path. Increases robustness to phrasing variation.
- **Abstraction lift**: Replace concrete values with variables or generalize the scenario. Improves the student's ability to generalize.
- **Difficulty scaling**: For conditional-pass examples, simplify the reasoning trace to reduce noise.

## Design Notes

- The `scratchpad` format is deliberately verbose — students learn best from maximally explicit traces
- Tags are used downstream to ensure balanced coverage across subtopics
- In the self-play strategy, variants are used as candidate prompts for the next Teacher Agent call

## System Prompt

```
You are an Augmentation Agent preparing training data for LLM fine-tuning.
Format examples as clean (question, scratchpad, answer) triples.
The scratchpad should be a numbered, step-by-step reasoning trace.
Generate diverse paraphrase variants. Output structured JSON only.
```
