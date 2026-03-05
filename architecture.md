# Architecture: LLM Distillation Multi-Agent Pipeline

## Overview

This pipeline automates the process of knowledge distillation from a large teacher LLM to a smaller student model. Rather than a monolithic script, the work is divided into five specialized agents, each owning a clearly scoped responsibility.

The system is designed to be **iterative** — the Eval Agent's output closes the loop back into the Curriculum Agent, enabling progressive refinement across distillation rounds.

---

## Agent Responsibilities

| Agent | Input | Output |
|---|---|---|
| Curriculum Agent | Topic, student profile, eval gaps | Prioritized prompt list |
| Teacher Agent | Prompt list | CoT traces + final answers |
| Quality/Filter Agent | Teacher outputs | Scored, filtered examples |
| Augmentation Agent | Filtered examples | Training triples + variants |
| Eval Agent | Training data + student benchmarks | Capability gap report |

---

## Data Flow

```
┌─────────────────────────────────────────────────┐
│                 DISTILLATION LOOP                │
│                                                  │
│  ┌──────────────────┐                            │
│  │  Curriculum Agent │◄──── Eval gaps (iter N-1) │
│  └────────┬─────────┘                            │
│           │ prompt list                          │
│  ┌────────▼─────────┐                            │
│  │   Teacher Agent   │                            │
│  └────────┬─────────┘                            │
│           │ raw traces                           │
│  ┌────────▼─────────┐                            │
│  │  Quality Agent    │                            │
│  └────────┬─────────┘                            │
│           │ scored + filtered examples           │
│  ┌────────▼──────────┐                           │
│  │ Augmentation Agent │                           │
│  └────────┬──────────┘                           │
│           │ (Q, scratchpad, A) triples           │
│  ┌────────▼─────────┐                            │
│  │    Eval Agent     │──── gaps → Curriculum     │
│  └──────────────────┘                            │
└─────────────────────────────────────────────────┘
```

---

## Agent Communication Protocol

Agents pass Python dicts between each other. The shared schema:

```python
# Curriculum → Teacher
{
  "topic": str,
  "student_profile": str,        # "small" | "medium" | "large"
  "strategy": str,               # "cot" | "rationale" | "selfplay"
  "prompts": list[str],          # selected training prompts
  "iteration": int
}

# Teacher → Quality
{
  "prompt": str,
  "reasoning_trace": str,        # full chain-of-thought
  "final_answer": str,
  "model": str                   # teacher model used
}

# Quality → Augmentation
{
  "example": dict,               # original teacher output
  "scores": dict,                # per-dimension scores
  "overall_score": float,        # 0.0 - 1.0
  "verdict": str,                # "pass" | "conditional" | "reject"
  "passed": bool
}

# Augmentation → Eval / Training
{
  "question": str,
  "scratchpad": str,
  "answer": str,
  "variants": list[dict],
  "metadata": dict               # difficulty, tags, etc.
}

# Eval → Curriculum (next iteration)
{
  "benchmark_delta": float,      # estimated improvement %
  "remaining_gaps": list[str],
  "next_focus_areas": list[str],
  "convergence_estimate": int    # rounds remaining
}
```

---

## Distillation Strategies

### Chain-of-Thought (CoT)
Teacher generates explicit step-by-step reasoning. Student learns to replicate the scratchpad pattern. Best for reasoning-heavy tasks (math, logic, code).

### Rationale Augmentation
Teacher appends post-hoc explanations to existing answers. Lower cost, good for classification or factual tasks.

### Self-Play / ReST Loop
Student generates candidate answers → Teacher scores them → Best candidates loop back as new training data. Iterative and data-efficient. Good for tasks with clear reward signals.

---

## Scaling Considerations

- **Parallelism**: Teacher Agent calls can be batched; run Quality Agent concurrently with new Teacher calls
- **Multi-teacher routing**: Route different capability domains to specialized teachers
- **Online distillation**: Flag low-confidence student outputs in production → queue to Teacher → retrain
