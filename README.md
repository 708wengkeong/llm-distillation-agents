# LLM Distillation Agents

A multi-agent pipeline for distilling knowledge from large teacher models into smaller student models. Each agent owns a distinct stage of the distillation loop.

## Folder Structure

```
llm-distillation-agents/
├── README.md                        # This file
├── agents/
│   ├── curriculum_agent.py          # Selects optimal training prompts
│   ├── teacher_agent.py             # Generates gold-standard reasoning traces
│   ├── quality_agent.py             # Scores and filters teacher outputs
│   ├── augmentation_agent.py        # Formats into training-ready triples
│   └── eval_agent.py                # Benchmarks student improvement
└── docs/
    ├── architecture.md              # Full pipeline architecture
    ├── curriculum_agent.md          # Curriculum Agent spec
    ├── teacher_agent.md             # Teacher Agent spec
    ├── quality_agent.md             # Quality/Filter Agent spec
    ├── augmentation_agent.md        # Augmentation Agent spec
    └── eval_agent.md                # Eval Agent spec
```

## Pipeline Overview

```
Curriculum Agent
    ↓  prompt selection targeting capability gaps
Teacher Agent
    ↓  CoT reasoning traces + completions
Quality/Filter Agent
    ↓  scored, validated supervision signal
Augmentation Agent
    ↓  (question, scratchpad, answer) triples
Eval Agent
    ↓  benchmark delta + next-iteration recommendations
    └→  feeds back into Curriculum Agent
```

## Quickstart

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here

# Run individual agents
python agents/curriculum_agent.py

# Run agents in sequence manually or wire them via an orchestrator
```

## Key Design Principles

- **Single responsibility** — each agent does exactly one job
- **Composable** — agents communicate via structured dicts, easy to swap
- **Observable** — every agent logs its inputs, outputs, and quality scores
- **Closable loop** — eval feeds back into curriculum for iterative improvement

## Dependencies

- `anthropic` — Claude API client
- `python >= 3.10`
