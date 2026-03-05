"""
pipeline.py — LLM Distillation Orchestrator
--------------------------------------------
Chains all 5 agents end-to-end with a CLI interface.
Supports single-run and multi-iteration loop modes.

Usage:
    # Single iteration
    python pipeline.py --topic "arithmetic reasoning"

    # Multiple iterations (closes the eval → curriculum loop)
    python pipeline.py --topic "code debugging" --iterations 3

    # Full options
    python pipeline.py \\
        --topic "chain-of-thought math" \\
        --student medium \\
        --strategy cot \\
        --iterations 2 \\
        --output ./runs \\
        --seed "focus on multi-step word problems"
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path

# ── Agent imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "agents"))
import curriculum_agent
import teacher_agent
import quality_agent
import augmentation_agent
import eval_agent


# ── Pretty printing ──────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"

AGENT_COLORS = {
    "curriculum":    "\033[96m",   # cyan
    "teacher":       "\033[95m",   # magenta
    "quality":       "\033[93m",   # yellow
    "augmentation":  "\033[92m",   # green
    "eval":          "\033[91m",   # red
}

AGENT_ICONS = {
    "curriculum":   "🎯",
    "teacher":      "🧠",
    "quality":      "🔍",
    "augmentation": "✨",
    "eval":         "📊",
}

def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════╗
║        LLM DISTILLATION AGENT PIPELINE          ║
║        claude-sonnet-4 · multi-agent            ║
╚══════════════════════════════════════════════════╝{RESET}
""")

def section(agent_id: str, label: str):
    color = AGENT_COLORS.get(agent_id, CYAN)
    icon  = AGENT_ICONS.get(agent_id, "▸")
    print(f"\n{color}{BOLD}{'─' * 50}")
    print(f"  {icon}  {label}")
    print(f"{'─' * 50}{RESET}")

def summary_line(label: str, value):
    print(f"  {DIM}{label:<28}{RESET}{BOLD}{value}{RESET}")


# ── Run a single distillation iteration ─────────────────────────────────────
def run_iteration(
    topic: str,
    student_profile: str,
    strategy: str,
    iteration: int,
    eval_gaps: list[str],
    seed_prompt: str,
    output_dir: Path,
) -> tuple[dict, list[str]]:
    """
    Run one full pass through all 5 agents.

    Returns:
        (eval_report, next_gaps)
    """
    iter_label = f"ITERATION {iteration}"
    print(f"\n{BOLD}{CYAN}{'═' * 50}")
    print(f"  {iter_label}")
    print(f"{'═' * 50}{RESET}")

    iter_dir = output_dir / f"iteration_{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Curriculum Agent ──────────────────────────────────────────────────
    section("curriculum", "Curriculum Agent — selecting prompts")
    curriculum_out = curriculum_agent.run(
        topic=topic,
        student_profile=student_profile,
        strategy=strategy,
        eval_gaps=eval_gaps,
        iteration=iteration,
        seed_prompt=seed_prompt if iteration == 1 else "",
    )
    save(iter_dir / "01_curriculum.json", curriculum_out)
    summary_line("Prompts selected:", len(curriculum_out["prompts"]))
    for i, p in enumerate(curriculum_out["prompts"], 1):
        print(f"    {DIM}{i}. {p[:72]}{'…' if len(p) > 72 else ''}{RESET}")

    # ── 2. Teacher Agent ─────────────────────────────────────────────────────
    section("teacher", "Teacher Agent — generating reasoning traces")
    teacher_out = teacher_agent.run(curriculum_out)
    save(iter_dir / "02_teacher.json", teacher_out)
    summary_line("Traces generated:", len(teacher_out))

    # ── 3. Quality / Filter Agent ────────────────────────────────────────────
    section("quality", "Quality Agent — scoring & filtering")
    quality_out = quality_agent.run(teacher_out)
    save(iter_dir / "03_quality.json", quality_out)

    passed  = sum(1 for r in quality_out if r["passed"])
    avg_score = sum(r["overall_score"] for r in quality_out) / len(quality_out) if quality_out else 0
    summary_line("Pass rate:", f"{passed}/{len(quality_out)}")
    summary_line("Avg quality score:", f"{avg_score:.2f}")

    rejected = [r for r in quality_out if not r["passed"]]
    if rejected:
        print(f"  {YELLOW}  ⚠  {len(rejected)} example(s) rejected{RESET}")

    if passed == 0:
        print(f"\n  {RED}✗  No examples passed quality filter. "
              f"Try a different topic or seed.{RESET}")
        return {}, []

    # ── 4. Augmentation Agent ────────────────────────────────────────────────
    section("augmentation", "Augmentation Agent — formatting training triples")
    aug_out = augmentation_agent.run(quality_out, topic=topic)
    save(iter_dir / "04_augmentation.json", aug_out)

    total_examples = sum(1 + len(d.get("variants", [])) for d in aug_out)
    summary_line("Primary triples:", len(aug_out))
    summary_line("Total w/ variants:", total_examples)

    # Also save as JSONL for direct fine-tuning use
    save_jsonl(iter_dir / "training_data.jsonl", aug_out)
    summary_line("Training JSONL saved:", str(iter_dir / "training_data.jsonl"))

    # ── 5. Eval Agent ────────────────────────────────────────────────────────
    section("eval", "Eval Agent — benchmarking & gap analysis")
    eval_out = eval_agent.run(
        training_data=aug_out,
        topic=topic,
        student_profile=student_profile,
        iteration=iteration,
        previous_gaps=eval_gaps,
    )
    save(iter_dir / "05_eval.json", eval_out)

    summary_line("Est. benchmark delta:", f"+{eval_out.get('benchmark_delta', 0):.1f}%")
    summary_line("Remaining gaps:", len(eval_out.get("remaining_gaps", [])))
    summary_line("Convergence estimate:", f"{eval_out.get('convergence_estimate', '?')} more rounds")

    gaps = eval_out.get("remaining_gaps", [])
    if gaps:
        print(f"\n  {DIM}Gaps → next curriculum:{RESET}")
        for g in gaps[:4]:
            print(f"    {DIM}• {g}{RESET}")

    rec = eval_out.get("recommendation", "")
    if rec:
        print(f"\n  {DIM}Recommendation: {rec[:120]}{'…' if len(rec) > 120 else ''}{RESET}")

    return eval_out, eval_out.get("next_focus_areas", gaps)


# ── Helpers ──────────────────────────────────────────────────────────────────
def save(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def save_jsonl(path: Path, triples: list[dict]):
    """Save augmented triples as JSONL for fine-tuning."""
    with open(path, "w") as f:
        for item in triples:
            # Primary example
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": f"{item['scratchpad']}\n\nAnswer: {item['answer']}"},
                ]
            }) + "\n")
            # Variants
            for v in item.get("variants", []):
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": v["question"]},
                        {"role": "assistant", "content": f"{v['scratchpad']}\n\nAnswer: {v['answer']}"},
                    ]
                }) + "\n")


# ── CLI entrypoint ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="LLM Distillation Agent Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --topic "arithmetic reasoning"
  python pipeline.py --topic "code debugging" --iterations 3 --student small
  python pipeline.py --topic "SQL query writing" --strategy rationale --output ./my_runs
        """
    )
    p.add_argument("--topic",      type=str, required=True,      help="Skill/domain to distill")
    p.add_argument("--student",    type=str, default="medium",    choices=["small", "medium", "large"],
                   help="Student model size (default: medium)")
    p.add_argument("--strategy",   type=str, default="cot",       choices=["cot", "rationale", "selfplay"],
                   help="Distillation strategy (default: cot)")
    p.add_argument("--iterations", type=int, default=1,           help="Number of distillation rounds (default: 1)")
    p.add_argument("--output",     type=str, default="./runs",    help="Output directory (default: ./runs)")
    p.add_argument("--seed",       type=str, default="",          help="Optional seed prompt hint")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{RED}✗  ANTHROPIC_API_KEY not set.{RESET}")
        sys.exit(1)

    banner()

    # Run metadata
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"{run_id}_{args.topic.replace(' ', '_')[:30]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {DIM}Topic:      {RESET}{BOLD}{args.topic}{RESET}")
    print(f"  {DIM}Student:    {RESET}{args.student}")
    print(f"  {DIM}Strategy:   {RESET}{args.strategy}")
    print(f"  {DIM}Iterations: {RESET}{args.iterations}")
    print(f"  {DIM}Output:     {RESET}{output_dir}")

    # Save run config
    config = {
        "run_id": run_id,
        "topic": args.topic,
        "student_profile": args.student,
        "strategy": args.strategy,
        "iterations": args.iterations,
        "seed_prompt": args.seed,
        "output_dir": str(output_dir),
        "started_at": datetime.datetime.now().isoformat(),
    }
    save(output_dir / "run_config.json", config)

    # ── Main loop ────────────────────────────────────────────────────────────
    eval_gaps: list[str] = []
    all_eval_reports = []

    for i in range(1, args.iterations + 1):
        eval_report, eval_gaps = run_iteration(
            topic=args.topic,
            student_profile=args.student,
            strategy=args.strategy,
            iteration=i,
            eval_gaps=eval_gaps,
            seed_prompt=args.seed,
            output_dir=output_dir,
        )
        if eval_report:
            all_eval_reports.append(eval_report)

        # Early stopping: if convergence_estimate is 0, we're done
        if eval_report.get("convergence_estimate", 99) == 0:
            print(f"\n  {GREEN}✓  Convergence reached at iteration {i}.{RESET}")
            break

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n\n{GREEN}{BOLD}{'═' * 50}")
    print(f"  ✓  PIPELINE COMPLETE")
    print(f"{'═' * 50}{RESET}\n")

    if all_eval_reports:
        total_delta = sum(r.get("benchmark_delta", 0) for r in all_eval_reports)
        summary_line("Iterations run:", len(all_eval_reports))
        summary_line("Total est. improvement:", f"+{total_delta:.1f}%")
        summary_line("Final gaps remaining:", len(eval_gaps))

    summary_line("All outputs saved to:", str(output_dir))
    print(f"\n  {DIM}Training data (JSONL) in each iteration_XX/ folder{RESET}")
    print(f"  {DIM}Ready for fine-tuning with OpenAI, Unsloth, or Axolotl{RESET}\n")

    # Save final summary
    summary = {
        "run_id": run_id,
        "config": config,
        "iterations_completed": len(all_eval_reports),
        "eval_reports": all_eval_reports,
        "final_gaps": eval_gaps,
        "completed_at": datetime.datetime.now().isoformat(),
    }
    save(output_dir / "run_summary.json", summary)


if __name__ == "__main__":
    main()
