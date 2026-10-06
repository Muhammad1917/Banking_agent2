"""Interactive command-line interface for the Banking Agent.

Usage:
    python main.py                 # chat (multi-turn, context kept)
    python main.py --demo          # run a scripted 3-scenario demo
    python main.py -q "سوال شما"   # single-shot question
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _print_result(res: dict, show_reasoning: bool = True) -> None:
    if show_reasoning and res.get("reasoning"):
        print("\n🧾 لاگ استدلال (multi-step reasoning):")
        for i, step in enumerate(res["reasoning"], 1):
            print(f"   {i}. {step}")
    print("\n🤖 دستیار:")
    print(res["answer"] or "(پاسخی تولید نشد)")


def chat(thread_id: str, show_reasoning: bool) -> None:
    from agent import run_turn

    print("=" * 60)
    print("دستیار هوشمند بانک دایا‌تدبیر — برای خروج exit را تایپ کنید")
    print("=" * 60)
    while True:
        try:
            text = input("\n🧑 کاربر: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit", "خروج"}:
            break
        try:
            res = run_turn(text, thread_id=thread_id)
            _print_result(res, show_reasoning)
        except Exception as exc:
            print(f"❌ خطا: {exc}", file=sys.stderr)


DEMO_SCRIPT = [
    "سلام، کد ملی من 0221536568 هست. کارتم چی شد؟",
    "چرا کارتم هنوز نیومده؟ چند روز شده؟",
    "میخوام یه کارت اعتباری جدید هم بگیرم",
    "آیا می‌تونم وام قرض‌الحسنه بگیرم؟",
    "زمان صدور کارت هدیه چقدره و سقف برداشت روزانه کارت نقدی چنده؟",
]


def demo(thread_id: str) -> None:
    from agent import run_turn

    for line in DEMO_SCRIPT:
        print("\n" + "=" * 60)
        print(f"🧑 کاربر: {line}")
        res = run_turn(line, thread_id=thread_id)
        _print_result(res)


def main() -> None:
    ap = argparse.ArgumentParser(description="Persian Banking Agent (LangGraph + OpenAI)")
    ap.add_argument("-q", "--question", help="single-shot Persian question")
    ap.add_argument("--demo", action="store_true", help="run scripted multi-scenario demo")
    ap.add_argument("--no-reasoning", action="store_true", help="hide reasoning trace")
    ap.add_argument("--thread", default=None, help="conversation thread id")
    args = ap.parse_args()

    thread_id = args.thread or uuid.uuid4().hex[:8]
    show = not args.no_reasoning

    if args.question:
        from agent import run_turn

        res = run_turn(args.question, thread_id=thread_id)
        _print_result(res, show)
    elif args.demo:
        demo(thread_id)
    else:
        chat(thread_id, show)


if __name__ == "__main__":
    main()
