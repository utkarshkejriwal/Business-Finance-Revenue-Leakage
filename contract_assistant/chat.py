"""Command-line chat over the merchant agreements and Schedule A.

    python chat.py                 interactive chat
    python chat.py "question"      ask one question
    python chat.py --sources ...   also print the retrieved chunks
    python chat.py --provider gemini|openrouter|anthropic ...
"""

import sys
import uuid

from graph import ask, build_graph, get_client


def print_turn(result, show_sources):
    print(f"\nAssistant: {result['answer']}\n")
    if show_sources and result.get("context"):
        print("Sources:")
        for chunk in result["context"]:
            score = "ref" if chunk["rerank_score"] is None else f"{chunk['rerank_score']:.2f}"
            print(f"  [{score:>6}] {chunk['section_id']:<18} {chunk['parent_key']:<10} {chunk['section_title']}")
        print()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    show_sources = "--sources" in args
    args = [a for a in args if a != "--sources"]
    provider = None
    if "--provider" in args:
        position = args.index("--provider")
        provider = args[position + 1]
        del args[position:position + 2]

    app = build_graph(provider)
    print(f"[provider: {get_client().name}]")
    thread_id = str(uuid.uuid4())

    if args:
        print_turn(ask(app, " ".join(args), thread_id), show_sources)
        return

    print("Shadowfax contract assistant - ask about M01-M12 agreements or Schedule A. Type 'exit' to quit.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print_turn(ask(app, question, thread_id), show_sources)


if __name__ == "__main__":
    main()
