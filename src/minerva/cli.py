"""The ``minerva`` command-line interface.

    minerva doctor                      check that a real engine is reachable
    minerva models                      list the model catalogue
    minerva tools                       list registered tools
    minerva thinking                    show the solfege thinking scale
    minerva prepare-data                build the corpus and tokenizer
    minerva train                       pretrain a model from scratch
    minerva finetune                    instruction-tune it onto the chat format
    minerva evaluate                    measure a base checkpoint
    minerva evaluate-instruct           measure an instruction-tuned model
    minerva ask "..." [-t mi]           one question, one answer
    minerva chat [-t mi]                interactive conversation
    minerva serve                       local web chat UI

Built on :mod:`argparse` so it has no dependencies beyond the stdlib.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .config import MinervaConfig, load_config, set_config
from .engines.registry import available_engines, get_engine
from .errors import MinervaError
from .messages import ToolCall
from .models.registry import get_spec, list_specs, load_model
from .runtime.agent import Agent
from .runtime.session import Session
from .thinking import ThinkingLevel, parse_thinking_level
from .tools.registry import default_registry

__all__ = ["build_parser", "main"]


# --------------------------------------------------------------------------
# Small terminal helpers. Colour is disabled when stdout is not a TTY, so
# piping the output into a file gives clean text.
# --------------------------------------------------------------------------

class _Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)


def _style(no_color: bool) -> _Style:
    return _Style(enabled=not no_color and sys.stdout.isatty())


def _config_from_args(args: argparse.Namespace) -> MinervaConfig:
    config = load_config() if args.config is None else load_config(args.config)
    changes: dict[str, Any] = {}
    if args.host:
        changes["ollama_host"] = args.host
    if getattr(args, "thinking", None):
        changes["thinking_level"] = parse_thinking_level(args.thinking)
    if getattr(args, "no_tools", False):
        changes["enable_tools"] = False
    if changes:
        config = config.replace(**changes)
    set_config(config)
    return config


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the real state of the world: engine up? models installed?"""
    config = _config_from_args(args)
    style = _style(args.no_color)

    print(style.bold("Minerva doctor"))
    print(f"  version        {__version__}")
    print(f"  config file    {config.source_file or '(defaults + environment)'}")
    print(f"  engine         {config.engine}  (available: {', '.join(available_engines())})")
    print(f"  thinking       {config.thinking_level.latin_name} "
          f"({config.thinking_level.hebrew_name})")
    print()

    engine = get_engine(config.engine, config=config)
    health = engine.health()
    if health.available:
        print(style.green(f"  OK  engine {engine.name}: {health}"))
    else:
        print(style.red(f"  !!  engine {engine.name}: {health}"))
        print()
        # Advice has to match the engine: one is trained, the other is served.
        if engine.name == "minerva":
            print("  To fix, build the data and train a model:")
            print("    minerva prepare-data")
            print("    minerva train --out checkpoints/swift")
        else:
            print("  To fix: install Ollama from https://ollama.com, then run `ollama serve`.")
        return 1

    print()
    print(style.bold("  Models"))
    exit_code = 0
    for spec in list_specs():
        model = load_model(spec.name, config=config)
        if model.is_installed():
            tag = model.resolve_engine_model()
            print(style.green(f"    OK  {spec.name:<10} -> {tag}"))
        else:
            print(
                style.yellow(
                    f"    --  {spec.name:<10} -> {spec.engine_model} not installed "
                    f"(run: minerva pull {spec.name})"
                )
            )
            exit_code = 1
    return exit_code


def cmd_models(args: argparse.Namespace) -> int:
    """List the catalogue."""
    _config_from_args(args)
    style = _style(args.no_color)

    specs = list_specs()
    if args.verbose:
        for spec in specs:
            print(style.bold(f"{spec.display_name}  ({spec.name})"))
            print(f"  version        {spec.version}")
            print(f"  tier           {spec.tier}  ({spec.parameter_count or 'size n/a'})")
            print(f"  engine         {spec.engine}:{spec.engine_model}")
            if spec.engine_model_fallbacks:
                print(f"  fallbacks      {', '.join(spec.engine_model_fallbacks)}")
            print(f"  context        {spec.context_length} tokens")
            print(
                f"  thinking       default {spec.default_thinking.latin_name} "
                f"({spec.default_thinking.hebrew_name}), "
                f"max {spec.max_thinking.latin_name} ({spec.max_thinking.hebrew_name})"
            )
            print(f"  tools          {'yes' if spec.supports_tools else 'no'}")
            print(f"  tags           {', '.join(spec.tags) or '-'}")
            print(f"  {spec.description}")
            print()
        return 0

    header = f"{'NAME':<10} {'TIER':<8} {'SIZE':<7} {'ENGINE MODEL':<18} DESCRIPTION"
    print(style.bold(header))
    for spec in specs:
        print(
            f"{spec.name:<10} {spec.tier:<8} {spec.parameter_count or '-':<7} "
            f"{spec.engine_model:<18} {spec.description.split('.')[0]}"
        )
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    """List the tools a model can call."""
    style = _style(args.no_color)
    registry = default_registry()

    if not len(registry):
        print("no tools registered")
        return 0

    for tool_obj in registry:
        print(style.bold(tool_obj.name))
        print(f"  {tool_obj.description.strip().splitlines()[0]}")
        properties = tool_obj.parameters.get("properties", {})
        required = set(tool_obj.parameters.get("required", []))
        for param, schema in properties.items():
            marker = "*" if param in required else " "
            kind = schema.get("type", "any")
            description = schema.get("description", "")
            print(f"    {marker} {param}: {kind}  {style.dim(description)}")
        print()
    print(style.dim("* = required"))
    return 0


def cmd_thinking(args: argparse.Namespace) -> int:
    """Show the solfege thinking scale."""
    style = _style(args.no_color)
    print(style.bold("Minerva thinking levels - the solfege scale"))
    print()
    print(
        style.bold(
            f"  {'#':<3} {'NOTE':<6} {'HEB':<5} {'BUDGET':>8}  {'EFFORT':<7} "
            f"{'EXTENDED':<9} DESCRIPTION"
        )
    )
    for level in ThinkingLevel:
        profile = level.profile
        budget = f"{profile.budget_tokens:,}" if profile.budget_tokens else "-"
        print(
            f"  {int(level):<3} {level.latin_name:<6} {level.hebrew_name:<5} {budget:>8}  "
            f"{(profile.effort or 'off'):<7} {('yes' if profile.extended else 'no'):<9} "
            f"{level.description}"
        )
    print()
    print(style.dim("  Use with: minerva ask -t sol \"...\"   or   MINERVA_THINKING_LEVEL=סול"))
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    """Install a model's weights on the engine, for real."""
    config = _config_from_args(args)
    style = _style(args.no_color)

    spec = get_spec(args.model or config.default_model)
    engine = get_engine(spec.engine, config=config)

    if not hasattr(engine, "pull"):
        print(
            style.red(
                f"engine {engine.name!r} has nothing to pull: its weights are trained, "
                f"not downloaded."
            )
        )
        print("  Build the data first:  minerva prepare-data")
        print(f"  Then train:            minerva train --out checkpoints/{spec.name}")
        return 1

    tag = spec.engine_model
    print(f"pulling {tag} for Minerva model {spec.name!r} ...")
    last = ""
    for event in engine.pull(tag):
        status = event.get("status", "")
        total, completed = event.get("total"), event.get("completed")
        if isinstance(total, (int, float)) and isinstance(completed, (int, float)) and total:
            line = (
                f"  {status}: {completed / total * 100:5.1f}% "
                f"({completed / 1e6:.0f}/{total / 1e6:.0f} MB)"
            )
            print(f"\r{line}", end="", flush=True)
            last = line
        elif status and status != last:
            print(f"\r  {status}{' ' * 40}")
            last = status
    print()
    print(style.green(f"done: {tag} is installed"))
    return 0


#: How much text the BPE tokenizer is trained on. 200 MB is far past the point
#: where merge frequencies stop moving, and keeps peak memory bounded no matter
#: how large the corpus grows.
_TOKENIZER_SAMPLE_BYTES = 200_000_000


def cmd_prepare_data(args: argparse.Namespace) -> int:
    """Build the corpus, train the tokenizer, and pack tokens - all for real."""
    from .training.data import build_corpus
    from .training.dataset import encode_corpus
    from .training.tokenizer import BPETokenizer

    style = _style(args.no_color)
    data_dir = args.data

    print(style.bold("1/3  building the corpus from real sources"))
    manifest = build_corpus(data_dir)
    chars = manifest["characters"]
    assert isinstance(chars, dict)
    print(f"     train {chars['train'] / 1e6:.2f} MB   val {chars['val'] / 1e6:.2f} MB")

    tokenizer_path = data_dir / "tokenizer.json"
    if tokenizer_path.exists() and not args.force:
        print(style.dim(f"\n2/3  tokenizer exists at {tokenizer_path} (use --force to retrain)"))
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print(style.bold(f"\n2/3  training a byte-level BPE tokenizer (vocab {args.vocab_size})"))
        # Trained on a bounded sample, not the whole corpus. BPE merge
        # frequencies converge long before a gigabyte, and reading the v0.5.0
        # corpus whole needs it twice over (str + encoded bytes) - which is
        # what killed the first build with MemoryError. The sample is taken
        # from evenly spaced points so it reflects the shuffled source mixture
        # rather than whichever source happens to sort first.
        train_path = data_dir / "train.txt"
        total = train_path.stat().st_size
        budget = min(total, _TOKENIZER_SAMPLE_BYTES)
        if budget < total:
            print(
                style.dim(
                    f"     sampling {budget / 1e6:.0f} MB of {total / 1e6:.0f} MB "
                    f"(merge frequencies converge well before the full corpus)"
                )
            )
            parts: list[str] = []
            windows = 40
            with train_path.open("r", encoding="utf-8", errors="replace") as handle:
                for index in range(windows):
                    handle.seek(total * index // windows)
                    handle.readline()  # discard the partial line the seek landed in
                    parts.append(handle.read(budget // windows))
            text = "".join(parts)
        else:
            text = train_path.read_text(encoding="utf-8")
        tokenizer = BPETokenizer.train(text, args.vocab_size)
        tokenizer.save(tokenizer_path)

    print(style.bold("\n3/3  tokenizing"))
    for split in ("train", "val"):
        encode_corpus(data_dir / f"{split}.txt", tokenizer, data_dir / f"{split}.bin")

    print(style.green(f"\nready. Train with: minerva train --data {data_dir}"))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Pretrain a Minerva model from scratch."""
    from .training.trainer import main as train_main

    forwarded: list[str] = ["--data", str(args.data), "--out", str(args.out)]
    for flag, value in (
        ("--steps", args.steps),
        ("--lr", args.lr),
        ("--batch-size", args.batch_size),
        ("--grad-accum", args.grad_accum),
        ("--seq-len", args.seq_len),
        ("--max-hours", args.max_hours),
        ("--threads", args.threads),
        ("--resume", args.resume),
        ("--checkpoint-interval", args.checkpoint_interval),
        ("--seed", args.seed),
        ("--stop-file", args.stop_file),
    ):
        if value is not None:
            forwarded += [flag, str(value)]
    return train_main(forwarded)


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Measure a trained checkpoint on held-out text."""
    from .training.evaluate import main as evaluate_main

    forwarded = ["--checkpoint", str(args.checkpoint), "--data", str(args.data)]
    if args.json:
        forwarded += ["--json", str(args.json)]
    if args.max_batches is not None:
        forwarded += ["--max-batches", str(args.max_batches)]
    return evaluate_main(forwarded)


def cmd_finetune(args: argparse.Namespace) -> int:
    """Instruction-tune a base checkpoint into a conversational model."""
    from .training.finetune import main as finetune_main

    forwarded = ["--base", str(args.base), "--out", str(args.out)]
    if args.tokenizer:
        forwarded += ["--tokenizer", str(args.tokenizer)]
    if args.epochs is not None:
        forwarded += ["--epochs", str(args.epochs)]
    if args.threads is not None:
        forwarded += ["--threads", str(args.threads)]
    return finetune_main(forwarded)


def cmd_evaluate_instruct(args: argparse.Namespace) -> int:
    """Measure an instruction-tuned model on held-out conversations."""
    from .training.instruct_eval import main as eval_main

    forwarded = ["--model", args.model]
    if args.json:
        forwarded += ["--json", str(args.json)]
    if args.thinking:
        forwarded += ["--thinking", args.thinking]
    if args.quiet:
        forwarded.append("--quiet")
    return eval_main(forwarded)


def cmd_ask(args: argparse.Namespace) -> int:
    """Ask one question."""
    config = _config_from_args(args)
    style = _style(args.no_color)

    model = load_model(args.model, config=config)
    agent = Agent(
        model,
        thinking=config.thinking_level,
        max_iterations=config.max_tool_iterations,
        on_tool_call=_tool_call_printer(style) if args.show_tools else None,
    )

    show_thinking = args.show_thinking or config.show_thinking

    if args.no_stream or not config.stream:
        run = agent.run(args.prompt)
        if show_thinking and run.thinking:
            print(style.dim(run.thinking))
            print()
        print(run.answer)
        return 0

    _stream_to_stdout(agent, args.prompt, style, show_thinking=show_thinking)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive multi-turn conversation."""
    config = _config_from_args(args)
    style = _style(args.no_color)

    model = load_model(args.model, config=config)
    session = Session(
        model,
        thinking=config.thinking_level,
        max_iterations=config.max_tool_iterations,
    )
    show_thinking = args.show_thinking or config.show_thinking

    print(style.bold(f"Minerva {model.spec.display_name}"))
    print(
        style.dim(
            f"thinking: {session.level.latin_name} ({session.level.hebrew_name})   "
            f"tools: {', '.join(session.tools.names()) if session.tools else 'none'}"
        )
    )
    print(style.dim("commands: /thinking <note>, /reset, /thinking?, /exit"))
    print()

    while True:
        try:
            line = input(style.cyan("you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/reset":
            session.reset()
            print(style.dim("conversation cleared"))
            continue
        if line == "/thinking?":
            print(style.dim(f"{session.level.latin_name} ({session.level.hebrew_name}) - "
                            f"{session.level.description}"))
            continue
        if line.startswith("/thinking "):
            try:
                session.thinking = parse_thinking_level(line.split(maxsplit=1)[1])
            except MinervaError as exc:
                print(style.red(str(exc)))
                continue
            print(style.dim(f"thinking level -> {session.level.latin_name} "
                            f"({session.level.hebrew_name})"))
            continue

        try:
            run = session.send_run(line)
        except MinervaError as exc:
            print(style.red(f"error: {exc}"))
            continue

        if show_thinking and run.thinking:
            print(style.dim(run.thinking))
        for call in run.tool_calls:
            print(style.yellow(f"[tool] {call}"))
        print(f"{style.bold(model.spec.name)}> {run.answer}")
        print()


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the local web chat UI."""
    from .webui import serve

    config = _config_from_args(args)
    serve(model_name=args.model, host=args.bind, port=args.port, config=config)
    return 0


# --------------------------------------------------------------------------
# Streaming output
# --------------------------------------------------------------------------


def _tool_call_printer(style: _Style) -> Callable[[ToolCall], None]:
    def printer(call: ToolCall) -> None:
        print(style.yellow(f"[tool] {call}"), file=sys.stderr)

    return printer


def _stream_to_stdout(
    agent: Agent, prompt: str, style: _Style, *, show_thinking: bool
) -> None:
    """Render a streamed run, keeping reasoning visually distinct from the answer."""
    in_thinking = False
    for chunk in agent.stream(prompt):
        if chunk.kind == "thinking":
            if show_thinking:
                if not in_thinking:
                    print(style.dim("[thinking] "), end="")
                    in_thinking = True
                print(style.dim(chunk.text), end="", flush=True)
        elif chunk.kind == "content":
            if in_thinking:
                print("\n")
                in_thinking = False
            print(chunk.text, end="", flush=True)
        elif chunk.kind == "tool_call":
            if in_thinking:
                print("\n")
                in_thinking = False
            print(style.yellow(f"\n[tool] {chunk.tool_call}"), flush=True)
        elif chunk.kind == "tool_result":
            print(style.dim(f"[result] {chunk.text[:200]}"), flush=True)
    print()


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minerva",
        description="Minerva - an AI model platform. First model: Swift.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "thinking levels (solfege scale, quietest to loudest):\n"
            "  do (דו)  re (רה)  mi (מי)  fa (פה)  sol (סול)  la (לה)  si (סי)\n"
            "\nexamples:\n"
            "  minerva doctor\n"
            "  minerva ask -t sol \"What is 17 * 43?\"\n"
            "  minerva chat --show-thinking\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"minerva {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", metavar="PATH", help="path to a TOML config file")
    common.add_argument("--host", metavar="URL", help="engine host, e.g. http://127.0.0.1:11434")
    common.add_argument("--no-color", action="store_true", help="disable coloured output")

    generation = argparse.ArgumentParser(add_help=False)
    generation.add_argument("-m", "--model", help="Minerva model name (default: swift)")
    generation.add_argument(
        "-t",
        "--thinking",
        metavar="NOTE",
        help="thinking level: do/re/mi/fa/sol/la/si, a Hebrew note name, or 0-6",
    )
    generation.add_argument(
        "--show-thinking", action="store_true", help="print the model's reasoning trace"
    )
    generation.add_argument("--no-tools", action="store_true", help="do not offer tools")
    generation.add_argument("--no-stream", action="store_true", help="wait for the full answer")

    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", parents=[common], help="check engine and model health")
    doctor.set_defaults(func=cmd_doctor)

    models = sub.add_parser("models", parents=[common], help="list the model catalogue")
    models.add_argument("-v", "--verbose", action="store_true", help="show full specs")
    models.set_defaults(func=cmd_models)

    tools = sub.add_parser("tools", parents=[common], help="list callable tools")
    tools.set_defaults(func=cmd_tools)

    thinking = sub.add_parser("thinking", parents=[common], help="show the thinking scale")
    thinking.set_defaults(func=cmd_thinking)

    pull = sub.add_parser("pull", parents=[common], help="install a model's weights")
    pull.add_argument("model", nargs="?", help="Minerva model name (default: swift)")
    pull.set_defaults(func=cmd_pull)

    prepare = sub.add_parser(
        "prepare-data", parents=[common], help="build the corpus and tokenizer"
    )
    prepare.add_argument("--data", type=Path, default=Path("data"))
    prepare.add_argument("--vocab-size", type=int, default=8192)
    prepare.add_argument("--force", action="store_true", help="retrain the tokenizer")
    prepare.set_defaults(func=cmd_prepare_data)

    train = sub.add_parser("train", parents=[common], help="pretrain a model from scratch")
    train.add_argument("--data", type=Path, default=Path("data"))
    train.add_argument("--out", type=Path, default=Path("checkpoints/swift"))
    train.add_argument("--steps", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--grad-accum", type=int, default=None)
    train.add_argument("--seq-len", type=int, default=None)
    train.add_argument("--max-hours", type=float, default=None)
    train.add_argument("--threads", type=int, default=None)
    train.add_argument("--resume", type=Path, default=None)
    train.add_argument("--checkpoint-interval", type=int, default=None)
    # Every flag the trainer understands has to be repeated here and forwarded
    # in cmd_train, or `minerva train` silently offers less than the module it
    # wraps. --seed was missing, which is how CI failed with "unrecognized
    # arguments: --seed 2027" the moment the two runs were given different
    # seeds so their checkpoints would be worth merging.
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--stop-file", default=None)
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser(
        "evaluate", parents=[common], help="measure a trained checkpoint"
    )
    evaluate.add_argument("--checkpoint", type=Path, default=Path("checkpoints/swift/best.pt"))
    evaluate.add_argument("--data", type=Path, default=Path("data"))
    evaluate.add_argument("--max-batches", type=int, default=None)
    evaluate.add_argument("--json", type=Path, default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    finetune = sub.add_parser(
        "finetune", parents=[common], help="instruction-tune a trained model"
    )
    finetune.add_argument("--base", type=Path, default=Path("checkpoints/swift/best.pt"))
    finetune.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    finetune.add_argument("--out", type=Path, default=Path("checkpoints/swift-instruct"))
    finetune.add_argument("--epochs", type=int, default=None)
    finetune.add_argument("--threads", type=int, default=None)
    finetune.set_defaults(func=cmd_finetune)

    eval_instruct = sub.add_parser(
        "evaluate-instruct",
        parents=[common],
        help="measure an instruction-tuned model on held-out conversations",
    )
    eval_instruct.add_argument("--model", default="swift-instruct")
    eval_instruct.add_argument("--thinking", default=None)
    eval_instruct.add_argument("--json", type=Path, default=None)
    eval_instruct.add_argument("--quiet", action="store_true")
    eval_instruct.set_defaults(func=cmd_evaluate_instruct)

    ask = sub.add_parser("ask", parents=[common, generation], help="continue a prompt")
    ask.add_argument("prompt", help="the question")
    ask.add_argument(
        "--show-tools", action="store_true", help="print tool calls as they happen"
    )
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", parents=[common, generation], help="interactive conversation")
    chat.set_defaults(func=cmd_chat)

    serve = sub.add_parser("serve", parents=[common], help="run a local web chat UI")
    serve.add_argument("-m", "--model", help="Minerva model name (default: swift-instruct)")
    serve.add_argument("--bind", default="127.0.0.1", help="address the web UI listens on")
    serve.add_argument("--port", type=int, default=8420)
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``minerva`` command."""
    # Hebrew is a first-class training language (see training/data.py), so
    # anything Minerva prints - a chat reply, a training sample, a doctor
    # report - can contain it. A Windows console or a redirected-to-file
    # stream defaults to the system code page (e.g. cp1252), which cannot
    # encode Hebrew and raises UnicodeEncodeError, killing the process
    # mid-command. Force UTF-8 here, once, for every command.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except MinervaError as exc:
        print(f"minerva: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
