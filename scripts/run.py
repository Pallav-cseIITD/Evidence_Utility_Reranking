"""Convenience commands for the frozen selector experiments."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def absolute(path):
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def command_plan(args):
    python = [sys.executable, "-B"]
    if args.command == "check":
        return [python + ["-m", "unittest", "discover", "-s", directory, "-v"]
                for directory in ("method/tests", "data/tests", "scripts/tests")]
    data = absolute(args.data_dir)
    if args.command in ("restore", "verify-data"):
        action = "restore" if args.command == "restore" else "verify"
        command = python + [str(ROOT / "data/prepare.py"), action, "--data-root", str(data),
                            "--groups", *args.groups]
        if args.command == "restore":
            if args.archives_dir is not None:
                command += ["--archives-dir", str(absolute(args.archives_dir))]
            else:
                command += ["--base-url", args.base_url]
        return [command]
    tokenizers = [str(data / "tokenizers/qwen25.json"), str(data / "tokenizers/qwen3.json")]
    if args.command == "train":
        return [python + ["-m", "method.train", str(data / "prepared/hotpot_train"),
                          str(absolute(args.output)), "--manifest", str(absolute(args.config)),
                          "--tokenizers", *tokenizers]]
    evaluation = data / "prepared/hotpot_eval"
    verify = python + [str(ROOT / "data/prepare.py"), "verify", "--data-root", str(data),
                       "--groups", "hotpot_eval", "tokenizers"]
    if args.command == "select":
        command = python + ["-m", "method.select", str(evaluation / "pools.json"),
                            str(evaluation / "features.json"), "--batch"]
    elif args.command == "reproduce":
        command = python + ["-m", "method.check_reproduction", str(evaluation / "pools.json"),
                            str(evaluation / "features.json"), str(evaluation / "selections.json")]
    else:
        raise ValueError(f"Unknown command: {args.command}")
    command += ["--model", str(absolute(args.model)), "--tokenizers", *tokenizers,
                "--output", str(absolute(args.output))]
    return [verify, command]


def execute(commands):
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Run the method, data and command wrapper tests")
    for name in ("restore", "verify-data", "train", "select", "reproduce"):
        sub = commands.add_parser(name)
        sub.add_argument("--data-dir", type=Path, default=Path("data"))
        if name in ("restore", "verify-data"):
            sub.add_argument("--groups", nargs="+", default=["hotpot_train", "hotpot_eval", "tokenizers"],
                             choices=("hotpot_train", "hotpot_eval", "tokenizers", "alce_transfer"))
        if name == "restore":
            source = sub.add_mutually_exclusive_group(required=True)
            source.add_argument("--archives-dir", type=Path)
            source.add_argument("--base-url")
        if name == "train":
            sub.add_argument("--config", type=Path, default=Path("configs/hotpot_repair.json"))
            sub.add_argument("--output", type=Path, default=Path("outputs/repair_model"))
        if name in ("select", "reproduce"):
            sub.add_argument("--model", type=Path, default=Path("method/fitted"))
            default = "outputs/hotpot_selections.json" if name == "select" else "outputs/selection_reproduction.json"
            sub.add_argument("--output", type=Path, default=Path(default))
    return result


def main():
    arguments = parser().parse_args()
    if arguments.command in ("train", "select", "reproduce") and absolute(arguments.output).exists():
        raise SystemExit("Choose a new output path with --output; the existing result will not be overwritten.")
    try:
        execute(command_plan(arguments))
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None


if __name__ == "__main__":
    main()
