"""Compare selected passages and token counts with a saved selection file."""

import argparse
import json
from pathlib import Path

from .io import read, sha, write
from .select import DEFAULT_MODEL, select_batch


def check(model, pools, features, expected, tokenizers):
    saved = read(expected)
    actual = select_batch(model, pools, features, tokenizers)
    if not isinstance(saved, list) or any(not isinstance(row, dict) or "qid" not in row for row in saved):
        raise ValueError("Expected a saved list of question selection records")
    if [row["qid"] for row in saved] != [row["qid"] for row in actual]:
        raise ValueError("Saved selections use a different question order or collection")
    mismatches = [row["qid"] for row, reference in zip(actual, saved, strict=True) if row != reference]
    report = dict(
        passed=not mismatches, questions=len(actual), mismatched_questions=mismatches,
        compared="Selected passage indices, method names and both native token counts",
        input_sha256={"pools": sha(pools), "features": sha(features), "expected": sha(expected)},
        reader_calls=0,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pools", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizers", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Choose a new output file")
    report = check(args.model, args.pools, args.features, args.expected, args.tokenizers)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write(args.output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
