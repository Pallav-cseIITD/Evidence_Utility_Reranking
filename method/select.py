"""Select passages with the saved final repair model and baseline controls."""

import argparse
import json
from pathlib import Path

from .inputs import load_tokenizers
from .io import public_entries, read, write
from .policy import load_fusion, predict


DEFAULT_MODEL = Path(__file__).with_name("fitted")


def select_batch(model_dir, pools_path, features_path, tokenizer_paths):
    models, policy = load_fusion(model_dir)
    toks, _ = load_tokenizers(tokenizer_paths, policy["tokenizer_sha256"])
    seen = set()
    rows = []
    stream = zip(public_entries(pools_path), public_entries(features_path, True), strict=True)
    for pool, (qid, ranking) in stream:
        if pool["qid"] != qid or qid in seen:
            raise ValueError("Duplicate or mismatched question identifiers")
        seen.add(qid)
        rows.append(dict(qid=qid, methods=predict(pool, ranking, toks, models, policy["thresholds"])))
    if not rows:
        raise ValueError("The passage collection is empty")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", type=Path, help="One question JSON, or a passage pool array with --batch")
    parser.add_argument("features", type=Path, help="One ranking JSON, or a question keyed mapping with --batch")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizers", type=Path, nargs=2, required=True,
                        metavar=("QWEN25_JSON", "QWEN3_JSON"))
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Choose a new output file")
    if args.batch:
        result = select_batch(args.model, args.pool, args.features, args.tokenizers)
    else:
        models, policy = load_fusion(args.model)
        toks, _ = load_tokenizers(args.tokenizers, policy["tokenizer_sha256"])
        result = predict(read(args.pool), read(args.features), toks, models, policy["thresholds"])
    if args.output is None:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write(args.output, result)
        print(f"Wrote selections to {args.output}")


if __name__ == "__main__":
    main()
