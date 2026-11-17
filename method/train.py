"""Train the final repair selector and its incoming passage control on CPU."""

import argparse
from datetime import timedelta
from pathlib import Path
import platform
import time

import numpy as np

from .inputs import DEFAULT_MANIFEST, load_tokenizers, verify_inputs
from .io import public_entries, read, sha, write
from .model import Student, distribution
from .policy import THRESHOLDS, choose_threshold, design, labels
from .sampling import training_indices


def train(raw, output, paths, manifest_path=DEFAULT_MANIFEST):
    if output.exists():
        raise ValueError("Training requires a new output directory")
    started = time.monotonic()
    manifest = read(manifest_path)
    inputs = verify_inputs(raw, manifest)
    roles = read(raw / "cohort_freeze.json")["question_ids"]
    refs = {
        role: {row["id"]: row for row in read(raw / f"{role}_references.json")}
        for role in ("train", "validation")
    }
    if any(set(refs[role]) != set(roles[role]) for role in refs):
        raise ValueError("Reference identifiers differ from the recorded split")
    if manifest["threshold_grid"] != list(THRESHOLDS):
        raise ValueError("This implementation uses the recorded threshold grid")
    toks, tok_hashes = load_tokenizers(paths, manifest["tokenizer_sha256"])
    info = dict(questions={}, edits=[], targets=[], protocol={
        "train_sampling_per_question": manifest["train_sampling_per_question"],
    })
    matrices, validation = [], []
    stream = zip(public_entries(raw / "pools.json"), public_entries(raw / "features.json", True), strict=True)
    question_ids = roles["train"] + roles["validation"]
    for number, qid in enumerate(question_ids, 1):
        pool, (rank_id, rank) = next(stream)
        if pool["qid"] != qid or rank_id != qid:
            raise ValueError("Input ordering differs from the recorded split")
        role = "train" if qid in refs["train"] else "validation"
        base, _, _, edits, x = design(pool, rank, toks, "fusion")
        targets, base_complete, complete = labels(refs[role][qid], pool, base, edits)
        if role == "train":
            start = len(info["edits"])
            info["edits"].extend(dict(slot=slot, incoming=incoming) for slot, incoming in edits)
            info["targets"].extend(targets)
            info["questions"][qid] = dict(start=start, end=len(info["edits"]))
            matrices.append(x)
        else:
            validation.append(dict(qid=qid, x=x, targets=targets, edits=edits,
                                   base_complete=base_complete, complete=complete))
        if number % 40 == 0:
            elapsed = timedelta(seconds=int(time.monotonic() - started))
            print(f"Prepared {number}/{len(question_ids)} questions in {elapsed}", flush=True)
    del stream
    ids = training_indices(info, roles["train"])
    tx = np.concatenate(matrices)[ids]
    ty = np.asarray([distribution([info["targets"][i]]) for i in ids])
    vx = np.concatenate([row["x"] for row in validation])
    vy = np.asarray([distribution([target]) for row in validation for target in row["targets"]])
    output.mkdir(parents=True, exist_ok=False)
    policy = dict(
        inputs=inputs, tokenizer_sha256=tok_hashes, thresholds={}, validation={},
        train_questions=len(roles["train"]), validation_questions=len(validation),
        training_actions_labelled=len(info["targets"]), training_rows_used=len(ids),
        validation_actions=len(vy), reader_calls=0, new_indexing=0, paid_api_calls=0,
    )
    settings = manifest["model"]
    for name, kind in (("fusion_scalar", "scalar"), ("independent", "independent")):
        a, b = tx.copy(), vx.copy()
        if kind == "independent":
            a[:, 11:] = 0
            b[:, 11:] = 0
        model = Student(kind, seed=settings["seed"], hidden=settings["hidden"]).fit(
            a, ty, (b, vy), epochs=settings["epochs"],
        )
        model.save(output / f"{name}.npz")
        picked, curve = choose_threshold(validation, model)
        policy["thresholds"][name] = picked["threshold"]
        policy["validation"][name] = dict(selected=picked, sweep=curve)
        print(name, picked, flush=True)
    policy["wall_seconds"] = round(time.monotonic() - started, 3)
    policy["files"] = {path.name: sha(path) for path in sorted(output.glob("*.npz*"))}
    policy["source"] = sha(Path(__file__))
    policy["code_sha256"] = {path.name: sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    policy["training_manifest_sha256"] = sha(manifest_path)
    policy["environment"] = dict(python=platform.python_version(), numpy=np.__version__)
    write(output / "policy.json", policy)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="Prepared training collection directory")
    parser.add_argument("output", type=Path, help="A new directory for both fitted models")
    parser.add_argument("--tokenizers", type=Path, nargs=2, required=True,
                        metavar=("QWEN25_JSON", "QWEN3_JSON"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    train(args.raw, args.output, args.tokenizers, args.manifest)


if __name__ == "__main__":
    main()
