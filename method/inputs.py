"""Verify prepared training inputs and the native tokenizer files."""

from pathlib import Path

from .io import read, sha


DEFAULT_MANIFEST = Path(__file__).with_name("training_inputs.json")
REQUIRED_INPUTS = {
    "pools.json", "features.json", "train_references.json",
    "validation_references.json", "cohort_freeze.json",
}


def verify_inputs(raw, manifest):
    expected = manifest["input_sha256"]
    if set(expected) != REQUIRED_INPUTS:
        raise ValueError("Training manifest must describe the five prepared input files")
    for filename, digest in expected.items():
        if sha(raw / filename) != digest:
            raise ValueError(f"Training input hash mismatch: {filename}")
    roles = read(raw / "cohort_freeze.json")["question_ids"]
    for role in ("train", "validation"):
        if len(roles[role]) != manifest["question_counts"][role]:
            raise ValueError(f"Unexpected number of {role} questions")
        if len(roles[role]) != len(set(roles[role])):
            raise ValueError(f"Duplicate {role} question identifiers")
    if set(roles["train"]) & set(roles["validation"]):
        raise ValueError("Training and validation questions overlap")
    if (set(roles["train"]) | set(roles["validation"])) & set(roles.get("test", [])):
        raise ValueError("Training or validation questions overlap the test split")
    return dict(expected)


def load_tokenizers(paths, expected):
    from tokenizers import Tokenizer

    if len(paths) != 2:
        raise ValueError("Supply the Qwen2.5 and Qwen3 tokenizer JSON files, in that order")
    hashes = [sha(path) for path in paths]
    if hashes != expected:
        raise ValueError("Native tokenizer hashes or order differ from the recorded experiment")
    return [Tokenizer.from_file(str(path)) for path in paths], hashes
