"""Replay QAMPARI utility labels, learning, calibration and transfer results."""

import argparse
import math
from pathlib import Path

import numpy as np

from .analysis import fit_models, pilot_stability, propose, summarize
from .model import BASELINES, LEARNED, choose_threshold, read, select, sha, utility, write


SAVED = Path(__file__).with_name("saved")


def compare(actual, expected, path="result"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Different fields: {path}")
        for key in expected:
            compare(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Different list length: {path}")
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            compare(a, b, path + f"[{i}]")
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Different numerical result: {path}")
    elif actual != expected:
        raise ValueError(f"Different result: {path}")


def scores(name):
    values = read(SAVED / name)
    for record in values.values():
        recomputed = utility(record["metrics"])
        compare(recomputed, record["metrics"]["utility"], "cached utility")
        record["metrics"]["utility"] = recomputed
    return values


def evaluate(data, output):
    if output.exists():
        raise ValueError("Choose a new output directory")
    manifest = read(SAVED / "manifest.json")
    for name, digest in manifest["files"].items():
        if sha(SAVED / name) != digest:
            raise ValueError(f"Saved experiment file changed: {name}")
    for name, metadata in manifest["prepared_files"].items():
        if sha(data / name) != metadata["sha256"]:
            raise ValueError(f"Prepared data changed: {name}")
    questions = read(data / "questions.json")
    lookup = {q["qid"]: q for q in questions}
    with np.load(data / "features.npz", allow_pickle=False) as arrays:
        x = arrays["x"]
    source = scores("source_scores.json")
    models, labels = fit_models(questions, x, source)
    compare(models, read(SAVED / "models.json"), "models")
    compare(labels, read(SAVED / "training_labels.json"), "training labels")
    proposals = propose(questions, x, models, "calibration")
    compare(proposals, read(SAVED / "calibration_proposals.json"), "calibration proposals")
    calibration_scores = scores("calibration_scores.json")
    thresholds, profiles = {}, {}
    for name in LEARNED:
        records = []
        for qid, values in proposals.items():
            proposal = values[name]
            if proposal["index"] is not None:
                delta = np.mean([calibration_scores[f"{qid}|{name}|{seed}"]["metrics"]["utility"] -
                                 source[f"{qid}|gtr_top5|{seed}"]["metrics"]["utility"] for seed in (17, 29)])
                records.append(dict(qid=qid, score=proposal["score"], delta=float(delta)))
        thresholds[name] = choose_threshold(records)
        profiles[name] = records
    baseline_utilities = {name: float(np.mean([source[f"{q['qid']}|{name}|{seed}"]["metrics"]["utility"]
        for q in questions if q["role"] == "calibration" for seed in (17, 29)])) for name in BASELINES}
    strongest = max(BASELINES, key=lambda name: (baseline_utilities[name], -BASELINES.index(name)))
    calibration = dict(thresholds=thresholds, profiles=profiles,
                       fixed_baseline_utilities=baseline_utilities, strongest_fixed_baseline=strongest)
    compare(calibration, read(SAVED / "calibration_policy.json"), "calibration policy")
    evaluation_proposals = propose(questions, x, models, "evaluation")
    predictions = {}
    for qid, values in evaluation_proposals.items():
        q = lookup[qid]
        methods = dict(q["baselines"])
        methods.update({name: select(q, p["index"], p["score"], thresholds[name]) for name, p in values.items()})
        predictions[qid] = dict(proposals=values, methods=methods)
    compare(predictions, read(SAVED / "evaluation_predictions.json"), "evaluation selections")
    pilot = pilot_stability({role: scores(f"stability_{role}_scores.json") for role in ("A", "B")},
                            [q["qid"] for q in questions if q["role"] == "train"][:32])
    compare(pilot, read(SAVED / "pilot_stability.json"), "stability pilot")
    frozen = read(SAVED / "selection_freeze.json")
    if frozen["strongest_fixed_baseline_from_A_calibration"] != strongest or frozen["B_outcomes_used_for_fitting"]:
        raise ValueError("Source reader calibration or transfer separation differs")
    result = summarize({role: scores(f"evaluation_{role}_scores.json") for role in ("B", "A")}, frozen)
    compare(result, read(SAVED / "result.json"))
    output.mkdir(parents=True)
    for name, value in (("models.json", models), ("training_labels.json", labels),
                        ("calibration_policy.json", calibration), ("evaluation_predictions.json", predictions),
                        ("pilot_stability.json", pilot), ("result.json", result)):
        write(output / name, value)
    verification = dict(passed=True, training_questions=labels["training_questions"],
        training_actions=labels["training_edits"], thresholds=thresholds,
        primary_baseline=strongest, evaluation_questions=len(frozen["evaluation_questions"]),
        cached_official_answer_and_citation_metrics_reused=True,
        models_labels_calibration_selections_stability_and_results_match=True, reader_calls=0, nli_calls=0)
    write(output / "verification.json", verification)
    print(verification)
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared/alce_transfer"))
    parser.add_argument("--output", type=Path, default=Path("outputs/alce_transfer_evaluation"))
    args = parser.parse_args()
    evaluate(args.data, args.output)
