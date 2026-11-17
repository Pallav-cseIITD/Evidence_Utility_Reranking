"""Recalculate HotpotQA answer and evidence results from the saved runs."""

import argparse
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from method.io import public_entries, read, sha, write
from method.support import covered, evidence_masks, facts, normalize
from .additional import METHODS, summarize
from .scoring import grade, parse_answer_output
from .statistics import pair, paired_binary


SAVED = Path(__file__).with_name("saved")


def compare(actual, expected, path="result"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Different result fields: {path}")
        for key in expected:
            compare(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Different result length: {path}")
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            compare(a, b, path + f"[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Different numeric result: {path}")
    elif actual != expected:
        raise ValueError(f"Different result: {path}")


def answer_summary(rows, qids):
    by_method = {name: [r for r in rows if r["method"] == name] for name in dict.fromkeys(r["method"] for r in rows)}
    sampled = np.random.default_rng(20261116).integers(len(qids), size=(10000, len(qids)))
    metrics, comparisons = {}, {}
    for name, values in by_method.items():
        if [r["qid"] for r in values] != qids:
            raise ValueError("Primary answer question order differs")
        metrics[name] = {key: sum(r[key] is True for r in values) for key in ("answer_em", "attributed_em", "valid", "capped")}
        metrics[name].update(questions=len(values), invalid=sum(not r["valid"] for r in values))
        if name != "fusion_scalar":
            comparisons[name] = {metric: pair([r[metric] is True for r in by_method["fusion_scalar"]],
                                             [r[metric] is True for r in values], sampled)
                                 for metric in ("answer_em", "attributed_em")}
    return dict(metrics=metrics, paired=comparisons)


def evidence_results(pools, rankings, references, choices, qids):
    rows, pool_complete = [], 0
    for pool, (qid, rank), record in zip(pools, rankings, choices, strict=True):
        if pool["qid"] != qid or record["qid"] != qid:
            raise ValueError("Evidence collection order differs")
        masks, full = evidence_masks(references[qid], pool["passages"], "rendered")
        gold_titles = {normalize(f["title"]) for f in facts(references[qid])}
        pool_complete += covered(masks, range(50)) == full
        embeddings = np.asarray(rank["embeddings"], dtype=float)
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-10)
        cosine = embeddings @ embeddings.T
        relevant = [int(mask != 0) for mask in masks]
        ideal = sum(1 / math.log2(i + 2) for i in range(min(5, sum(relevant))))
        baseline = record["methods"]["fusion_top5"]["indices"]
        for name in METHODS:
            indices = record["methods"][name]["indices"]
            mask = covered(masks, indices)
            dcg = sum(relevant[index] / math.log2(i + 2) for i, index in enumerate(indices))
            rows.append(dict(qid=qid, method=name, complete_chain=mask == full,
                supporting_sentence_recall=mask.bit_count() / full.bit_count(),
                gold_title_recall=len({normalize(pool["passages"][i]["title"]) for i in indices} & gold_titles) / 2,
                pool_support_ndcg_at5=dcg / ideal if ideal else 0.,
                support_passage_precision_at5=sum(relevant[i] for i in indices) / 5,
                mean_pairwise_cosine=float(np.mean([cosine[a, b] for a, b in combinations(indices, 2)])),
                original_token_counts=record["methods"][name]["tokens"], changed_set=set(indices) != set(baseline)))
    lookup = {(r["qid"], r["method"]): r for r in rows}
    aggregates = {}
    for name in METHODS:
        selected = [lookup[qid, name] for qid in qids]
        aggregates[name] = dict(questions=len(selected), complete_chains=sum(r["complete_chain"] for r in selected),
            changed_sets=sum(r["changed_set"] for r in selected),
            **{key: float(np.mean([r[key] for r in selected])) for key in (
                "supporting_sentence_recall", "gold_title_recall", "pool_support_ndcg_at5",
                "support_passage_precision_at5", "mean_pairwise_cosine")},
            mean_original_token_counts=np.mean([r["original_token_counts"] for r in selected], axis=0).tolist())
    pairs = {"fusion_scalar_minus_" + right: paired_binary(
        [lookup[qid, "fusion_scalar"]["complete_chain"] for qid in qids],
        [lookup[qid, right]["complete_chain"] for qid in qids])
        for right in ("fusion_top5", "independent", "submodular_coverage_top5", "forced_repair")}
    return dict(questions=len(qids), pool_complete_chains=pool_complete, methods=aggregates,
                support_chain_comparisons_exploratory=pairs)


def evaluate(data, output):
    if output.exists():
        raise ValueError("Choose a new output directory")
    manifest = read(SAVED / "manifest.json")
    if any(sha(SAVED / name) != digest for name, digest in manifest["files"].items()):
        raise ValueError("Saved experiment record changed")
    for name in ("pools.json", "features.json", "references.json"):
        if sha(data / name) != manifest["data_sha256"][name]:
            raise ValueError(f"Prepared evaluation input changed: {name}")
    references = {r["id"]: r for r in read(data / "references.json")}
    choices = read(SAVED / "selections.json")
    qids = [r["qid"] for r in choices]
    pools = {r["qid"]: r for r in public_entries(data / "pools.json")}
    sets = {(r["qid"], method): selected["indices"] for r in choices for method, selected in r["methods"].items()}
    rows = {reader: [] for reader in ("qwen8b", "qwen14b", "phi4")}
    for line in (SAVED / "reader_outputs.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        answers, error = parse_answer_output(record["output"])
        if (error is None) != (record["parse_error"] is None):
            raise ValueError("Saved parsing status differs from the original response")
        evidence = [pools[record["qid"]]["passages"][i] for i in sets[record["qid"], record["method"]]]
        score = grade(references[record["qid"]], dict(evidence=evidence),
                      dict(strict_valid=error is None, hit_token_cap=record["capped"], answers=answers))
        rows[record["reader"]].append(dict(qid=record["qid"], method=record["method"],
                                          capped=record["capped"], **score))
    additional = {}
    for role, reader in (("A", "qwen14b"), ("B", "phi4")):
        summary = summarize(rows[reader], qids)
        additional[role] = {key: summary[key] for key in ("methods", "comparisons")}
    result = dict(primary=answer_summary(rows["qwen8b"], qids), additional=additional,
                  evidence=evidence_results(iter(pools.values()), public_entries(data / "features.json", True),
                                            references, choices, qids))
    compare(result, read(SAVED / "expected_results.json"))
    output.mkdir(parents=True)
    write(output / "results.json", result)
    write(output / "answers_per_question.json", rows)
    write(output / "verification.json", dict(passed=True, logical_answer_records=sum(len(r) for r in rows.values()),
          questions=len(qids), matched_original_results=True, reader_calls=0,
          qualification=read(SAVED / "qualification.json")))
    print(json.dumps(dict(passed=True, questions=len(qids), scored_responses=sum(len(r) for r in rows.values()),
                         fusion_complete_chains=result["evidence"]["methods"]["fusion_top5"]["complete_chains"],
                         repair_complete_chains=result["evidence"]["methods"]["fusion_scalar"]["complete_chains"],
                         reader_calls=0), indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared/hotpot_eval"))
    parser.add_argument("--output", type=Path, default=Path("outputs/hotpot_evaluation"))
    args = parser.parse_args()
    evaluate(args.data, args.output)
