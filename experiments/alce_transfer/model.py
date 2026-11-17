"""Reference-free one-exchange policies and source-only calibration."""

import hashlib
import json
from pathlib import Path

import numpy as np

LEARNED = ("mean_set", "conservative_set", "conservative_independent")
BASELINES = ("gtr_top5", "ce_top5", "mmr_0.5", "coverage")
THRESHOLDS = (0., .5, 1., 2., 5., 10., 20., None)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                       encoding="utf-8", newline="\n")
    pending.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def target(deltas, conservative):
    a = np.asarray(deltas, dtype=float)
    if a.shape[-1] != 2 or not np.isfinite(a).all():
        raise ValueError("Need exactly two finite source draws per edit")
    return a.min(axis=-1) if conservative else a.mean(axis=-1)


def transform(x, independent):
    result = np.asarray(x, dtype=float).copy()
    if independent:
        result[:, 11:] = 0.
    return result


def fit(x, deltas, groups, name):
    a = transform(x, name.endswith("independent"))
    center, scale = a.mean(axis=0), np.maximum(a.std(axis=0), 1e-4)
    a = np.column_stack(((a - center) / scale, np.ones(len(a))))
    y = target(deltas, name.startswith("conservative"))
    unique, counts = np.unique(groups, return_counts=True)
    sizes = dict(zip(unique, counts, strict=True))
    weights = np.asarray([1 / sizes[g] for g in groups])
    weights /= weights.sum()
    ridge = np.diag([1.] * x.shape[1] + [0.])
    coefficients = np.linalg.solve(a.T @ (a * weights[:, None]) + ridge, a.T @ (y * weights))
    return dict(name=name, center=center.tolist(), scale=scale.tolist(), coefficients=coefficients.tolist(),
                ridge_penalty=1., fit_rows=len(y), fit_questions=len(unique))


def scores(x, model):
    a = transform(x, model["name"].endswith("independent"))
    a = np.column_stack(((a - model["center"]) / model["scale"], np.ones(len(a))))
    return a @ model["coefficients"]


def proposal(question, x, model):
    if not question["actions"]:
        return None, None
    values = scores(x, model)
    best = min(range(len(values)), key=lambda i: (-values[i], question["actions"][i]["incoming"],
                                                 question["actions"][i]["slot"]))
    return best, float(values[best])


def choose_threshold(records):
    def objective(threshold):
        accepted = [r["delta"] for r in records if threshold is not None and
                    r["score"] is not None and r["score"] > threshold]
        return (sum(accepted), -len(accepted), float("inf") if threshold is None else threshold)
    return max(THRESHOLDS, key=objective)


def select(question, proposal_index, score, threshold):
    use = proposal_index is not None and threshold is not None and score > threshold
    return question["actions"][proposal_index]["selected"] if use else list(range(5))


def utility(metrics):
    f, c = metrics["qampari_f1_top5"], metrics["citation_rec"]
    return 2 * f * c / (f + c) if f + c else 0.
