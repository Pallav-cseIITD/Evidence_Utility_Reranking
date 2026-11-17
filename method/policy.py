"""Fusion initialized passage replacement and validation threshold selection."""

import numpy as np

from .edits import TokenBudget, apply_edit, deterministic_order, shortlist
from .features import FeatureTable, bridge_rule
from .io import read, sha
from .model import Student
from .support import support_target, support_groups, transition

BASELINE = 'fusion_top5'
COMPARATORS = (BASELINE, 'relevance_ce_top5', 'dense_mmr_0.2', 'dense_mmr_0.5', 'dense_mmr_0.8', 'submodular_coverage_top5')


METHODS = (BASELINE, "relevance_ce_top5", "dense_mmr_0.8", "submodular_coverage_top5",
           "bridge_rule", "independent", "fusion_scalar")


THRESHOLDS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, None)


def fusion_candidates(qid, table, base, budget):
    unselected = [i for i in range(table.n) if i not in base]
    links = table.link + table.link.T
    result = []
    for slot, outgoing in enumerate(base):
        retained = [i for i in base if i != outgoing]
        options = [(slot, incoming) for incoming in unselected]

        def ce(edit):
            return -table.ce[edit[1]], edit[1]

        lists = [
            sorted(options, key=ce),
            sorted(options, key=lambda edit: (-sum(links[edit[1], retained]), *ce(edit))),
            sorted(options, key=lambda edit: (-sum(links[edit[1], unselected]), *ce(edit))),
            sorted(options, key=lambda edit: deterministic_order(qid, edit)),
        ]
        result.extend(shortlist(lists, 4, lambda edit: budget.fits(apply_edit(base, (edit,)))))
    return sorted(result)


def design(pool, rank, toks, method):
    base = tuple(rank["selections"][BASELINE if method == "fusion" else "dense_mmr_0.8"])
    if len(base) != 5 or len(set(base)) != 5 or any(not 0 <= i < 50 for i in base):
        raise ValueError("Expected five unique baseline passages")
    budget = TokenBudget(pool["passages"], toks, base)
    if max(budget.limit) > 1536:
        raise ValueError("Baseline exceeds the shared evidence cap")
    table = FeatureTable(pool, rank)
    if method == "fusion":
        edits = fusion_candidates(pool["qid"], table, base, budget)
    else:
        edits = [(slot, incoming) for incoming in range(table.n) if incoming not in base
                 for slot in range(len(base)) if budget.fits(apply_edit(base, ((slot, incoming),)))]
    x = np.asarray([table.features(base, *edit) for edit in edits], dtype=np.float32)
    return base, budget, table, edits, x.reshape(-1, len(table.feature_names))


def scores(model, x):
    if not len(x):
        return np.empty(0)
    if model.kind == "independent":
        x = x.copy()
        x[:, 11:] = 0
    return model.predict(x)["mean_gain"]


def best_edit(values, edits, minimum):
    if minimum is None or not len(edits):
        return None
    best = min(range(len(edits)), key=lambda i: (-values[i], edits[i][1], edits[i][0]))
    return best if values[best] > minimum else None


def selection_report(indices, budget):
    return dict(indices=list(indices), tokens=list(budget.count(indices)))


def predict(pool, rank, toks, models, thresholds):
    base, budget, table, edits, x = design(pool, rank, toks, "fusion")
    options = [dict(slot=slot, incoming=incoming) for slot, incoming in edits]
    heuristic = bridge_rule(table, list(base), options)
    chosen = {"bridge_rule": heuristic}
    for name, model in models.items():
        chosen[name] = best_edit(scores(model, x), edits, thresholds[name])
    selections = {name: rank["selections"][name] for name in METHODS if name in COMPARATORS}
    for name, choice in chosen.items():
        result = base if choice is None else apply_edit(base, (edits[choice],))
        if not budget.fits(result):
            raise ValueError("A proposed replacement exceeds fusion's evidence limit")
        selections[name] = result
    return {name: selection_report(selections[name], budget) for name in METHODS}


def labels(reference, pool, base, edits):
    support = support_target(reference, pool)
    title_masks = support["title_masks"]
    sentence_masks = support["sentence_masks"]
    masks = [support_groups(sentence_masks, title_masks, choice)
             for choice in [base, *(apply_edit(base, (edit,)) for edit in edits)]]
    full = (1 << len(title_masks)) - 1
    if not support["complete_mask"] or not title_masks:
        raise ValueError("No HotpotQA support groups")
    return [transition(masks[0], after) for after in masks[1:]], masks[0] == full, [m == full for m in masks[1:]]


def choose_threshold(validation, model):
    profiles = []
    for row in validation:
        values = scores(model, row["x"])
        profiles.append((row, values))
    metrics = []
    for threshold in THRESHOLDS:
        wins = losses = edits = 0
        for row, values in profiles:
            choice = best_edit(values, row["edits"], threshold)
            if choice is not None:
                edits += 1
                wins += int(row["complete"][choice] and not row["base_complete"])
                losses += int(row["base_complete"] and not row["complete"][choice])
        metrics.append(dict(threshold=threshold, wins=wins, losses=losses, edits=edits, net=wins-losses))
    return max(metrics, key=lambda r: (r["net"], -r["edits"])), metrics


def load_fusion(directory):
    policy = read(directory / "policy.json")
    expected = {f"{stem}{suffix}" for stem in ("fusion_scalar", "independent")
                for suffix in (".npz", ".npz.json")}
    if set(policy["files"]) != expected or any(sha(directory / f) != s for f, s in policy["files"].items()):
        raise ValueError("Fitted model hash mismatch")
    return {name: Student.load(directory / f"{name}.npz") for name in ("fusion_scalar", "independent")}, policy
