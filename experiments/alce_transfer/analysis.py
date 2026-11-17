import numpy as np

from .model import BASELINES, LEARNED, fit, proposal, target


def fit_models(questions, x, source_scores):
    if any(r["role"] != "A" for r in source_scores.values()):
        raise ValueError("Only A outcomes may be used as fitting targets")
    features, deltas, groups, ledger = [], [], [], []
    invalid = 0
    for q in questions:
        if q["role"] != "train":
            continue
        for j, action_index in enumerate(q["sampled"]):
            base = [source_scores[f"{q['qid']}|gtr_top5|{s}"] for s in (17, 29)]
            after = [source_scores[f"{q['qid']}|probe_{j}|{s}"] for s in (17, 29)]
            if not all(r["valid"] for r in (*base, *after)):
                invalid += 1
                continue
            delta = [a["metrics"]["utility"] - b["metrics"]["utility"] for a,b in zip(after, base, strict=True)]
            features.append(x[q["start"] + action_index])
            deltas.append(delta)
            groups.append(q["qid"])
            ledger.append(dict(qid=q["qid"], action_index=action_index, deltas=delta,
                               mean=float(np.mean(delta)), conservative=float(min(delta))))
    if len(features) < 64 or len(set(groups)) < 32:
        raise RuntimeError("Too few valid A labels to fit a 43-feature selector")
    models = {name:fit(np.asarray(features), np.asarray(deltas), np.asarray(groups), name) for name in LEARNED}
    return models, dict(labels=ledger, rejected_invalid_actions=invalid,
         training_questions=len(set(groups)), training_edits=len(features),
         mean_nonzero_labels=int(np.sum(np.abs(target(deltas, False)) > 1)),
         conservative_positive_labels=int(np.sum(target(deltas, True) > 1)),
         strict_seed_reversals=int(np.sum(np.prod(np.asarray(deltas), axis=1) < 0)))


def propose(questions, x, models, role):
    result = {}
    for q in questions:
        if q["role"] == role:
            block = x[q["start"]:q["end"]]
            result[q["qid"]] = {name:dict(zip(("index", "score"), proposal(q, block, model), strict=True))
                                 for name,model in models.items()}
    return result


def paired(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(20261117)
    indices = rng.integers(0, len(d), (10000, len(d)))
    interval = np.quantile(d[indices].mean(axis=1), [.025, .975]).tolist()
    flips = rng.choice((-1., 1.), size=(20000, len(d)))
    observed = abs(float(d.mean()))
    p = (1 + int(np.sum(np.abs((flips*d).mean(axis=1)) >= observed - 1e-12))) / 20001
    return dict(mean_difference=float(d.mean()), bootstrap95=interval, sign_flip_p=p,
                wins=int(np.sum(d > 1e-9)), losses=int(np.sum(d < -1e-9)), ties=int(np.sum(np.abs(d) <= 1e-9)))


def pilot_stability(scored, ids):
    states = ("pilot_retained", "gtr_top5", "probe_0")
    complete = [q for q in ids if all(scored[r][f"{q}|{tag}|{seed}"]["valid"]
                for r in ("A", "B") for tag in states for seed in (17, 29))]
    values = np.asarray([[[[scored[r][f"{q}|{tag}|{s}"]["metrics"]["utility"] for tag in states]
                            for s in (17,29)] for r in ("A", "B")] for q in complete]).reshape(-1,2,2,3)
    insertions = values[:,:,:,1:] - values[:,:,:,0:1]
    preferences = values[:,:,:,2] - values[:,:,:,1]

    def agreement(left, right):
        a = (np.asarray(left)>1).astype(int) - (np.asarray(left)<-1).astype(int)
        b = (np.asarray(right)>1).astype(int) - (np.asarray(right)<-1).astype(int)
        active = (a != 0) | (b != 0)
        n = int(active.sum())
        return dict(total=len(a), informative=n, both_tied=int(np.sum(~active)),
                    strict_reversals=int(np.sum(a*b == -1)),
                    agreement_excluding_both_tied=float(np.mean(a[active]==b[active])) if n else None)

    def comparisons(array):
        return dict(A_seed_repeat=agreement(array[:,0,0], array[:,0,1]),
                    B_seed_repeat=agreement(array[:,1,0], array[:,1,1]),
                    cross_reader_means=agreement(array[:,0].mean(axis=1), array[:,1].mean(axis=1)),
                    cross_reader_conservative=agreement(array[:,0].min(axis=1), array[:,1].min(axis=1)))
    return dict(questions=len(ids), complete_questions=len(complete), utility_tie_margin=1.,
                insert_original=comparisons(insertions[:,:,:,0]), insert_candidate=comparisons(insertions[:,:,:,1]),
                repair_preferences=comparisons(preferences),
                B_pilot_outcomes_used_for_fitting=False,
                interpretation="32-question pre-training stability measurement; B pilot labels never become training targets.")


def summarize(eval_scores, frozen):
    ids = frozen["evaluation_questions"]
    result = {}
    for role, rows in eval_scores.items():
        methods = frozen["evaluation_methods"][role]
        report = {}
        for method in methods:
            values = [rows[f"{q}|{method}|17"] for q in ids]
            keys = values[0]["metrics"].keys()
            report[method] = dict(metrics={k:float(np.mean([v["metrics"][k] for v in values])) for k in keys},
                                  valid=sum(v["valid"] for v in values), questions=len(values))
        result[role] = report
    strongest = frozen["strongest_fixed_baseline_from_A_calibration"]
    comparisons = {}
    for comparator in (*BASELINES, "mean_set", "conservative_independent"):
        comparisons[comparator] = {metric:paired(
            [eval_scores["B"][f"{q}|conservative_set|17"]["metrics"][metric] for q in ids],
            [eval_scores["B"][f"{q}|{comparator}|17"]["metrics"][metric] for q in ids])
            for metric in ("utility", "qampari_f1_top5", "citation_rec")}
    main = comparisons[strongest]
    checks = dict(two_point_utility_gain=main["utility"]["mean_difference"] >= 2.,
                  positive_answer_f1_change=main["qampari_f1_top5"]["mean_difference"] > 0.,
                  nonnegative_citation_recall_change=main["citation_rec"]["mean_difference"] >= 0.)
    return dict(results=result, B_comparisons=comparisons, primary_baseline=strongest,
                practical_encouragement=all(checks.values()), checks=checks,
                interpretation="Primary B comparison is frozen from A calibration; other comparisons are exploratory. No global novelty claim.")
