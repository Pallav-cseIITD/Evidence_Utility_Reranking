from .statistics import paired_binary

METHODS = ("fusion_top5", "fusion_scalar", "dense_mmr_0.8", "relevance_ce_top5", "independent", "submodular_coverage_top5", "forced_repair", "fusion_reverse", "repair_reverse")
PRIMARY = ("fusion_scalar", "fusion_top5")


def summarize(rows, qids):
    lookup = {(r["qid"], r["method"]):r for r in rows}
    methods = {}
    for name in METHODS:
        ordered = [lookup[(q, name)] for q in qids if (q, name) in lookup]
        n = len(ordered)
        methods[name] = dict(completed=n, invalid=sum(not r["valid"] for r in ordered),
            answer_em=sum(bool(r["answer_em"]) for r in ordered),
            attributed_em=sum(bool(r["attributed_em"]) for r in ordered),
            answer_em_percent=100*sum(bool(r["answer_em"]) for r in ordered)/n if n else None,
            attributed_em_percent=100*sum(bool(r["attributed_em"]) for r in ordered)/n if n else None)
    comparisons = {}
    pairs = [PRIMARY, ("fusion_scalar", "relevance_ce_top5"), ("fusion_scalar", "dense_mmr_0.8"),
             ("fusion_scalar", "submodular_coverage_top5"), ("fusion_scalar", "independent"),
             ("fusion_scalar", "forced_repair"), ("repair_reverse", "fusion_scalar"),
             ("fusion_reverse", "fusion_top5")]
    for left, right in pairs:
        ids = [q for q in qids if (q, left) in lookup and (q, right) in lookup]
        if not ids:
            continue
        comparisons[left+"_minus_"+right] = {metric:paired_binary(
            [bool(lookup[q, left][metric]) for q in ids], [bool(lookup[q, right][metric]) for q in ids])
            for metric in ("answer_em", "attributed_em")}
    return dict(methods=methods, comparisons=comparisons,
                inference="Primary: fusion_scalar versus fusion_top5, attributed EM. Other comparisons exploratory; no family-wide significance claim.")
