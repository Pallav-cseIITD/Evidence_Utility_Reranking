"""Balanced sampling of training replacements within each question."""

import hashlib


def training_indices(info, qids):
    """Same deterministic, per-question balanced training rows for every model."""
    output = []
    for qid in qids:
        row = info['questions'][qid]
        categories = dict(clean_gain=[], harm=[], neutral=[])
        for i in range(row['start'], row['end']):
            y = info['targets'][i]
            category = 'harm' if y['loss'] else ('clean_gain' if y['gain'] else 'neutral')
            categories[category].append(i)
        for name, indices in categories.items():
            ordered = sorted(indices, key=lambda i: hashlib.sha256(
                ('hotpot-edit-sample-v1'+qid+str(info['edits'][i])).encode()).hexdigest())
            output += ordered[:info['protocol']['train_sampling_per_question'][name]]
    return output
