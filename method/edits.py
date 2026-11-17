"""Deterministic replacement candidates and token budget checks."""

import hashlib
from itertools import zip_longest


def apply_edit(base, edits):
    result = list(base)
    for slot, incoming in edits:
        result[slot] = incoming
    return tuple(result)


def deterministic_order(qid, value):
    return hashlib.sha256(f"paired-replacement-v1|{qid}|{value}".encode()).digest()


def shortlist(lists, limit, feasible):
    """Take up to limit new feasible candidates from each prioritized list."""
    taken = [0] * len(lists)
    found = {}
    for candidates in zip_longest(*lists):
        for index, candidate in enumerate(candidates):
            if candidate is None or candidate in found or taken[index] == limit:
                continue
            if feasible(candidate):
                found[candidate] = None
                taken[index] += 1
        if min(taken) == limit:
            break
    return list(found)


class TokenBudget:
    def __init__(self, passages, tokenizers, base):
        self.rendered = [p["rendered"] for p in passages]
        self.tokenizers = tokenizers
        self.cache = {}
        self.limit = self.count(base)

    def count(self, selection):
        selection = tuple(selection)
        if selection not in self.cache:
            text = "\n\n".join(self.rendered[i] for i in selection)
            self.cache[selection] = tuple(
                len(tokenizer.encode(text, add_special_tokens=False).ids)
                for tokenizer in self.tokenizers
            )
        return self.cache[selection]

    def fits(self, selection):
        return all(a <= b for a, b in zip(self.count(selection), self.limit, strict=True))


def replace(seed, descriptor):
    result = list(seed)
    result[descriptor['slot']] = descriptor['incoming']
    if len(result) != 5 or len(set(result)) != 5:
        raise ValueError('Replacement must preserve five unique passages')
    return result
