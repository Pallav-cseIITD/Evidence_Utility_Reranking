"""Passage and evidence set features available at selection time."""

import re

import numpy as np

from .edits import replace


def words(text):
    return re.findall(r'\w+', text.casefold())


def scalar_stats(values):
    return [float(np.min(values)), float(np.mean(values)), float(np.max(values))]


class FeatureTable:
    """Only query, public passages and frozen relevance/embedding scores are read."""
    def __init__(self, pool, ranking):
        self.ce = np.asarray(ranking['ce_scores'], dtype=float)
        self.n = len(self.ce)
        if self.n != 50 or len(pool['passages']) != self.n:
            raise ValueError('Expected the fixed natural top 50')
        order = sorted(range(self.n), key=lambda i: (-self.ce[i], i))
        self.percentile = np.zeros(self.n)
        for j, i in enumerate(order):
            self.percentile[i] = 1-j/(self.n-1)
        self.ce_z = (self.ce-self.ce.mean())/max(self.ce.std(), 1e-4)
        block = np.asarray(ranking['embeddings'], dtype=float)
        block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-10)
        self.sim = block @ block.T
        query = set(words(pool['question']))
        self.doc = []
        titles, bodies = [], []
        for d in pool['passages']:
            title = words(d['title']); body = words(d['body_for_scoring'])
            titles.append(title); bodies.append(' '+' '.join(body)+' ')
            tq, bq = set(title) & query, set(body) & query
            self.doc.append([len(tq)/max(1, len(query)), len(bq)/max(1, len(query)),
                len(tq)/max(1, len(set(title))), len(bq)/max(1, len(set(body))),
                len(title)/20, len(body)/240, *[x/240 for x in d['reader_token_counts']]])
        self.doc = np.asarray(self.doc, dtype=float)
        self.link = np.array([[float(i != j and bool(titles[i]) and
            ' '+' '.join(titles[i])+' ' in bodies[j]) for j in range(self.n)] for i in range(self.n)])
        self.feature_names = [f'incoming_{k}' for k in ('ce_z', 'ce_percentile', 'fusion_rank',
            'query_title_recall', 'query_body_recall', 'title_query_precision', 'body_query_precision',
            'title_words', 'body_words', 'tokens_qwen25', 'tokens_qwen3')]
        self.feature_names += ['outgoing_'+n[len('incoming_'):] for n in self.feature_names]
        self.feature_names += ['ce_delta', 'percentile_delta', 'token_delta_qwen25', 'token_delta_qwen3',
            'slot', 'retained_ce_min', 'retained_ce_mean', 'retained_ce_max',
            'incoming_similarity_min', 'incoming_similarity_mean', 'incoming_similarity_max',
            'outgoing_similarity_min', 'outgoing_similarity_mean', 'outgoing_similarity_max',
            'incoming_title_in_retained', 'retained_title_in_incoming',
            'outgoing_title_in_retained', 'retained_title_in_outgoing', 'link_delta',
            'retained_mutual_links', 'incoming_outgoing_similarity']

    def individual(self, i):
        return [self.ce_z[i], self.percentile[i], (i+1)/50, *self.doc[i]]

    def features(self, seed, slot, incoming, independent=False):
        out = seed[slot]
        retained = [i for i in seed if i != out]
        a, b = self.individual(incoming), self.individual(out)
        additions = [self.link[incoming, retained].sum(), self.link[retained, incoming].sum()]
        removals = [self.link[out, retained].sum(), self.link[retained, out].sum()]
        state = [self.ce_z[incoming]-self.ce_z[out], self.percentile[incoming]-self.percentile[out],
            self.doc[incoming,-2]-self.doc[out,-2], self.doc[incoming,-1]-self.doc[out,-1], slot/4,
            *scalar_stats(self.ce_z[retained]), *scalar_stats(self.sim[incoming,retained]),
            *scalar_stats(self.sim[out,retained]), *additions, *removals,
            sum(additions)-sum(removals), self.link[np.ix_(retained,retained)].sum(), self.sim[incoming,out]]
        result = a + ([0.]*len(b) if independent else b) + ([0.]*len(state) if independent else state)
        if len(result) != len(self.feature_names) or not np.isfinite(result).all():
            raise ValueError('Invalid public feature vector')
        return result

    def link_delta(self, seed, descriptor):
        after = replace(seed, descriptor)
        return float(self.link[np.ix_(after,after)].sum()-self.link[np.ix_(seed,seed)].sum())


def bridge_rule(table, seed, descriptors):
    # A fixed, reference-free graph heuristic competes with the learned model.
    # Keep unless a new title link is added without going below the seed's
    # weakest cross-encoder relevance. No threshold is fitted on outcomes.
    options = [i for i, d in enumerate(descriptors) if table.link_delta(seed,d) > 0
               and table.ce[d['incoming']] >= min(table.ce[seed])]
    if not options:
        return None
    return max(options, key=lambda i: (table.link_delta(seed,descriptors[i]),
        table.ce[descriptors[i]['incoming']]-table.ce[seed[descriptors[i]['slot']]],
        -descriptors[i]['incoming'], -descriptors[i]['slot']))
