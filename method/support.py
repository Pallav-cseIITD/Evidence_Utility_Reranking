"""Benchmark support matching and marginal evidence labels."""

import html
import re


def normalize(text):
    return ' '.join(re.findall(r'\w+', html.unescape(text).casefold()))


def facts(row):
    contexts = dict(zip(row['context']['title'], row['context']['sentences'], strict=True))
    result = []
    for title, number in zip(row['supporting_facts']['title'], row['supporting_facts']['sent_id'], strict=True):
        sentence = contexts[title][number]
        if not normalize(sentence):
            raise ValueError('An annotated supporting sentence is empty')
        result.append(dict(title=title, sentence=sentence))
    if len({normalize(r['title']) for r in result}) != 2:
        raise ValueError('Expected two distinct gold document titles')
    return result


def evidence_masks(reference, passages, text_field):
    support = facts(reference)
    masks = []
    for doc in passages:
        title, text = normalize(doc['title']), ' '+normalize(doc[text_field])+' '
        mask = sum(1 << i for i, fact in enumerate(support)
                   if title == normalize(fact['title']) and ' '+normalize(fact['sentence'])+' ' in text)
        masks.append(mask)
    return masks, (1 << len(support))-1


def covered(masks, indices):
    result = 0
    for index in indices:
        result |= masks[index]
    return result


def support_groups(sentence_masks, title_masks, indices):
    merged = 0
    for i in indices:
        merged |= int(sentence_masks[i])
    return sum(1 << j for j, mask in enumerate(title_masks) if merged & mask == mask)


def transition(before, after):
    return dict(gain=(after & ~before).bit_count(), loss=(before & ~after).bit_count(), unsupported=0)


def support_target(reference,pool):
    masks,complete=evidence_masks(reference,pool['passages'],'body_for_scoring')
    titles=[];support=facts(reference)
    for f in support:
        title=normalize(f['title'])
        if title not in titles:
            titles.append(title)
    title_masks=[sum(1<<i for i,f in enumerate(support) if normalize(f['title'])==title) for title in titles]
    return dict(sentence_masks=masks,complete_mask=complete,title_masks=title_masks)
