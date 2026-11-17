import json

from method.support import facts, normalize


def parse_answer_output(text):
    """Parse a strict schema; never silently repair malformed model answers."""
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    try:
        result = json.loads(stripped)
        if not isinstance(result, dict) or set(result) != {"answers"} or not isinstance(result["answers"], list):
            raise ValueError("Expected an object with an answers list")
        for item in result["answers"]:
            if (not isinstance(item, dict) or set(item) != {"answer", "citations"}
                or not isinstance(item["answer"], str) or not item["answer"].strip()
                or not isinstance(item["citations"], list)
                or not all(isinstance(c, str) for c in item["citations"])):
                raise ValueError("Invalid answer/citations schema")
        return result["answers"], None
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return [], str(exc)


def normalize_answer(text):
    # The official HotpotQA EM convention lowercases, strips punctuation/articles,
    # then collapses whitespace. Keep this separate from support-span normalization.
    import re
    import string

    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def grade(reference, case, draw):
    if not draw['strict_valid'] or draw['hit_token_cap']:
        return dict(valid=False, answer_em=None, attributed_em=None)
    answers = draw['answers']
    correct = len(answers) == 1 and normalize_answer(answers[0]['answer']) == normalize_answer(reference['answer'])
    supported = False
    if correct:
        cited = set(answers[0]['citations'])
        docs = {d['id']: d for d in case['evidence']}
        required = facts(reference)
        supported = cited <= docs.keys() and all(any(normalize(docs[c]['title']) == normalize(f['title'])
            and ' '+normalize(f['sentence'])+' ' in ' '+normalize(docs[c]['rendered'])+' '
            for c in cited) for f in required)
    return dict(valid=True, answer_em=bool(correct), attributed_em=bool(supported))
