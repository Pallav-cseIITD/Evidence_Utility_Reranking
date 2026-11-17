"""Checks for answer scoring, evidence attribution and paired comparisons."""

import unittest
import numpy as np

from experiments.hotpot.evaluate import compare
from experiments.hotpot.scoring import grade, normalize_answer, parse_answer_output
from experiments.hotpot.statistics import paired_binary


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(answer="Town", context=dict(title=["Film", "Director"],
            sentences=[["Arin directed Film."], ["Arin was born in Town."]]),
            supporting_facts=dict(title=["Film", "Director"], sent_id=[0, 0]))
        self.case = dict(evidence=[dict(id="D01", title="Film", rendered="Arin directed Film."),
                                  dict(id="D02", title="Director", rendered="Arin was born in Town.")])

    def draw(self, citations):
        return dict(strict_valid=True, hit_token_cap=False,
                    answers=[dict(answer="Town", citations=citations)])

    def test_correct_answer_requires_both_links_for_attribution(self):
        complete = grade(self.reference, self.case, self.draw(["D01", "D02"]))
        missing = grade(self.reference, self.case, self.draw(["D02"]))
        self.assertTrue(complete["answer_em"] and complete["attributed_em"])
        self.assertTrue(missing["answer_em"])
        self.assertFalse(missing["attributed_em"])

    def test_unknown_citation_prevents_attribution(self):
        result = grade(self.reference, self.case, self.draw(["D01", "D02", "D99"]))
        self.assertTrue(result["answer_em"])
        self.assertFalse(result["attributed_em"])

    def test_invalid_or_capped_answers_are_failures(self):
        for key in ("strict_valid", "hit_token_cap"):
            draw = self.draw(["D01", "D02"])
            draw[key] = key == "hit_token_cap"
            result = grade(self.reference, self.case, draw)
            self.assertFalse(result["valid"])
            self.assertIsNone(result["answer_em"])

    def test_answer_normalization_preserves_the_original_convention(self):
        self.assertEqual(normalize_answer(" The Town! "), "town")

    def test_malformed_answer_schema_is_not_repaired(self):
        answers, error = parse_answer_output('{"answers":[{"answer":"Town"}]}')
        self.assertEqual(answers, [])
        self.assertIsNotNone(error)

    def test_multiple_returned_answers_fail_single_answer_exact_match(self):
        draw = self.draw(["D01", "D02"])
        draw["answers"].append(dict(answer="Other", citations=["D02"]))
        self.assertFalse(grade(self.reference, self.case, draw)["answer_em"])


class ComparisonTests(unittest.TestCase):
    def test_all_ties_have_no_gain(self):
        result = paired_binary([True, False], [True, False])
        self.assertEqual(result["wins"], 0)
        self.assertEqual(result["losses"], 0)
        self.assertEqual(result["exact_mcnemar_two_sided"], 1.)
        self.assertEqual(result["bootstrap_95_pp"], [0., 0.])

    def test_four_wins_produce_the_exact_two_sided_probability(self):
        result = paired_binary([1, 1, 1, 1], [0, 0, 0, 0])
        self.assertEqual(result["exact_mcnemar_two_sided"], 0.125)
        self.assertEqual(result["difference_pp"], 100.)

    def test_changed_result_is_detected(self):
        with self.assertRaisesRegex(ValueError, "Different numeric result"):
            compare(dict(score=0.1), dict(score=0.2))
        compare(dict(score=0.1 + 1e-16), dict(score=0.1))


if __name__ == "__main__":
    unittest.main()
