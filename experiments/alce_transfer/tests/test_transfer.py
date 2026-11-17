"""Checks for source reader labels, calibration and paired evaluation."""

import unittest
import numpy as np

from experiments.alce_transfer.analysis import fit_models, paired
from experiments.alce_transfer.model import choose_threshold, fit, select, target, utility


class TransferTests(unittest.TestCase):
    def test_mean_and_conservative_targets_use_the_same_two_draws(self):
        changes = [[5., -1.], [4., 2.]]
        np.testing.assert_array_equal(target(changes, False), [2., 3.])
        np.testing.assert_array_equal(target(changes, True), [-1., 2.])

    def test_target_requires_two_finite_observations(self):
        for changes in ([[1.]], [[1., float("nan")]]):
            with self.assertRaises(ValueError):
                target(changes, True)

    def test_reader_B_cannot_supply_fitting_targets(self):
        with self.assertRaisesRegex(ValueError, "Only A outcomes"):
            fit_models([], np.empty((0, 43)), {"example": {"role": "B"}})

    def test_utility_is_the_per_response_harmonic_mean(self):
        self.assertEqual(utility(dict(qampari_f1_top5=20., citation_rec=60.)), 30.)
        self.assertEqual(utility(dict(qampari_f1_top5=0., citation_rec=0.)), 0.)

    def test_each_question_has_equal_total_fitting_weight(self):
        x = np.zeros((10, 43))
        changes = np.array([[1., 1.]] + [[0., 0.]] * 9)
        groups = np.array(["one_action"] + ["nine_actions"] * 9)
        fitted = fit(x, changes, groups, "mean_set")
        self.assertAlmostEqual(fitted["coefficients"][-1], 0.5)

    def test_harmful_calibration_proposals_can_select_keep_all(self):
        records = [dict(score=0.6, delta=-3.), dict(score=1.2, delta=-1.)]
        self.assertIsNone(choose_threshold(records))

    def test_selection_uses_a_strict_threshold(self):
        question = dict(actions=[dict(selected=[0, 1, 2, 3, 8])])
        self.assertEqual(select(question, 0, 0.5, 0.5), [0, 1, 2, 3, 4])
        self.assertEqual(select(question, 0, 0.6, 0.5), [0, 1, 2, 3, 8])
        self.assertEqual(select(question, 0, 20., None), [0, 1, 2, 3, 4])

    def test_tied_paired_scores_have_no_improvement(self):
        result = paired([1., 2., 3.], [1., 2., 3.])
        self.assertEqual(result["mean_difference"], 0.)
        self.assertEqual(result["bootstrap95"], [0., 0.])
        self.assertEqual(result["sign_flip_p"], 1.)
        self.assertEqual(result["ties"], 3)


if __name__ == "__main__":
    unittest.main()
