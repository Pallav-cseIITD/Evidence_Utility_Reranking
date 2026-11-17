"""Selector checks using small constructed inputs and the saved model files."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

from method.check_reproduction import check
from method.edits import TokenBudget, apply_edit, shortlist
from method.features import FeatureTable
from method.inputs import load_tokenizers, verify_inputs
from method.io import public_entries, read, sha, write
from method.model import Student, distribution
from method.policy import THRESHOLDS, best_edit, labels, load_fusion
from method.sampling import training_indices
from method.select import DEFAULT_MODEL, select_batch
from method.support import evidence_masks, support_groups, support_target, transition
from method.train import train


def example(qid="example"):
    passages = []
    for index in range(50):
        title = f"Document {index}"
        text = "This passage discusses another subject and gives no relevant fact about the question."
        if index == 0:
            title, text = "Film", "Film was directed by Arin."
        elif index == 5:
            title, text = "Director", "Arin was born in Town."
        passages.append(dict(title=title, body_for_scoring=text,
                             rendered=f"[D{index + 1:02d}] {title}\n{text}", reader_token_counts=[12, 12]))
    pool = dict(qid=qid, question=f"Where was the director of Film born? {qid}", passages=passages)
    ranking = dict(
        ce_scores=[2.0 if index == 5 else 1 - index / 50 for index in range(50)],
        embeddings=np.eye(50).tolist(),
        selections={name: [0, 1, 2, 3, 4] for name in (
            "fusion_top5", "relevance_ce_top5", "dense_mmr_0.8", "submodular_coverage_top5",
        )},
    )
    reference = dict(
        id=qid, context=dict(title=["Film", "Director"],
                            sentences=[["Film was directed by Arin."], ["Arin was born in Town."]]),
        supporting_facts=dict(title=["Film", "Director"], sent_id=[0, 0]),
    )
    return pool, ranking, reference


class LabelTests(unittest.TestCase):
    def test_gain_and_loss_account_for_the_removed_passage(self):
        pool, _, reference = example()
        targets, before, complete = labels(reference, pool, [0, 1, 2, 3, 4], [(1, 5), (0, 5)])
        self.assertFalse(before)
        self.assertEqual(targets[0], dict(gain=1, loss=0, unsupported=0))
        self.assertEqual(targets[1], dict(gain=1, loss=1, unsupported=0))
        self.assertEqual(complete, [True, False])

    def test_partial_document_support_is_not_a_complete_group(self):
        pool, _, reference = example()
        reference["context"]["sentences"][0].append("The film was released in 1990.")
        reference["supporting_facts"]["title"].append("Film")
        reference["supporting_facts"]["sent_id"].append(1)
        target = support_target(reference, pool)
        self.assertEqual(support_groups(target["sentence_masks"], target["title_masks"], [0]), 0)
        self.assertEqual(transition(3, 2), dict(gain=0, loss=1, unsupported=0))

    def test_sentence_match_requires_its_supporting_title(self):
        _, _, reference = example()
        masks, full = evidence_masks(reference, [dict(title="Other", text="Arin was born in Town.")], "text")
        self.assertEqual(masks, [0])
        self.assertEqual(full, 3)


class FeatureAndPolicyTests(unittest.TestCase):
    def test_features_have_no_reference_input(self):
        pool, ranking, _ = example()
        original = np.asarray(FeatureTable(pool, ranking).features([0, 1, 2, 3, 4], 1, 5))
        pool["supporting_facts"] = dict(title=["irrelevant metadata"])
        pool["answers"] = ["irrelevant metadata"]
        table = FeatureTable(pool, ranking)
        np.testing.assert_array_equal(original, table.features([0, 1, 2, 3, 4], 1, 5))
        self.assertEqual(original.shape, (43,))
        control = table.features([0, 1, 2, 3, 4], 1, 5, independent=True)
        np.testing.assert_array_equal(control[:11], original[:11])
        np.testing.assert_array_equal(control[11:], np.zeros(32))

    def test_threshold_is_strict_and_ties_use_passage_then_slot(self):
        edits = [(2, 9), (3, 8), (1, 8)]
        self.assertIsNone(best_edit(np.array([0.15]), [(0, 5)], 0.15))
        self.assertEqual(best_edit(np.array([0.2, 0.2, 0.2]), edits, 0.15), 2)
        self.assertIsNone(best_edit(np.array([1.0]), [(0, 5)], None))
        self.assertIsNone(best_edit(np.array([]), [], 0.15))

    def test_token_budget_checks_both_counters(self):
        class Counter:
            def __init__(self, characters):
                self.characters = characters

            def encode(self, text, add_special_tokens=False):
                return SimpleNamespace(ids=list(text) if self.characters else text.split())

        passages = [dict(rendered=text) for text in ("aa bb", "x", "longword", "y")]
        budget = TokenBudget(passages, [Counter(False), Counter(True)], [0, 1])
        self.assertEqual(budget.count([0, 1])[0], budget.count([0, 2])[0])
        self.assertFalse(budget.fits([0, 2]))
        self.assertTrue(budget.fits([0, 3]))
        self.assertEqual(apply_edit([0, 1], [(1, 3)]), (0, 3))

    def test_shortlist_skips_infeasible_candidates_and_duplicates(self):
        self.assertEqual(shortlist([[1, 2, 3], [1, 4, 5]], 1, lambda value: value != 1), [2, 4])


class ModelAndInputTests(unittest.TestCase):
    def test_saved_models_have_the_expected_shape_and_thresholds(self):
        loaded, policy = load_fusion(DEFAULT_MODEL)
        self.assertEqual(policy["thresholds"], dict(fusion_scalar=0.15, independent=0.3))
        for model in loaded.values():
            self.assertEqual(sum(weight.size for weight in model.w), 1441)
            self.assertEqual(model.mean.shape, (43,))

    def test_model_save_and_load_preserves_predictions(self):
        x = np.random.default_rng(17).normal(size=(24, 43))
        y = np.stack([distribution([dict(gain=i % 2, loss=0, unsupported=0)]) for i in range(24)])
        model = Student("scalar", seed=17, hidden=32).fit(x[:16], y[:16], (x[16:], y[16:]), epochs=3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            model.save(path)
            np.testing.assert_array_equal(model.predict(x)["mean_gain"], Student.load(path).predict(x)["mean_gain"])

    def test_streamed_input_formats(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "records.json"
            rows = [dict(qid="a"), dict(qid="b")]
            write(path, rows)
            self.assertEqual(list(public_entries(path)), rows)
            write(path, dict(a=dict(value=1), b=dict(value=2)))
            self.assertEqual(dict(public_entries(path, True)), read(path))

    def test_sampling_balances_outcomes(self):
        info = dict(
            questions=dict(q1=dict(start=0, end=6)),
            edits=[dict(slot=0, incoming=i) for i in range(6)],
            targets=[dict(gain=1, loss=0), dict(gain=2, loss=0), dict(gain=0, loss=1),
                     dict(gain=1, loss=1), dict(gain=0, loss=0), dict(gain=0, loss=0)],
            protocol=dict(train_sampling_per_question=dict(clean_gain=1, harm=1, neutral=1)),
        )
        picked = training_indices(info, ["q1"])
        self.assertEqual(picked, training_indices(info, ["q1"]))
        self.assertEqual(len(picked), 3)
        categories = [(info["targets"][i]["gain"] > 0, info["targets"][i]["loss"] > 0) for i in picked]
        self.assertEqual(sum(harm for _, harm in categories), 1)
        self.assertEqual(sum(gain and not harm for gain, harm in categories), 1)
        self.assertEqual(sum(not gain and not harm for gain, harm in categories), 1)

    def test_training_and_batch_selection_with_constructed_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            raw = folder / "inputs"
            raw.mkdir()
            token_paths = [folder / "tokenizer_a.json", folder / "tokenizer_b.json"]
            tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            for path in token_paths:
                tokenizer.save(str(path))
            qids = ["train_a", "train_b", "validation_a"]
            rows = [example(qid) for qid in qids]
            write(raw / "pools.json", [row[0] for row in rows])
            write(raw / "features.json", {qid: row[1] for qid, row in zip(qids, rows, strict=True)})
            write(raw / "train_references.json", [row[2] for row in rows[:2]])
            write(raw / "validation_references.json", [rows[2][2]])
            write(raw / "cohort_freeze.json", dict(question_ids=dict(train=qids[:2], validation=qids[2:], test=[])))
            manifest = dict(
                input_sha256={path.name: sha(path) for path in raw.iterdir()},
                tokenizer_sha256=[sha(path) for path in token_paths],
                question_counts=dict(train=2, validation=1),
                threshold_grid=list(THRESHOLDS), model=dict(seed=17, hidden=32, epochs=3),
                train_sampling_per_question=dict(clean_gain=4, harm=4, neutral=4),
            )
            manifest_path = folder / "manifest.json"
            write(manifest_path, manifest)
            output = folder / "fitted"
            with redirect_stdout(io.StringIO()):
                receipt = train(raw, output, token_paths, manifest_path)
            self.assertEqual(receipt["train_questions"], 2)
            self.assertEqual(receipt["validation_questions"], 1)
            self.assertEqual(receipt["reader_calls"], 0)
            actual = select_batch(output, raw / "pools.json", raw / "features.json", token_paths)
            self.assertEqual(len(actual), 3)
            expected = folder / "selections.json"
            write(expected, actual)
            self.assertTrue(check(output, raw / "pools.json", raw / "features.json", expected, token_paths)["passed"])
            for row in actual:
                baseline = row["methods"]["fusion_top5"]
                for name in ("fusion_scalar", "independent"):
                    selected = row["methods"][name]
                    self.assertEqual(len(set(selected["indices"])), 5)
                    self.assertTrue(all(a <= b for a, b in zip(selected["tokens"], baseline["tokens"], strict=True)))
            actual[0]["methods"]["fusion_scalar"]["indices"] = [999, 1, 2, 3, 4]
            write(expected, actual)
            self.assertFalse(check(output, raw / "pools.json", raw / "features.json", expected, token_paths)["passed"])
            original = (raw / "pools.json").read_bytes()
            (raw / "pools.json").write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_inputs(raw, manifest)
            with self.assertRaisesRegex(ValueError, "hashes or order"):
                load_tokenizers(token_paths, ["0" * 64, "0" * 64])


if __name__ == "__main__":
    unittest.main()
