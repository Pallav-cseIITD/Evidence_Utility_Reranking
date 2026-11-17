# HotpotQA evaluation

This experiment evaluates marginal support repair initialized by fusion on the 128 final evaluation questions. The selector was trained on 512 questions and calibrated on 128 different questions. The datasets are research subsets of the public development collection.

## Reproduce the results on CPU

Install `requirements/analysis.txt` and restore the `hotpot_eval` data group. From the repository root:

```bash
python3 -m experiments.hotpot.evaluate
```

The command writes a new `outputs/hotpot_evaluation/` directory containing:

| File | Contents |
| :--- | :--- |
| `results.json` | Evidence metrics, answer metrics and paired statistical comparisons |
| `answers_per_question.json` | Recomputed scores for each question, method and reader |
| `verification.json` | Agreement with the original results and the preserved reader qualification outcomes |

Use `--data /path/to/hotpot_eval` for a different location of the restored inputs, or `--output outputs/another_evaluation` for a new output directory. Existing results are not overwritten.

The replay parses the saved answer text again, checks its recorded parsing status, grades the answer against the reference, and checks every required supporting fact in the cited passages. It also recalculates evidence coverage, ranking scores and the embedding similarity proxy from the prepared pools. It compares all resulting metrics and statistical comparisons against the original saved results. A mismatch stops the command.

The saved records contain 2,944 logical question and method responses, including reused generations: 640 for the primary reader and 1,152 for each additional reader. This replay makes no new reader calls. Original qualification audits are preserved in `saved/qualification.json`; the replay recalculates the natural question results, rather than regenerating qualification responses or repeating the original GPU provenance audit.

## Main comparison

| Selection | Complete evidence chains | Correct answers | Correct answers with complete cited support |
| :--- | :--- | :--- | :--- |
| Fusion | 54 | 39 | 8 |
| Marginal support repair | 67 | 42 | 15 |
| Incoming passage control | 55 | 39 | 10 |
| Relevance selection | 57 | 37 | 15 |
| MMR | 57 | 35 | 14 |

Answer columns use the primary Qwen3 8B reader, which passed the original 24 controls and all 12 expanded answer type controls. Repair versus fusion has nine attributed answer wins and two losses, with a paired p value of 0.0654296875. Evidence coverage improved more clearly than answer quality in this collection.

The later Qwen3 14B and Phi 4 comparisons include nine evidence arrangements: the five above, similarity coverage, forced replacement and reversed fusion and repair order. They reuse the same 128 questions. Both passed the original controls; they passed 11 and nine of the 12 expanded controls respectively. Their results remain diagnostic and are kept separate from the primary result.

## Metric definitions

An answer is exact only when the reader returns one answer matching the reference after the original HotpotQA normalization. Attributed exact match additionally requires all annotated supporting facts in the cited passages, with valid passage identifiers. This is a benchmark support check. Invalid or capped responses stay in the denominator as failures.

A complete evidence chain contains every annotated supporting sentence for the question. Supporting sentence coverage, supporting title coverage and the pool based ranking score use the same annotations. Pairwise embedding similarity measures similarity of passage representations; contradiction was not measured.

Primary answer comparisons use the original seed 20261116 for 10,000 paired bootstrap samples. Additional reader and evidence comparisons use seed 20261117. Exact paired binary tests are reported alongside confidence intervals. Comparisons beyond repair versus fusion are exploratory.

## Files and checks

`scoring.py`, `statistics.py` and `additional.py` retain the original answer grading and comparison functions. `evaluate.py` connects them to the portable prepared data. Saved responses retain their draw identifiers so reused requests can be traced. Frozen evidence arrangements, original qualification outcomes, expected results and input hashes are under `saved/`.

```bash
python3 -m unittest discover -s experiments/hotpot/tests -v
```

These tests cover incomplete citations, unknown citations, invalid and capped answers, answer normalization, multiple answers and paired statistical calculations. `reproduction.json` records the completed replay on the actual saved experiment data.
