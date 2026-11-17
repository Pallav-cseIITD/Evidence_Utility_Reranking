# Results

The compact tables under `summary/` are exported from the original saved experiment summaries. The CPU replays under `experiments/` reproduced these summaries before export. Raw responses, cached measurements and original qualification records remain with their experiments.

| File | Contents |
| :--- | :--- |
| [hotpot_evidence.csv](summary/hotpot_evidence.csv) | Evidence coverage and retrieval diagnostics on the 128 final questions |
| [hotpot_answers.csv](summary/hotpot_answers.csv) | Answer and attribution counts, with diagnostic readers marked |
| [qampari_reader_b.csv](summary/qampari_reader_b.csv) | Reader B answer, citation and utility scores |
| [hotpot_results.json](summary/hotpot_results.json) | Full HotpotQA aggregates and paired comparisons |
| [qampari_results.json](summary/qampari_results.json) | Full QAMPARI aggregates and transfer comparisons |

## Main findings

On HotpotQA, repair increased complete supporting evidence from 54 to 67 questions. With the primary reader, correct answers increased from 39 to 42 and correct answers with complete cited support from eight to 15. Relevance selection also produced 15 attributed answers. The primary repair versus fusion comparison had nine wins and two losses, with a paired p value of 0.0654296875. These findings support the evidence coverage gain more clearly than the answer quality gain.

The additional Qwen3 14B and Phi 4 results use the same questions. Their expanded qualification checks failed, so their answer comparisons remain diagnostic.

On QAMPARI, all three learned policies kept the original GTR evidence. Their combined utility on reader B was 11.30, compared with 14.21 for MMR, selected using reader A calibration. This experiment did not establish improved transfer from A's labels to B.

## Recreate the tables

After restoring the data, run the two CPU replays and export their results:

```bash
python3 -m experiments.hotpot.evaluate
python3 -m experiments.alce_transfer.evaluate
python3 results/export.py \
  outputs/hotpot_evaluation/results.json \
  outputs/alce_transfer_evaluation/result.json \
  --output outputs/result_tables
```

Each output path must be new. HotpotQA replay regrades saved answer text and evidence. QAMPARI replay refits models and recalculates comparisons while reusing official answer and citation measurements. These commands do not perform new GPU inference.

CSV answer values are question counts. QAMPARI correctness and citation scores range from zero to 100; combined utility is averaged per question. Evidence recalls in the CSV are fractions. Counts from different datasets or development experiments are not direct comparisons.
