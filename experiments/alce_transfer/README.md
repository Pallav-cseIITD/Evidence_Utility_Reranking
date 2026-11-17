# QAMPARI utility transfer

This experiment asks whether marginal utility labels from reader A can train a selector that improves reader B. Reader A is Qwen3 14B and reader B is Phi 4. Both passed the original 24 QAMPARI reader qualification controls. Their exact revisions and generation settings are recorded in `configs/alce_transfer.json`.

The experiment uses ALCE's released GTR passage ranking, retaining its first 50 raw passages and selecting five. It samples 128 training, 40 calibration and 100 evaluation questions within the released evaluation collection. It is a research split, not a complete official dataset training run.

## Reproduce the learning experiment on CPU

With `requirements/analysis.txt` installed, restore the prepared QAMPARI inputs:

```bash
python3 data/prepare.py restore \
  --archives-dir /path/to/release_assets \
  --groups alce_transfer
python3 -m experiments.alce_transfer.evaluate
```

The replay writes a new `outputs/alce_transfer_evaluation/` directory. Use `--data /path/to/alce_transfer` or `--output outputs/another_transfer_evaluation` to choose other locations. Existing outputs are not overwritten.

It recalculates utility from the saved official answer and citation scores, constructs the intervention labels, refits the three predictors, repeats calibration using A alone, reconstructs evaluation selections, and recalculates stability and paired statistical comparisons. All outputs are checked against the original saved experiment, with numerical tolerance of one part in a billion for floating point results. Discrete choices must match exactly.

The official answer scores and citation support measurements are cached inputs. This CPU replay reuses them; it does not regenerate answers, rerun the citation support model or independently validate its judgments. The original reader and citation evaluator qualification records are retained under `saved/`.

## Label and model design

For each training intervention, compare the original five passages with a proposed single replacement. Reader A generates answers at seeds 17 and 29. Utility is the harmonic mean of ALCE's answer score with its five answer adjustment and citation coverage, computed for each response on a zero to 100 scale.

The ordinary label averages the two utility changes. The conservative label takes their minimum, rewarding an exchange only when both observed changes are positive. A third predictor uses the conservative label with incoming passage features only. All three are ridge regression models with penalty 1, standardized inputs and equal total weight for each training question. The first two use 43 features; the control retains the first 11 incoming passage features.

Ten proposed actions had invalid source outputs and were excluded from fitting. The remaining 502 labels cover 126 training questions. Calibration uses the separate 40 questions and reader A to choose thresholds and the strongest fixed baseline. Reader B's stability measurements are diagnostic and never fitting targets.

## Result

Every learned policy selected the option to keep the starting evidence during source calibration. All consequently retained the original GTR selection on the 100 evaluation questions. MMR was selected as the strongest fixed baseline using A's calibration results.

| Method on reader B | Answer score | Citation coverage | Citation precision | Combined utility |
| :--- | :--- | :--- | :--- | :--- |
| Original GTR ranking | 15.01 | 28.95 | 27.67 | 11.30 |
| Relevance selection | 17.06 | 34.65 | 31.99 | 12.96 |
| MMR | 16.31 | 32.59 | 31.72 | 14.21 |
| Similarity coverage | 12.90 | 20.72 | 19.07 | 8.77 |
| Mean label selector | 15.01 | 28.95 | 27.67 | 11.30 |
| Conservative label selector | 15.01 | 28.95 | 27.67 | 11.30 |
| Incoming passage control | 15.01 | 28.95 | 27.67 | 11.30 |

All scores retain the 100 question denominator. MMR has three invalid outputs, scored as empty. Combined utility is averaged after computing each response's harmonic mean; it cannot be reconstructed by combining the aggregate columns in this table.

The 32 question stability pilot had ten informative comparisons across readers, with agreement on five. One disagreement was an opposite preference; the others involved a tie for one reader. Using the lower of the two utility changes did not improve agreement. The replay reconstructs these counts from the saved repeated measurements.

The experiment established no learned improvement on reader B or on A. Since the policies already retained the starting evidence during source calibration, the outcome cannot be attributed solely to differences between readers. This branch was not adopted as the final method; the main method uses HotpotQA supporting fact annotations for its labels.

## Code and checks

`model.py` retains the original feature transformation, ridge fitting, targets and threshold logic. `analysis.py` retains label construction, proposal generation, stability analysis and statistical comparisons. `evaluate.py` joins these functions to the portable inputs. Saved measurements and original outputs are under `saved/`, with checksums and source freeze metadata. `reproduction.json` records the completed CPU replay.

```bash
python3 -m unittest discover -s experiments/alce_transfer/tests -v
```

The tests cover label targets, reader separation, per question fitting weights, utility calculation, keeping the starting set and paired comparisons. They require no GPU or benchmark download.
