# Marginal support repair

The selector starts with fusion's five passages and considers replacing one passage. It predicts the support gained and lost by the exchange, taking account of the incoming passage, the outgoing passage and the retained evidence. It makes the highest scoring feasible replacement only when its predicted gain exceeds the validation threshold. Otherwise, it keeps the starting set.

Each replacement preserves five unique passages and cannot increase the starting evidence token count under either of the two native tokenizers. The starting set must fit the 1,536 token evidence cap. Candidate construction considers up to 80 replacements per question.

## Contents

| File or folder | Purpose |
| :--- | :--- |
| `train.py` | Construct labels, fit both selectors and choose their thresholds |
| `select.py` | Select passages for one question or a prepared collection |
| `policy.py` | Candidate construction, prediction and the keep or replace decision |
| `features.py` | The 43 query, passage and evidence set features |
| `support.py` | Benchmark supporting fact matching and support change labels |
| `model.py` | NumPy neural model, fitting and model save and load |
| `edits.py`, `sampling.py`, `io.py`, `inputs.py` | Token budgets, deterministic sampling, input parsing and verification |
| `training_inputs.json` | Input hashes, tokenizer hashes and the training settings |
| `fitted/` | The original trained repair model, incoming passage control and policy metadata |
| `check_reproduction.py` | Compare selected passages and token counts with a saved selection file |
| `tests/` | Small software checks that require no benchmark download |

The package imports no code from the historical experiment folders.

## Install and check

Run commands from the repository root with the CPU environment active:

```bash
python3 -m pip install -r requirements/analysis.txt
python3 -m unittest discover -s method/tests -v
```

The tests include training and selection on small constructed inputs, support gained versus lost, partial supporting facts, strict thresholds, duplicate candidates, token budget constraints, changed input rejection and model persistence. These software checks are separate from benchmark evaluation.

## Run the saved selector

The following paths describe the prepared asset layout. Passage pools, retrieval features and tokenizer files are external inputs, not included in this method folder. Their manifests and preparation instructions accompany the data release.

```bash
python3 -m method.select \
  data/prepared/hotpot_eval/pools.json \
  data/prepared/hotpot_eval/features.json \
  --batch \
  --tokenizers data/tokenizers/qwen25.json data/tokenizers/qwen3.json \
  --output outputs/hotpot_selections.json
```

The default model directory is `method/fitted`. To use a retrained model, supply `--model outputs/repair_model`.

For one question, omit `--batch` and supply one question object and its corresponding feature object. The batch format is a JSON array of question objects and a JSON object mapping question identifiers to features, with matching order. These are the original saved formats, using two space indentation. Passage indices in the output start at zero.

Each result contains the question identifier and selected passages for fusion, repair, the incoming passage control, relevance selection, MMR, similarity coverage and the title link heuristic. It also contains token counts under both native tokenizers. This command selects evidence; it does not generate answers.

The selector reads no reference answers or support annotations. The two tokenizers must match the stored file hashes and be supplied in Qwen2.5, Qwen3 order.

## Retrain

```bash
python3 -m method.train \
  data/prepared/hotpot_train \
  outputs/repair_model \
  --tokenizers data/tokenizers/qwen25.json data/tokenizers/qwen3.json
```

The prepared training directory must contain:

```text
cohort_freeze.json
pools.json
features.json
train_references.json
validation_references.json
```

`training_inputs.json` fixes their hashes to the recorded collection. The training command reads support annotations for the 512 training and 128 validation questions. It never opens test reference annotations. The original pool and feature files also contain the later evaluation entries, which are not used for fitting or threshold selection.

For each candidate exchange, the label counts supporting document groups completed and lost. A group is complete when all its annotated supporting sentences are present in the selected passages. The scalar training target is the number gained minus the number lost. This is a benchmark evidence support target; generated answer scores are evaluated separately.

Training samples up to four clean gains, four harmful replacements and four neutral replacements per question. The model uses 43 inputs, 32 hidden units and one output, with 1,441 trainable parameters. It is fitted for 60 epochs using seed 17; the state with the lowest validation loss is retained. The incoming passage control uses the same architecture and labels but masks all features except the first 11 incoming passage features.

Validation chooses from the recorded threshold grid by net complete chain gains, breaking ties in favor of fewer edits. The saved repair threshold is 0.15; the control threshold is 0.30. The saved training has 33,673 labeled training actions, 4,159 sampled training rows and 8,392 validation actions. Training needs no GPU or reader API. The original run took about four minutes on its local CPU; runtime depends on hardware and prepared input access.

The command requires a new output directory and writes both models, their training histories and `policy.json`. The policy records input hashes, selected thresholds, validation sweeps, code hashes and environment versions. To run a different prepared training collection, supply `--manifest path/to/manifest.json` with its corresponding inputs and settings; that constitutes a different experiment.

## Check saved selections

With the prepared evaluation inputs and the saved original selection file available:

```bash
python3 -m method.check_reproduction \
  data/prepared/hotpot_eval/pools.json \
  data/prepared/hotpot_eval/features.json \
  results/hotpot/selections.json \
  --tokenizers data/tokenizers/qwen25.json data/tokenizers/qwen3.json
```

The check compares question order, every method's passage indices and both token counts. It exits with a nonzero status if any result differs. It makes no reader calls and therefore does not regenerate answer quality metrics.

The original saved policy's `source` hash identifies the experiment training script that produced it. New training records identify the reorganized package. The saved model files and original metadata are preserved byte for byte.

## Reproduction records

`selection_reproduction.json` records the completed comparison against all 128 original evaluation selections. Every method's passage indices and both token counts matched. `training_reproduction.json` records the separate full CPU retraining comparison, including model array differences and the tolerance used. These checks reuse the original prepared collections and do not represent new benchmark samples or new reader runs.
