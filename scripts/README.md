# Run the selector experiments

These commands use the existing method and data preparation code. They run on CPU and do not download reader weights, submit cluster jobs or generate new answers. Activate the environment installed from `requirements/analysis.txt` first.

Paths passed to `run.py` are relative to the repository root, even when the script is launched from another directory. Absolute paths are also accepted. Run the examples below from the repository root.

## 1. Check the software

```bash
python3 scripts/run.py check
```

This runs the method, data restoration and command wrapper tests. It needs no benchmark downloads. A failed check returns a nonzero exit status.

## 2. Restore the prepared inputs

With the four release archives downloaded:

```bash
python3 scripts/run.py restore --archives-dir /path/to/release_assets
```

After the data release has been published, the equivalent download command is:

```bash
python3 scripts/run.py restore \
  --base-url https://github.com/Pallav-cseIITD/Evidence_Utility_Reranking/releases/download/stage3-data-v1
```

The default groups are HotpotQA training, HotpotQA evaluation and tokenizers. Restore QAMPARI separately with `--groups alce_transfer tokenizers`. To check the restored default groups again:

```bash
python3 scripts/run.py verify-data
```

## 3. Reproduce the saved evidence selections

```bash
python3 scripts/run.py reproduce
```

This verifies the evaluation inputs, runs the saved models and compares all 128 questions with the original selection file. It writes `outputs/selection_reproduction.json`. Success means that question order, method names, passage indices and both token counts match. It does not recalculate answer accuracy or citation quality.

## 4. Retrain and check the new model

```bash
python3 scripts/run.py train
python3 scripts/run.py reproduce \
  --model outputs/repair_model \
  --output outputs/retrained_selection_reproduction.json
```

Training uses `configs/hotpot_repair.json`, the 512 training questions and 128 validation questions. It writes both fitted models, training histories, validation sweeps, selected thresholds and input hashes under `outputs/repair_model/`. No evaluation references enter training. The original CPU run took about four minutes; hardware and file access affect runtime.

## 5. Write the selected evidence

```bash
python3 scripts/run.py select
```

This writes `outputs/hotpot_selections.json`, containing all seven selection methods and their token counts. To select with a retrained model, add `--model outputs/repair_model` and use a new `--output` filename.

Every output path must be new. Repeated runs can use names such as `--output outputs/repeat_02.json`; the commands do not overwrite existing results or change the saved original models. `--data-dir` selects a different location for the restored data. `train --config` selects a different training manifest, which constitutes a different experiment.

The reader evaluation entry points and statistical analysis of saved reader responses accompany the evaluation code. These commands cover software checks, data restoration, selector training and evidence selection.

Keep `method/fitted/.gitattributes` with the saved models. It prevents Git from changing the line endings of their original metadata files, which are checked by byte level hashes.
