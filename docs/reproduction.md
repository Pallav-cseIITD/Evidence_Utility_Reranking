# Reproduction

Use Python 3.11 or 3.12 for the CPU route below. Commands are run from the repository root in Bash on Linux or WSL.

```bash
python3 -m venv .venv-analysis
source .venv-analysis/bin/activate
python3 -m pip install -r requirements/analysis.txt
python3 scripts/run.py check
```

Restore the data archives from a local directory:

```bash
python3 scripts/run.py restore --archives-dir /path/to/release_assets
python3 scripts/run.py restore --archives-dir /path/to/release_assets --groups alce_transfer
```

Published Release downloads are an alternative documented in `data/README.md`. The release assets must be attached to the repository's data release first. The archives contain about 687 MB of uncompressed inputs and about 212 MB of compressed data; allow 2 GiB free disk space. No reader weights or full retrieval index are needed for the CPU route.

## Saved selector and retraining

```bash
python3 scripts/run.py reproduce
python3 scripts/run.py train
python3 scripts/run.py reproduce --model outputs/repair_model --output outputs/retrained_check.json
```

The original selector reproduction checks all 128 evaluation questions. Training uses the exact prepared files, tokenizer hashes, 512 training questions and 128 validation questions. Full CPU retraining during repository preparation reproduced both saved models' parameter arrays exactly. Its measured time was about four minutes on that machine.

## Recalculate the reported outcomes

```bash
python3 -m experiments.hotpot.evaluate
python3 -m experiments.alce_transfer.evaluate
python3 results/export.py outputs/hotpot_evaluation/results.json \
  outputs/alce_transfer_evaluation/result.json --output outputs/result_tables
```

The HotpotQA command regrades 2,944 logical answer records and recalculates evidence metrics and paired comparisons. These records include cached generation reuse. The QAMPARI command reconstructs 502 valid labels, refits the three utility predictors and repeats calibration, selection, stability analysis and statistical comparisons from cached official measurements.

Run the experiment specific tests with:

```bash
python3 -m unittest discover -s experiments/hotpot/tests -v
python3 -m unittest discover -s experiments/alce_transfer/tests -v
```

Every output path must be new; use another output name when repeating a command. Data restoration reuses valid groups and rejects changed inputs. Saved model metadata must retain its original bytes, protected by `method/fitted/.gitattributes`.

The CPU replay is distinct from generating reader outputs again. The reader settings are in `configs/`, and the measured GPU environment is in `requirements/recorded_environment.json`. GPU collection code and cluster launchers are separate components; the commands above reproduce the available saved experiments.

The earlier branch history is documented in `research_history.md`. Its table summarizes the completed investigation; earlier branch code has not been included in the current upload batch. Benchmark splits are public development or released evaluation subsets, rather than full official training runs.
