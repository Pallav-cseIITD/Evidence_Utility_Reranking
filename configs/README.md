# Experiment configurations

| File | Experiment | Use |
| :--- | :--- | :--- |
| `hotpot_repair.json` | Final HotpotQA selector training | Passed directly to `method.train` with `--manifest` |
| `hotpot_primary_reader.json` | Main HotpotQA answer evaluation | Recorded reader, decoding, qualification and statistical settings |
| `hotpot_additional_readers.json` | Additional readers and evidence arrangements | Recorded settings for the later comparison on the same 128 questions |
| `alce_transfer.json` | QAMPARI labels from reader A and evaluation with reader B | Recorded transfer design, label targets and calibration settings |

## Train with the final configuration

From the repository root, with the CPU environment active and prepared inputs available:

```bash
python3 -m method.train \
  data/prepared/hotpot_train \
  outputs/repair_model \
  --manifest configs/hotpot_repair.json \
  --tokenizers data/tokenizers/qwen25.json data/tokenizers/qwen3.json
```

The training configuration matches the default manifest bundled in `method/`. The explicit `--manifest` argument selects this central copy. The prepared files and tokenizer hashes must match the configuration before fitting starts. The output directory must be new.

The trained model directory records its own selected thresholds. Inference loads them from that directory's `policy.json`; it does not take thresholds from a reader profile. Keep the saved original models unchanged when training a different configuration.

## Reader profiles

The three reader profiles document the executed experiments. Their collection and scoring code will accompany the evaluation entry points. They are not input files for `method.select`, and passing a profile alone does not run a reader. The archived experiment protocols and output records remain the source for replaying an original run's identity.

The primary HotpotQA reader uses reduced memory model loading. The additional readers use 16 bit weights. Their generation seeds differ, so these are separate profiles. The additional reader profile also preserves the distinction between failing the original qualification checks, which stops evaluation, and failing the expanded checks, which limits the later results to diagnostic use.

The QAMPARI experiment uses the official ALCE GTR passage release. It is a separate retrieval setting from HotpotQA's sparse and dense fusion. It fits utility predictors from reader A's repeated measurements and calibrates them using A alone. Reader B's stability outputs are diagnostic and never fitting labels. Evaluation uses the unchanged selected evidence with B.

Within a configuration, `seed` controls a specific operation. The selector training seed, reader generation seed and statistical resampling seed are separate values. `null` in a threshold grid means keep the starting evidence for every question. With greedy generation, the recorded temperature and probability settings are inactive.

Dataset manifests and split identifiers belong in `data/`; fitted models and selected thresholds are in `method/fitted/`. Reader weights, downloaded passages, caches and private machine settings are not included here. Original experiment settings are preserved; changing a setting creates a new run rather than reproducing the old one.
