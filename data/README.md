# Data and prepared inputs

The repository stores question splits, source information and checksums. The larger prepared inputs are separate data release assets. They preserve the original passage text, retrieval features and tokenizer files, so the selector can be reproduced without downloading the full retrieval index or regenerating embeddings.

| Group | Contents | Restored location |
| :--- | :--- | :--- |
| `hotpot_train` | Original passage pools, retrieval features, split record and training and validation support annotations | `data/prepared/hotpot_train/` |
| `hotpot_eval` | Final 128 question passage pools, retrieval features, references and saved evidence selections | `data/prepared/hotpot_eval/` |
| `tokenizers` | The four native tokenizer JSON files used for evidence budgets and additional reader preparation | `data/tokenizers/` |
| `alce_transfer` | Prepared QAMPARI questions, replacement features, references and the original input freeze | `data/prepared/alce_transfer/` |

`manifests/prepared.json` records every archive and uncompressed file's size and SHA256 checksum. `manifests/sources.json` records the public sources and pinned versions. The question identifiers are browsable under `splits/`.

## Restore from downloaded archives

With the release archives in a local directory:

```bash
python3 data/prepare.py restore --archives-dir /path/to/release_assets
python3 data/prepare.py verify
```

The default groups are `hotpot_train`, `hotpot_eval` and `tokenizers`. To restore the QAMPARI experiment as well:

```bash
python3 data/prepare.py restore \
  --archives-dir /path/to/release_assets \
  --groups alce_transfer tokenizers
```

The restore program uses the Python standard library. It checks the archive before unpacking, permits only the recorded regular files, and checks every restored file. Existing valid groups are reused; existing changed inputs are reported rather than overwritten. Allow about 2 GiB of free disk space for the release archives and restored inputs together.

## Download from a published release

After the data assets have been attached to a GitHub Release, use its asset base URL:

```bash
python3 data/prepare.py restore \
  --base-url https://github.com/Pallav-cseIITD/Evidence_Utility_Reranking/releases/download/stage3-data-v1
```

The URL above requires a published release with the four recorded archive filenames. The source code and manifests do not publish those assets. Downloading the original full corpus and index is unnecessary for this cached input route.

## Reproduce the selector

With `requirements/analysis.txt` installed:

```bash
python3 -m method.check_reproduction \
  data/prepared/hotpot_eval/pools.json \
  data/prepared/hotpot_eval/features.json \
  data/prepared/hotpot_eval/selections.json \
  --tokenizers data/tokenizers/qwen25.json data/tokenizers/qwen3.json
```

This checks all saved selection methods on the final 128 questions, including selected passage order and both token counts. It makes no reader calls.

To reproduce training:

```bash
python3 -m method.train \
  data/prepared/hotpot_train \
  outputs/repair_model \
  --manifest configs/hotpot_repair.json \
  --tokenizers data/tokenizers/qwen25.json data/tokenizers/qwen3.json
```

The evaluation references are restored separately from the training annotations. Training uses only the 512 training and 128 validation reference records. No test reference file is included in the training group.

## Research splits

These are research subsets of the public development or released evaluation collections, not training on the complete official datasets.

`splits/hotpot.json` lists 512 training, 128 validation and 128 final evaluation identifiers. It also preserves the earlier scaling collection's separate 128 test identifiers, which were not the final evaluation collection. The final evaluation questions are disjoint from all 768 earlier collection questions by identifier and normalized question text. The original training pool and feature files retain those 768 entries to preserve their recorded hashes; the trainer consumes the first 640 entries only.

`splits/alce_qampari.json` lists the 128 selected training, 40 calibration and 100 evaluation identifiers. The original experiment retained 502 valid action labels from 126 training questions after rejecting invalid source outputs. The recorded selected split remains 128 questions; validity filtering happens during label construction.

Earlier pilot inputs and outputs accompany their historical experiments. The release assets here cover the final HotpotQA method and the ALCE transfer experiment. Reader weights, full retrieval indexes, private review notes and machine configuration are not included.

## Verification

```bash
python3 -m unittest discover -s data/tests -v
```

These tests cover successful restoration, changed inputs, bad checksums, duplicate or unexpected archive members, links, invalid paths and download handling. Download tests use local fixtures and do not contact a server.

`manifests/verification.json` records the separate check on the actual four release archives: all 19 files were restored and verified, the original training input hashes matched, and all 128 saved evidence selections were reproduced. Publishing and downloading the GitHub Release is a separate step; this local check does not claim that the remote assets are already available.

See [sources and attribution](SOURCES.md) for benchmark and tokenizer provenance.
