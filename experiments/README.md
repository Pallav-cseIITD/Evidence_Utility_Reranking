# Experiments

| Folder | Purpose | Starting point |
| :--- | :--- | :--- |
| [hotpot](hotpot/README.md) | Recalculate the final HotpotQA evidence and answer results from saved responses | `python3 -m experiments.hotpot.evaluate` |
| [alce_transfer](alce_transfer/README.md) | Refit source reader utility predictors and reproduce the QAMPARI transfer evaluation using cached measurements | `python3 -m experiments.alce_transfer.evaluate` |

The final selector's training and inference implementation is in `method/`. Experiment folders contain their evaluation code, frozen inputs or links to prepared data, saved results and reproduction instructions.
