# AI Usage Log — Homework Three

> Fill in / adjust entries below to match the exact format you used in Homework One/Two.
> This is a starting template based on the work done in this session — edit dates, tool
> version, and wording to reflect your actual process and your course's required format.

| Date | Tool Used | Purpose | Prompt Summary | How Output Was Used |
|---|---|---|---|---|
| 2026-07-19 | Claude (Anthropic) | Data pipeline | Requested a stratified train/val/test split, resize/normalize transforms, and train-only augmentation for a fish image dataset in PyTorch | Reviewed and used as `src/data_pipeline.py`, verified split proportions and pixel ranges by running the script |
| 2026-07-19 | Claude (Anthropic) | Baseline CNN architecture | Requested a CNN with 3 conv blocks (32/64/128 filters), max pooling, and a dense classifier head in PyTorch | Reviewed architecture, confirmed output shape and parameter count, used as `src/model.py` |
| 2026-07-19 | Claude (Anthropic) | Baseline training loop | Requested a training/validation loop with Adam optimizer, loss/accuracy tracking, and plotting | Ran training for 10 epochs, reviewed resulting curves for correctness (no divergence, reasonable convergence) |
| 2026-07-19 | Claude (Anthropic) | Hyperparameter search | Requested a random search script over learning rate, batch size, dropout, and weight decay with results saved to JSON | Ran 6 trials, manually reviewed and ranked results by validation loss to select best config |
| 2026-07-19 | Claude (Anthropic) | Evaluation & analysis | Requested classification report generation (precision/recall/F1) and a combined comparison visualization (loss/accuracy curves + confusion matrix) | Ran evaluation on test set, reviewed metrics, used the written qualitative analysis as a starting point and edited to reflect actual results |

## Disclosure Statement
AI assistance (Claude) was used to help design the data pipeline, CNN architecture, training/evaluation scripts, and to draft an initial qualitative analysis of results. All code was reviewed, executed, and verified by [YOUR NAME] before inclusion in this repository. All reported metrics come from actual runs of this code on the provided dataset, not fabricated or estimated values.
