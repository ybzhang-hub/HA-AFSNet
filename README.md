# HA-AFSNet

HA-AFSNet for context-conditioned temporal event forecasting on EG, IR and IS. The repository includes the adaptive model, the learned static-gate control and the encoder-swap control.

## Layout

```text
.
├── train.py / evaluate.py        # Training and independent evaluation
├── src/ha_afsnet/
│   ├── model.py                  # Graph encoders, fusion gate and temporal decoder
│   ├── data.py / context.py      # Chronological data and article-context fitting
│   ├── task.py / trainer.py      # Causal graph inputs, objective and training loop
│   ├── evaluation.py             # Full-vocabulary filtered ranking
│   ├── checkpoint.py / utils.py  # Checkpoints, random states and CUDA measurement
│   ├── config.py                 # Configuration validation and portable locations
│   └── experiments.py / statistics.py / analysis.py / benchmark.py
├── scripts/                      # Experiment and analysis commands
├── configs/                      # Model settings, optional training profiles and protocols
├── splits/                       # Released split identifiers and input checksums
├── data/README.md                # Required external data layout
└── requirements.txt / environment.yml
```

## Install

Run commands from the repository root on Linux. Python 3.10 and PyTorch 2.4.0 with CUDA 12.1 are specified in the environment files.

```bash
conda env create -f environment.yml
conda activate ha-afsnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Alternatively, create a Python 3.10 environment and run `pip install -r requirements.txt`. HA-AFSNet does not require DGL. Use `--device cpu` for CPU execution.

## Prepare data

Place the released archives in `data/`, or point to their extracted parent directory. The expected layout is described in [data/README.md](data/README.md).

```bash
export HA_AFSNET_DATA_ROOT=/path/to/released-data
python scripts/prepare_data.py
```

Preparation checks file hashes, vocabulary, chronology, context labels and popularity exclusions against `splits/`. It writes caches, date mappings and the exact train/validation/test article MD5 lists to `cache/prepared/`. The input files remain unchanged.

## Train and evaluate

```bash
python train.py --dataset EG --seed 42
python train.py --model HA-static --dataset EG --seed 42
python train.py --model HA-swap --dataset EG --seed 42

# Restore model, Adam, random states, epoch progress and checkpoint selection.
python train.py --dataset EG --seed 42 --resume

python evaluate.py --dataset EG --seed 42 \
  --checkpoint outputs/runs/HA-AFSNet/EG/seed_42/best.pt
```

Every epoch uses the complete validation split for checkpoint selection. The selected checkpoint is evaluated on the complete test split. Each run saves `config.json`, `protocol.json`, `epochs.json`, `result.json`, `best.pt`, `last.pt`, `test_ranks.npz` and `gates.csv` under `outputs/runs/<model>/<region>/seed_<seed>/`.

`epochs.json` also records learning rate, optimizer steps, patience, pre-clipping gradient norms and the fraction of steps above the clipping threshold. Every 100 steps it samples positive/negative BCE, L2, logits, gate saturation and module gradient norms. These detached signals do not change the loss or random sequence. Sampled signals are unweighted means over sampled batches, and their sample count is retained; the positive-versus-negative fraction is a training diagnostic rather than a ranking metric. Use `--diagnostics-every-steps 10` for denser logging or `--diagnostics-every-steps 0` to disable it. Resume with the original diagnostics interval.

Negative tails are drawn on CPU using the original generator and draw order before transferring queries and negatives to the accelerator. Answer dictionaries use a bounded date/context cache. Only parameter-independent inputs are cached; trainable graph encodings are recomputed after optimizer updates.

For a short execution check, use `python train.py --dataset EG --seed 42 --quick-run`. This uses three training dates and two validation/test dates; its outputs are separate and excluded from experiment statistics.

## Repeated experiments

```bash
# Three models × three regions × seeds 42–46; four independent GPU processes.
python scripts/run_experiments.py --workers 4

# Restrict the queue if needed.
python scripts/run_experiments.py --workers 4 --models HA-AFSNet --regions EG

python scripts/summarize.py
python scripts/analyze_gates.py
```

The queue skips compatible completed runs and resumes interrupted runs. `outputs/tables/` contains per-seed MRR, Hits@1/3/10, best epoch, validation MRR, configurations and checkpoint references. Paired differences are computed only after all five matched seeds complete. `outputs/figures/` contains gate time series; `gates.csv` stores the model's actual gate values and causal activity/composition inputs.

Use the same `--output-root` for training, the experiment queue, summaries, gate analysis and profiling when storing outputs elsewhere. Paths in summary tables are relative to that output root.

## Validation-only configuration selection

The default model configuration remains the reference setting. Three optional profiles change one setting at a time: `lr_3e4.json` lowers Adam learning rate to 0.0003; `transformer_independent.json` independently initializes Transformer layers; `relation_xavier.json` applies Xavier initialization to each relation matrix individually. They keep the architecture, parameter count, objective, negatives and evaluation protocol. Initialization profiles preserve the random stream and other modules' initial weights. Their forecasting performance must be measured separately.

Use a separate output root and `--validation-only` when comparing configurations. This trains and selects the checkpoint using validation, then stops without evaluating test. Its `completed_validation` results are excluded from formal test summaries.

```bash
python scripts/run_experiments.py --workers 3 --models HA-AFSNet --regions EG \
  --seeds 42 43 44 --config configs/ha_afsnet.json --validation-only \
  --output-root outputs/selection/reference

# Change --config and --output-root for each candidate; run campaigns sequentially.
python scripts/run_experiments.py --workers 3 --models HA-AFSNet --regions EG \
  --seeds 42 43 44 --config configs/lr_3e4.json --validation-only \
  --output-root outputs/selection/lr_3e4
```

Compare `validation_MRR` in each campaign's `runs/HA-AFSNet/EG/seed_*/result.json`. Freeze the selected configuration before running full seeds 42–46 in a fresh output root without `--validation-only`. Report the configuration-selection budget with any resulting comparisons. A custom configuration in the queue requires exactly one model; static and swap controls need their corresponding configurations.

## Optional article-context fitting

The main experiment uses released LDA K5 event labels. To fit separate article-context models using training articles only:

```bash
python -m nltk.downloader stopwords wordnet punkt_tab
python scripts/fit_contexts.py --methods lda doc2vec kmeans gmm
```

All fitting settings are in `configs/contexts.json`. Heldout articles use frozen inference or assignment. Fitted parameter hashes are checked before and after assignment. Outputs are written to `outputs/contexts/` without changing the main experiment labels.

## Resource measurement

After all other GPU jobs finish:

```bash
python scripts/profile_resources.py \
  --checkpoint outputs/runs/HA-AFSNet/EG/seed_42/best.pt
python scripts/summarize.py
```

This measures a complete representative training epoch from the validation-selected checkpoint and three complete test evaluations. Training warm-up performs one forward/backward and restores model/random states; inference warm-up covers three dates. CUDA is synchronized before and after each timed region. Allocated and reserved peak PyTorch allocator memory are reset and recorded separately, including resident weights and tensor graph caches, excluding CUDA context and non-PyTorch allocations. Summaries use arithmetic mean, sample standard deviation, range and measurement count.

Training timing includes causal graph construction/transfer, encoding, scoring, backward, clipping and Adam. It excludes initial data/statistic preparation, article fitting, validation, checkpoint and logging I/O. Full-test timing includes graph construction, encoding, full-vocabulary scoring, popularity selection and filtering, excluding initial preparation and checkpoint/rank-file I/O. This is a representative post-selection epoch, not the mean of the original optimization trajectory. The serial benchmark excludes training diagnostics and rejects other active CUDA processes. Concurrent timings go to `resources_observed_training.csv`; serial measurements go to `resources.csv`. Compare models on the same machine and with the stated widths.

## Evaluation and statistics

| Split | Timestamp IDs | Dates |
| --- | --- | --- |
| Train | 0–2067 | 2015-02-19–2020-10-17 |
| Validation | 2068–2325 | 2020-10-18–2021-07-02 |
| Test | 2326–2583 | 2021-07-03–2022-03-17 |

The reference model uses width 128, five contexts and five history snapshots, two graph layers, a two-layer four-head entity Transformer and a relation GRU. Fusion weights the context-specific state by alpha and the shared state by 1−alpha. For historical snapshot τ, the gate uses information available to forecast τ+1. Training begins at timestamp 10. Rolling evaluation uses strictly earlier observed facts, including earlier validation/test dates.

Adam uses learning rate 0.001, batch size 128, dropout 0.2, explicit L2 coefficient 1e-5 and gradient clipping at 1. The loss sums positive and one sampled-negative softplus term per query plus explicit L2. Training runs for at most 50 epochs with patience 5; equal validation MRR updates the checkpoint and resets patience. Test does not select checkpoints. Training diagnostics use detached values, preserve the random sequence and resume counters, and report sampled-batch means with sample counts.

Popularity counts subject and object occurrences in training fact rows; self-edges count twice. Sort by descending frequency, retaining first-occurrence order for ties. Add an entity while the previous cumulative count is at most floor(total occurrences / 3), so the final total can exceed the threshold. Popular entities are excluded as evaluation targets only and remain in the complete candidate vocabulary, training examples and histories.

| Region | Candidate entities | Threshold | Selected occurrences | Excluded target IDs |
| --- | ---: | ---: | ---: | --- |
| EG | 2,594 | 251,620 | 268,725 | 0, 2, 9 |
| IR | 2,988 | 649,168 | 672,377 | 1, 0, 16 |
| IS | 3,465 | 953,592 | 1,120,026 | 5, 3, 15 |

Evaluation predicts tails over the full released entity vocabulary. Filtering removes other true tails sharing (subject, relation, timestamp), merging contexts and retaining the query target. Average rank is `1 + strictly_higher + 0.5 * (equal_including_target - 1)`; Hits@K uses rank ≤ K, including fractional ranks. Training samples uniformly from the full vocabulary while excluding training true tails of the same (subject, relation, context, timestamp); validation/test answers are not consulted.

Main-experiment contexts are labels from articles linked to the target event, using argmax of their mean LDA topic posterior. They are distinct from an analyst-provided scenario available before an event. Optional LDA/doc2vec/K-means/GMM fitting uses training articles only; heldout articles use frozen inference/assignment. K-means and GMM share training-fitted doc2vec vectors. Exact fitting IDs, assignments and before/after fitted-parameter hashes are exported. Fitting seed 2023 is independent of forecasting seeds 42–46; fitting sets contain 82,698 EG, 196,785 IR and 284,808 IS articles.

Summaries report arithmetic mean, sample standard deviation and range across complete runs. For five matched seeds, paired differences d have 95% interval `mean(d) ± t(0.975,4) * sample_sd(d) / sqrt(5)`. This assumes independent seed pairs and approximately normal differences conditional on the fixed dataset, split and protocol. Exact sign-flip calculations enumerate 32 sign assignments under a symmetry/exchangeability assumption; five nonzero pairs give a minimum two-sided p-value of 0.0625. Missing or incompatible pairs are withheld. Five runs have limited statistical power.

## Gate analysis

Gate files contain seed, date, timestamp, context, alpha, activity, activity change and composition change for forecast timestamps 6–2583. Forecast t uses t−1 and earlier observations. Gate activity is `log(1 + unique incident entities + unique directed relational edges)`, with edges deduplicated across contexts. Composition change is KL divergence against the previous EMA, with coefficient 0.9 and epsilon 1e-8.

Descriptive activity groups use fact rows instead of deduplicated edges. Thresholds are the 25th and 75th percentiles across all 2,068 training dates: low ≤ q25, high ≥ q75. Group means are calculated per seed before aggregation; thresholds are shared by all seeds. Time series cover all 258 test dates and show the five-seed mean and sample standard deviation by context. Group associations are descriptive because activity and composition are gate inputs.
