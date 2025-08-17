# NLP Lab

Reproducible NLP pipeline: dataset unification → preprocessing → baselines and transformers → evaluation and artifacts.

## Structure

```
repo/
  src/nlp_lab/
    dataio/ preprocessing/ models/ training/ utils/
  data/{raw,interim,processed}/
  models/
  tests/
  notebooks/
  scripts/
```

## Setup

```
make venv
make install
pre-commit install
```

## Datasets
Place raw csv files under `data/raw/` or pass paths to CLI. Supported names:
- Corona_NLP_test.csv → corona
- IMDb_Reviews.csv → imdb
- Indigo-Tweets.csv → indigo
- opinion_dataset.csv → opinion

## Commands
- Prep splits (stratified):
```
PYTHONPATH=src python -m nlp_lab.training.cli prep --corona data/raw/Corona_NLP_test.csv --imdb data/raw/IMDb_Reviews.csv --run-id run
```
- Train:
```
PYTHONPATH=src python -m nlp_lab.training.cli train --model baseline --run-id run
```
- Eval:
```
PYTHONPATH=src python -m nlp_lab.training.cli eval --run models/run
```

## Makefile
```
make format
make lint
make typecheck
make test
make cov
```

## Troubleshooting
- Install CPU `torch` if CUDA unavailable.
- Ensure parquet support via `pandas` (pyarrow is vendored in pandas>=2). If needed: `pip install pyarrow`.
- Set env via `.env` or `.env.local`.