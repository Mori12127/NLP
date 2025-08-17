## 0.1.0 - Initial structured pipeline
- Standardized src/ layout with package `nlp_lab`
- Added dataset loaders and unification with stratified splits
- Implemented preprocessing utilities
- Baselines (TF-IDF + LogReg/LinearSVC/MNB) and DistilBERT trainer
- Training and evaluation modules with Typer CLI
- Tests: dataio, preprocessing, models, CLI
- Tooling: pyproject with ruff/black/isort/mypy/pytest; Makefile; pre-commit
- README and .env.example