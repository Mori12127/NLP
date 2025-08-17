from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ..utils.config import BaselineConfig, PreprocessConfig
from ..preprocessing.text import preprocess_corpus


ClassifierName = Literal["logreg", "linearsvc", "mnb"]


def build_baseline_pipeline(cfg: BaselineConfig, min_df_override: Optional[int] = None) -> Pipeline:
	min_df_value = min_df_override if min_df_override is not None else cfg.min_df
	vectorizer = TfidfVectorizer(
		max_features=cfg.max_features,
		ngram_range=cfg.ngram_range,
		min_df=min_df_value,
	)
	if cfg.classifier == "logreg":
		clf = LogisticRegression(max_iter=200, n_jobs=None, C=cfg.C)
	elif cfg.classifier == "linearsvc":
		clf = LinearSVC()
	elif cfg.classifier == "mnb":
		clf = MultinomialNB()
	else:
		raise ValueError(f"Unknown classifier: {cfg.classifier}")
	return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train_baseline_pipeline(
	X_text: list[str], y: np.ndarray, preprocess: PreprocessConfig, cfg: BaselineConfig
) -> Pipeline:
	X_norm = preprocess_corpus(X_text, preprocess)
	# Adjust min_df for tiny datasets
	effective_min_df = cfg.min_df
	if len(X_norm) < 10 and cfg.min_df > 1:
		effective_min_df = 1
	pipe = build_baseline_pipeline(cfg, min_df_override=effective_min_df)
	pipe.fit(X_norm, y)
	return pipe


def eval_baseline_pipeline(pipe: Pipeline, X_text: list[str], y: np.ndarray, preprocess: PreprocessConfig) -> Dict[str, float]:
	X_norm = preprocess_corpus(X_text, preprocess)
	pred = pipe.predict(X_norm)
	acc = float(accuracy_score(y, pred))
	f1 = float(f1_score(y, pred, average="macro"))
	return {"accuracy": acc, "f1_macro": f1}


def save_baseline(pipe: Pipeline, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(pipe, path)


def load_baseline(path: Path) -> Pipeline:
	return joblib.load(path)


__all__ = [
	"build_baseline_pipeline",
	"train_baseline_pipeline",
	"eval_baseline_pipeline",
	"save_baseline",
	"load_baseline",
]