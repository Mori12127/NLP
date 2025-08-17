from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..models.baselines import (
	build_baseline_pipeline,
	eval_baseline_pipeline,
	save_baseline,
	train_baseline_pipeline,
)
from ..models.transformer import train_transformer_model
from ..utils.config import Config
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def _load_splits_from_dir(split_prefix: str, processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	train_path = next(processed_dir.glob(f"{split_prefix}_train.parquet"))
	val_path = next(processed_dir.glob(f"{split_prefix}_val.parquet"))
	test_path = next(processed_dir.glob(f"{split_prefix}_test.parquet"))
	train_df = pd.read_parquet(train_path)
	val_df = pd.read_parquet(val_path)
	test_df = pd.read_parquet(test_path)
	return train_df, val_df, test_df


def _ensure_train_has_samples(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if len(train_df) == 0 and not val_df.empty:
		row = val_df.iloc[[0]]
		train_df = pd.concat([train_df, row]).reset_index(drop=True)
		val_df = val_df.iloc[1:].reset_index(drop=True)
	if len(train_df) == 0 and not test_df.empty:
		row = test_df.iloc[[0]]
		train_df = pd.concat([train_df, row]).reset_index(drop=True)
		test_df = test_df.iloc[1:].reset_index(drop=True)
	# Ensure at least two classes if possible
	if train_df["label"].nunique() < 2:
		all_labels = set(pd.concat([train_df["label"], val_df["label"], test_df["label"]], ignore_index=True).dropna().astype(str).unique())
		missing = list(all_labels - set(train_df["label"].dropna().astype(str).unique()))
		if missing:
			# try to add one sample with a missing label from val or test
			cand = val_df[val_df["label"].astype(str) == missing[0]]
			if not cand.empty:
				row = cand.iloc[[0]]
				train_df = pd.concat([train_df, row]).reset_index(drop=True)
				val_df = val_df.drop(index=row.index).reset_index(drop=True)
			else:
				cand = test_df[test_df["label"].astype(str) == missing[0]]
				if not cand.empty:
					row = cand.iloc[[0]]
					train_df = pd.concat([train_df, row]).reset_index(drop=True)
					test_df = test_df.drop(index=row.index).reset_index(drop=True)
	return train_df, val_df, test_df


def train_baseline(config: Config) -> Dict[str, float]:
	prefixes = [f"{config.run_id or 'run'}_{ds}" for ds in config.datasets]
	frames = []
	for pr in prefixes:
		tr, va, te = _load_splits_from_dir(pr, config.paths.processed_dir)
		frames.append((tr, va, te))
	train_df = pd.concat([f[0] for f in frames], ignore_index=True)
	val_df = pd.concat([f[1] for f in frames], ignore_index=True)
	test_df = pd.concat([f[2] for f in frames], ignore_index=True)

	train_df, val_df, test_df = _ensure_train_has_samples(train_df, val_df, test_df)

	pipe = train_baseline_pipeline(
		X_text=train_df["text"].astype(str).tolist(),
		y=train_df["label"].astype(str).to_numpy(),
		preprocess=config.preprocess,
		cfg=config.baseline,
	)
	# choose eval set fallback if val is empty
	eval_df = val_df if len(val_df) > 0 else (test_df if len(test_df) > 0 else train_df)
	metrics = eval_baseline_pipeline(
		pipe,
		X_text=eval_df["text"].astype(str).tolist(),
		y=eval_df["label"].astype(str).to_numpy(),
		preprocess=config.preprocess,
	)
	# save
	run_id = config.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
	art_dir = config.paths.models_dir / run_id / "baseline"
	art_dir.mkdir(parents=True, exist_ok=True)
	save_baseline(pipe, art_dir / "model.joblib")
	(art_dir / "metrics_val.json").write_text(json.dumps(metrics, indent=2))
	LOGGER.info("Saved baseline artifacts to %s", art_dir)
	return metrics


def train_transformer(config: Config) -> Dict[str, float]:
	prefixes = [f"{config.run_id or 'run'}_{ds}" for ds in config.datasets]
	frames = []
	for pr in prefixes:
		tr, va, te = _load_splits_from_dir(pr, config.paths.processed_dir)
		frames.append((tr, va, te))
	train_df = pd.concat([f[0] for f in frames], ignore_index=True)
	val_df = pd.concat([f[1] for f in frames], ignore_index=True)

	out_dir = config.paths.models_dir / (config.run_id or datetime.now().strftime("%Y%m%d-%H%M%S"))
	out = train_transformer_model(
		train_df["text"].astype(str).tolist(),
		train_df["label"].astype(str).tolist(),
		val_df["text"].astype(str).tolist(),
		val_df["label"].astype(str).tolist(),
		out_dir,
		config.transformer,
	)
	( out.best_model_dir / "metrics_val.json" ).write_text(json.dumps(out.metrics, indent=2))
	LOGGER.info("Saved transformer artifacts to %s", out.best_model_dir)
	return out.metrics


__all__ = ["train_baseline", "train_transformer"]