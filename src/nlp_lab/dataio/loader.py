from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from langdetect import DetectorFactory, detect  # type: ignore
from sklearn.model_selection import train_test_split

from ..utils.config import DatasetName
from ..utils.logging import get_logger

DetectorFactory.seed = 0

LOGGER = get_logger(__name__)

COMMON_COLUMNS = ["id", "text", "label", "source", "lang"]
LABELS = ["negative", "neutral", "positive"]


@dataclass
class SplitPaths:
	train: Path
	val: Path
	test: Path


def _safe_detect_lang(text: str) -> Optional[str]:
	try:
		if not isinstance(text, str) or not text.strip():
			return None
		return detect(text)
	except Exception:
		return None


def _ensure_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
	for col in COMMON_COLUMNS:
		if col not in df.columns:
			df[col] = None
	df["source"] = source
	# enforce dtypes
	df = df[["id", "text", "label", "source", "lang"]].copy()
	df["id"] = pd.util.hash_pandas_object(df[["text", "label"]].astype(str), index=False).astype("int64")
	df["text"] = df["text"].astype("string")
	df["label"] = pd.Categorical(df["label"], categories=LABELS)
	df["source"] = df["source"].astype("string")
	df["lang"] = df["lang"].astype("string")
	return df


def _split_save(df: pd.DataFrame, out_dir: Path, run_name: str, source: str, seed: int = 42) -> SplitPaths:
	out_dir.mkdir(parents=True, exist_ok=True)
	try:
		train_df, test_df = train_test_split(
			df, test_size=0.15, random_state=seed, stratify=df["label"]
		)
		train_df, val_df = train_test_split(
			train_df, test_size=0.1765, random_state=seed, stratify=train_df["label"]
		)  # ~15% val
	except ValueError:
		# Fallback for tiny datasets: non-stratified split with common proportions
		test_size = max(1, int(round(0.2 * len(df)))) if len(df) >= 3 else 1
		train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
		val_size = max(1, int(round(0.25 * len(train_df)))) if len(train_df) >= 3 else 1 if len(train_df) > 1 else 0
		if val_size > 0:
			train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=seed, shuffle=True)
		else:
			val_df = train_df.iloc[[]].copy()
	# Ensure train has at least two classes when possible
	if df["label"].nunique() >= 2 and train_df["label"].nunique() < 2:
		needed_label = next(iter(set(df["label"].dropna().unique()) - set(train_df["label"].dropna().unique())), None)
		candidate_idx = None
		if needed_label is not None:
			cand = val_df[val_df["label"] == needed_label]
			if not cand.empty:
				candidate_idx = cand.index[0]
			else:
				cand = test_df[test_df["label"] == needed_label]
				if not cand.empty:
					candidate_idx = cand.index[0]
		# If no specific label found, just move first available from val or test
		if candidate_idx is None:
			if not val_df.empty:
				candidate_idx = val_df.index[0]
			elif not test_df.empty:
				candidate_idx = test_df.index[0]
		if candidate_idx is not None:
			row = (val_df if candidate_idx in val_df.index else test_df).loc[[candidate_idx]]
			train_df = pd.concat([train_df, row]).reset_index(drop=True)
			if candidate_idx in val_df.index:
				val_df = val_df.drop(index=candidate_idx)
			else:
				test_df = test_df.drop(index=candidate_idx)
	paths = SplitPaths(
		train=out_dir / f"{run_name}_{source}_train.parquet",
		val=out_dir / f"{run_name}_{source}_val.parquet",
		test=out_dir / f"{run_name}_{source}_test.parquet",
	)
	train_df.to_parquet(paths.train, index=False)
	val_df.to_parquet(paths.val, index=False)
	test_df.to_parquet(paths.test, index=False)
	LOGGER.info("Saved splits to %s", out_dir)
	return paths


def load_corona(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	# Expected columns: 'OriginalTweet','Sentiment'
	label_map = {
		"Extremely Negative": "negative",
		"Negative": "negative",
		"Neutral": "neutral",
		"Positive": "positive",
		"Extremely Positive": "positive",
	}
	text_col = "OriginalTweet" if "OriginalTweet" in df.columns else df.columns[0]
	label_col = "Sentiment" if "Sentiment" in df.columns else df.columns[-1]
	out = pd.DataFrame({
		"text": df[text_col].astype(str),
		"label": df[label_col].map(label_map).fillna("neutral"),
	})
	out["lang"] = out["text"].map(_safe_detect_lang)
	return _ensure_schema(out, source="corona")


def load_imdb(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	# Expected columns: 'review','sentiment' (pos/neg)
	label_map = {"positive": "positive", "pos": "positive", "negative": "negative", "neg": "negative"}
	text_col = "review" if "review" in df.columns else df.columns[0]
	label_col = "sentiment" if "sentiment" in df.columns else df.columns[-1]
	out = pd.DataFrame({
		"text": df[text_col].astype(str),
		"label": df[label_col].astype(str).str.lower().map(label_map).fillna("neutral"),
	})
	out["lang"] = out["text"].map(_safe_detect_lang)
	return _ensure_schema(out, source="imdb")


def load_indigo(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	# Hypothetical columns: 'text','label' with values {-1,0,1}
	label_map = {"-1": "negative", "0": "neutral", "1": "positive", -1: "negative", 0: "neutral", 1: "positive"}
	text_col = "text" if "text" in df.columns else df.columns[0]
	label_col = "label" if "label" in df.columns else df.columns[-1]
	out = pd.DataFrame({
		"text": df[text_col].astype(str),
		"label": df[label_col].map(label_map).fillna("neutral"),
	})
	out["lang"] = out["text"].map(_safe_detect_lang)
	return _ensure_schema(out, source="indigo")


def load_opinion(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	# Hypothetical columns: 'text','class' with values {0,1,2} or strings
	label_map = {
		0: "negative",
		1: "neutral",
		2: "positive",
		"negative": "negative",
		"neutral": "neutral",
		"positive": "positive",
	}
	text_col = "text" if "text" in df.columns else df.columns[0]
	label_col = "class" if "class" in df.columns else ("label" if "label" in df.columns else df.columns[-1])
	out = pd.DataFrame({
		"text": df[text_col].astype(str),
		"label": df[label_col].map(label_map).fillna("neutral"),
	})
	out["lang"] = out["text"].map(_safe_detect_lang)
	return _ensure_schema(out, source="opinion")


LOADER_BY_NAME: Dict[DatasetName, callable] = {
	"corona": load_corona,
	"imdb": load_imdb,
	"indigo": load_indigo,
	"opinion": load_opinion,
}


def unify_and_split(
	name: DatasetName, csv_path: Path, out_dir: Path, run_name: str, seed: int = 42
) -> Tuple[pd.DataFrame, SplitPaths]:
	LOGGER.info("Loading dataset %s from %s", name, csv_path)
	loader = LOADER_BY_NAME[name]
	df = loader(csv_path)
	# Remove rows with unknown labels
	df = df[df["label"].isin(LABELS)].reset_index(drop=True)
	paths = _split_save(df, out_dir=out_dir, run_name=run_name, source=name, seed=seed)
	return df, paths


def save_label_mapping_json(path: Path) -> None:
	mapping = {
		"corona": {
			"Extremely Negative/Negative": "negative",
			"Neutral": "neutral",
			"Extremely Positive/Positive": "positive",
		},
		"imdb": {"pos/positive": "positive", "neg/negative": "negative"},
		"indigo": {"-1": "negative", "0": "neutral", "1": "positive"},
		"opinion": {"0": "negative", "1": "neutral", "2": "positive"},
	}
	path.write_text(json.dumps(mapping, indent=2))


__all__ = [
	"load_corona",
	"load_imdb",
	"load_indigo",
	"load_opinion",
	"unify_and_split",
	"save_label_mapping_json",
]