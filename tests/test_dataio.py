from __future__ import annotations

from pathlib import Path

import pandas as pd

from nlp_lab.dataio.loader import (
	LABELS,
	load_corona,
	load_imdb,
	load_indigo,
	load_opinion,
)


def _tmp_csv(tmp_path: Path, df: pd.DataFrame, name: str) -> Path:
	p = tmp_path / f"{name}.csv"
	df.to_csv(p, index=False)
	return p


def test_loaders_unify_schema(tmp_path: Path):
	corona = pd.DataFrame({"OriginalTweet": ["bad", "ok", "good"], "Sentiment": ["Negative", "Neutral", "Positive"]})
	imdb = pd.DataFrame({"review": ["bad", "good"], "sentiment": ["neg", "pos"]})
	indigo = pd.DataFrame({"text": ["bad", "meh", "good"], "label": [-1, 0, 1]})
	opinion = pd.DataFrame({"text": ["bad", "meh", "good"], "class": [0, 1, 2]})

	pc = _tmp_csv(tmp_path, corona, "corona")
	pi = _tmp_csv(tmp_path, imdb, "imdb")
	pn = _tmp_csv(tmp_path, indigo, "indigo")
	po = _tmp_csv(tmp_path, opinion, "opinion")

	df_c = load_corona(pc)
	df_i = load_imdb(pi)
	df_n = load_indigo(pn)
	df_o = load_opinion(po)

	for d in (df_c, df_i, df_n, df_o):
		assert list(d.columns) == ["id", "text", "label", "source", "lang"]
		assert set(d["label"].dropna().astype(str).unique()).issubset(set(LABELS))