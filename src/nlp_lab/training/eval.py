from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from ..models.baselines import load_baseline
from ..models.transformer import ID2LABEL, LABEL2ID, load_transformer
from ..utils.config import Config
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def _load_test_df(prefixes: list[str], processed_dir: Path) -> pd.DataFrame:
	frames = []
	for pr in prefixes:
		path = processed_dir / f"{pr}_test.parquet"
		frames.append(pd.read_parquet(path))
	return pd.concat(frames, ignore_index=True)


def eval_run(run_dir: Path, config: Config) -> Dict[str, float]:
	prefixes = [f"{config.run_id or 'run'}_{ds}" for ds in config.datasets]
	df = _load_test_df(prefixes, config.paths.processed_dir)
	X = df["text"].astype(str).tolist()
	y_true = df["label"].astype(str).tolist()

	if (run_dir / "baseline" / "model.joblib").exists():
		pipe = load_baseline(run_dir / "baseline" / "model.joblib")
		pred = pipe.predict(X)
	else:
		tok, model = load_transformer(run_dir / "transformer")
		import torch
		from torch.utils.data import DataLoader
		from transformers import DataCollatorWithPadding

		enc = tok(X, truncation=True, padding=False, max_length=256)
		labels = [LABEL2ID[l] for l in y_true]
		class DS:
			def __init__(self, enc, labels):
				self.enc, self.labels = enc, labels
			def __len__(self): return len(self.labels)
			def __getitem__(self, i):
				item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
				item["labels"] = torch.tensor(self.labels[i])
				return item
		ds = DS(enc, labels)
		collator = DataCollatorWithPadding(tokenizer=tok)
		loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collator)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device)
		pred_ids = []
		with torch.no_grad():
			for batch in loader:
				batch = {k: v.to(device) for k, v in batch.items()}
				out = model(**batch)
				pred_ids.extend(out.logits.argmax(-1).cpu().numpy().tolist())
		pred = [ID2LABEL[i] for i in pred_ids]

	acc = float(accuracy_score(y_true, pred))
	f1 = float(f1_score(y_true, pred, average="macro"))
	report = classification_report(y_true, pred, output_dict=True)
	metrics = {"test_accuracy": acc, "test_f1_macro": f1}
	(run_dir / "metrics_test.json").write_text(json.dumps({"metrics": metrics, "report": report}, indent=2))
	LOGGER.info("Wrote test report to %s", run_dir / "metrics_test.json")
	return metrics


__all__ = ["eval_run"]