from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from ..utils.config import TransformerConfig

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class TextDataset(Dataset):
	def __init__(self, texts: List[str], labels: List[int]):
		self.texts = texts
		self.labels = labels

	def __len__(self) -> int:
		return len(self.texts)

	def __getitem__(self, idx: int):
		return {"text": self.texts[idx], "label": self.labels[idx]}


@dataclass
class TrainOutput:
	best_model_dir: Path
	metrics: Dict[str, float]


def train_transformer_model(
	X_train: List[str], y_train: List[str], X_val: List[str], y_val: List[str], out_dir: Path, cfg: TransformerConfig
) -> TrainOutput:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
	model = AutoModelForSequenceClassification.from_pretrained(
		cfg.model_name,
		num_labels=3,
		id2label=ID2LABEL,
		label2id=LABEL2ID,
	)
	model.to(device)

	def encode_batch(batch):
		return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

	train_enc = tokenizer(X_train, truncation=True, padding=False, max_length=cfg.max_length)
	val_enc = tokenizer(X_val, truncation=True, padding=False, max_length=cfg.max_length)
	train_labels = [LABEL2ID[l] for l in y_train]
	val_labels = [LABEL2ID[l] for l in y_val]

	class EncodedDataset(Dataset):
		def __init__(self, encodings, labels):
			self.enc = encodings
			self.labels = labels

		def __len__(self):
			return len(self.labels)

		def __getitem__(self, idx):
			item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
			item["labels"] = torch.tensor(self.labels[idx])
			return item

	train_ds = EncodedDataset(train_enc, train_labels)
	val_ds = EncodedDataset(val_enc, val_labels)

	collator = DataCollatorWithPadding(tokenizer=tokenizer)
	train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
	val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)

	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
	t_total = len(train_loader) * cfg.epochs
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		int(cfg.warmup_ratio * t_total),
		t_total,
	)

	best_f1 = -1.0
	best_dir = out_dir / "transformer"
	best_dir.mkdir(parents=True, exist_ok=True)
	patience = 0

	for epoch in range(cfg.epochs):
		model.train()
		each_step = 0
		optimizer.zero_grad()
		for batch in train_loader:
			batch = {k: v.to(device) for k, v in batch.items()}
			outputs = model(**batch)
			loss = outputs.loss
			loss.backward()
			if (each_step + 1) % cfg.gradient_accumulation_steps == 0:
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
			each_step += 1

		# eval
		model.eval()
		all_preds: List[int] = []
		all_labels: List[int] = []
		with torch.no_grad():
			for batch in val_loader:
				labels = batch["labels"].numpy().tolist()
				batch = {k: v.to(device) for k, v in batch.items()}
				outputs = model(**batch)
				preds = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()
				all_preds.extend(preds)
				all_labels.extend(labels)
		acc = float(accuracy_score(all_labels, all_preds))
		f1 = float(f1_score(all_labels, all_preds, average="macro"))
		if f1 > best_f1:
			best_f1 = f1
			model.save_pretrained(best_dir)
			tokenizer.save_pretrained(best_dir)
			patience = 0
		else:
			patience += 1
			if patience >= cfg.early_stopping_patience:
				break

	return TrainOutput(best_model_dir=best_dir, metrics={"val_accuracy": acc, "val_f1_macro": best_f1})


def load_transformer(path: Path):
	tokenizer = AutoTokenizer.from_pretrained(path)
	model = AutoModelForSequenceClassification.from_pretrained(path)
	return tokenizer, model


__all__ = ["train_transformer_model", "load_transformer", "LABEL2ID", "ID2LABEL"]