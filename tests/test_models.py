from __future__ import annotations

import numpy as np
import pandas as pd

from nlp_lab.models.baselines import eval_baseline_pipeline, train_baseline_pipeline
from nlp_lab.utils.config import BaselineConfig, PreprocessConfig


def test_baseline_smoke():
	texts = ["bad", "good", "meh", "bad product", "great movie", "average"] * 5
	labels = np.array(["negative", "positive", "neutral", "negative", "positive", "neutral"] * 5)
	pipe = train_baseline_pipeline(texts, labels, PreprocessConfig(), BaselineConfig(classifier="logreg"))
	metrics = eval_baseline_pipeline(pipe, texts, labels, PreprocessConfig())
	assert metrics["accuracy"] >= 0.3