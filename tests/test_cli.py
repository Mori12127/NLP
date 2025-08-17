from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from nlp_lab.training.cli import app


def test_cli_prep_and_train(tmp_path: Path):
	runner = CliRunner()
	corona = pd.DataFrame({"OriginalTweet": ["bad", "ok", "good"], "Sentiment": ["Negative", "Neutral", "Positive"]})
	pc = tmp_path / "corona.csv"
	corona.to_csv(pc, index=False)

	# prep only corona
	res = runner.invoke(app, ["prep", "--corona", str(pc), "--run-id", "t"])
	assert res.exit_code == 0

	# train baseline on just prepared data
	res = runner.invoke(app, ["train", "--model", "baseline", "--datasets", "corona", "--run-id", "t"])
	assert res.exit_code == 0