from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from ..dataio.loader import LOADER_BY_NAME, save_label_mapping_json, unify_and_split
from ..utils.config import Config
from ..utils.logging import get_logger
from .train import train_baseline, train_transformer
from .eval import eval_run

app = typer.Typer(add_completion=False)
LOGGER = get_logger(__name__)


@app.command()
def prep(
	corona: Optional[Path] = typer.Option(None, exists=True),
	imdb: Optional[Path] = typer.Option(None, exists=True),
	indigo: Optional[Path] = typer.Option(None, exists=True),
	opinion: Optional[Path] = typer.Option(None, exists=True),
	run_id: Optional[str] = None,
):
	cfg = Config()
	if run_id:
		cfg.run_id = run_id
	name_to_path = {"corona": corona, "imdb": imdb, "indigo": indigo, "opinion": opinion}
	for name, path in name_to_path.items():
		if path is None:
			LOGGER.info("Dataset %s path not provided; skipping", name)
			continue
		_, paths = unify_and_split(name, path, out_dir=cfg.paths.processed_dir, run_name=cfg.run_id or "run", seed=cfg.seed)
	LOGGER.info("Prep finished")


@app.command()
def train(model: str = typer.Option("baseline", help="baseline|transformer"), datasets: str = typer.Option("all", help="comma-separated or 'all'"), run_id: Optional[str] = None):
	cfg = Config()
	if run_id:
		cfg.run_id = run_id
	cfg.model = model  # type: ignore
	if datasets != "all":
		cfg.datasets = [d.strip() for d in datasets.split(",") if d.strip()]
	if cfg.model == "baseline":
		metrics = train_baseline(cfg)
	else:
		metrics = train_transformer(cfg)
	typer.echo(metrics)


@app.command()
def eval(run: Path, run_id: Optional[str] = None):
	cfg = Config()
	if run_id:
		cfg.run_id = run_id
	metrics = eval_run(run, cfg)
	typer.echo(metrics)


if __name__ == "__main__":
	app()