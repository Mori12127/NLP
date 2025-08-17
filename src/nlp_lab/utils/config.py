from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DatasetName = Literal["corona", "imdb", "indigo", "opinion"]
ModelName = Literal["baseline", "transformer"]


class Paths(BaseModel):
	project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3])
	src_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
	data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data")
	models_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "models")
	processed_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "processed")
	interim_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "interim")
	raw_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "raw")


class PreprocessConfig(BaseModel):
	lowercase: bool = True
	strip_urls: bool = True
	strip_mentions: bool = True
	normalize_unicode: bool = True
	handle_emoji: bool = False
	use_speller: bool = False


class BaselineConfig(BaseModel):
	classifier: Literal["logreg", "linearsvc", "mnb"] = "logreg"
	max_features: int = 50000
	ngram_range: tuple[int, int] = (1, 2)
	min_df: int = 2
	C: float = 1.0


class TransformerConfig(BaseModel):
	model_name: str = "distilbert-base-uncased"
	epochs: int = 2
	batch_size: int = 8
	learning_rate: float = 5e-5
	gradient_accumulation_steps: int = 1
	early_stopping_patience: int = 2
	max_length: int = 256
	warmup_ratio: float = 0.0


class Config(BaseSettings):
	model_config = SettingsConfigDict(env_file=(".env", ".env.local"), env_prefix="NLP_LAB_", case_sensitive=False)

	seed: int = 42
	datasets: List[DatasetName] = Field(default_factory=lambda: ["corona", "imdb", "indigo", "opinion"])
	paths: Paths = Field(default_factory=Paths)
	preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
	baseline: BaselineConfig = Field(default_factory=BaselineConfig)
	transformer: TransformerConfig = Field(default_factory=TransformerConfig)
	model: ModelName = "baseline"
	run_id: Optional[str] = None


__all__ = [
	"Config",
	"DatasetName",
	"ModelName",
]