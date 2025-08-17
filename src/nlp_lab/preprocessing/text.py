from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

from ..utils.config import PreprocessConfig

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, cfg: PreprocessConfig) -> str:
	"""Apply basic normalization to text according to config.

	- NFC normalization
	- optional lowercase
	- remove URLs and mentions
	- collapse whitespace
	"""
	if text is None:
		return ""
	result = text
	if cfg.normalize_unicode:
		result = unicodedata.normalize("NFC", result)
	if cfg.strip_urls:
		result = URL_RE.sub(" ", result)
	if cfg.strip_mentions:
		result = MENTION_RE.sub(" ", result)
	result = WHITESPACE_RE.sub(" ", result).strip()
	if cfg.lowercase:
		result = result.lower()
	return result


def preprocess_corpus(texts: Iterable[str], cfg: PreprocessConfig) -> List[str]:
	return [normalize_text(t, cfg) for t in texts]


def speller_stub(text: str) -> str:
	return text


__all__ = ["normalize_text", "preprocess_corpus", "speller_stub"]