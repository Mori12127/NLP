from __future__ import annotations

import logging
import os
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
	level = os.getenv("NLP_LAB_LOGLEVEL", "INFO").upper()
	logging.basicConfig(
		level=getattr(logging, level, logging.INFO),
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)
	return logging.getLogger(name if name else "nlp_lab")


__all__ = ["get_logger"]