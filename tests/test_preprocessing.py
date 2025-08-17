from __future__ import annotations

from nlp_lab.preprocessing.text import normalize_text
from nlp_lab.utils.config import PreprocessConfig


def test_normalize_basic():
	cfg = PreprocessConfig()
	text = "Hello   World! visit https://example.com @user"
	out = normalize_text(text, cfg)
	assert "http" not in out
	assert "@" not in out
	assert "  " not in out
	assert out == out.lower()


def test_unicode_nfc():
	cfg = PreprocessConfig(normalize_unicode=True)
	text = "A"  # odd control char
	out = normalize_text(text, cfg)
	assert isinstance(out, str)
	assert len(out) >= 1