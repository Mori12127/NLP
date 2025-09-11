# Spell Correction System (NLP Assignment)

## 1. Overview
This project implements a probabilistic spelling correction system for non-word and real-word errors using an English science field corpus (>100,000 words) and advanced NLP methods. It includes a simple GUI for demonstration and a sorted searchable dictionary explorer.

### Key Features
- Detects and highlights spelling errors (non-words and real-word/context errors)
- Suggests corrections with minimum edit distance and bigram context
- GUI editor (500 char limit), dictionary search, clickable error suggestions
- Based on NLTK's Brown corpus (over 1M words, science domain)

---

## 2. Requirements
- Python 3.8+
- `nltk` (Natural Language Toolkit)

Install dependencies:
```bash
pip install nltk
```

---

## 3. Usage
Run the spell checker GUI:
```bash
python spell_checker.py
```

---

## 4. Files
- `spell_checker.py` — Main GUI application and spelling correction logic
- `requirements.txt` — Python dependencies

---

## 5. Brief Theory/Report
### 1.1. Candidate Techniques
Probabilistic and edit distance-based spelling correction is widely used. Key Python packages: `nltk`, `pyspellchecker`, `textblob`, `editdistance`. Approaches include n-gram models (bigrams), Minimum Edit Distance (Levenshtein), and context models (Noisy Channel).

### 1.2. Edit Distance
Levenshtein distance (insert, delete, substitute), Damerau-Levenshtein (also transpositions), Hamming (fixed length), Jaro-Winkler (short strings). This system uses Levenshtein distance from NLTK for candidate suggestion.

### 1.3. System Design
- Corpus: NLTK Brown (science/news/academic, >1M words)
- Dictionary: set of unique words, with frequency
- Bigram counts: for context-based correction
- GUI: Tkinter, 500-char text editor, dictionary explorer, error highlighting
- Suggestions: edit distance + bigram context

### 1.4. Implementation
- Loads Brown corpus, builds vocab and bigram stats
- GUI with spell check, word search, error highlighting, clickable suggestions
- Efficient edit distance computation and candidate ranking

### 1.5. Results
- Accurately finds and suggests corrections for common and context-based errors
- Easy to test and demonstrate via GUI
- Limitations: real-word errors beyond bigram context (needs full sentence context/model)

---

## License
MIT
