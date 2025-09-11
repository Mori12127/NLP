# NLP SpellCorrector

---

## “Let there be clarity in this text”

---

### The Legend

Once upon a time, a group of students sought the perfect clarity in communication. They dreamed of a spell checker so user-friendly and visually delightful, that anyone — from a scientist to a poet — could correct errors and learn from their mistakes with joy. This project is the result of that dream: a gentle, cloud-like English spell checker, made for education, demos, and discovery.

---

## Features

- **Gentle Mint-Green Theme:** The entire interface is styled in cloud-soft, minty green colors for comfort and clarity.
- **Modern, Rounded UI:** All buttons and fields have rounded corners, soft shadows, and a welcoming appearance.
- **English-Only Interface:** All labels, messages, and instructions are in English for international accessibility.
- **Motto:** “Let there be clarity in this text” is displayed on every launch, setting the tone for your session.
- **No Console Needed:** All actions (corpus loading, training, spell checking, dictionary search) are performed via intuitive buttons.
- **Dataset Loader:** Upload any text file as your corpus with a single click.
- **Train in One Click:** Instantly build a custom word dictionary and language model from your chosen dataset.
- **Advanced Spell Checking:** 
  - Detects both non-word errors and real-word (contextual) errors.
  - Offers suggestions ranked by edit distance and context (bigrams).
  - Click on errors to choose a correction directly in the text!
- **Dictionary Explorer:** Search and browse the vocabulary from your current model.
- **Student Signature:** A unique color theme and playful legend, ensuring your work stands out in any demo or grading session.

---

## Quickstart

### Requirements

- Python 3.8+
- `nltk` (Natural Language Toolkit)

#### Install dependencies

```bash
pip install nltk
```

### Run the App

```bash
python spell_checker.py
```

### How to Use

1. **Load dataset:** Click “Load dataset” and select any English `.txt` file (or use the built-in Brown corpus by default).
2. **Train:** Click “Train” to use your corpus for spell checking.
3. **Type or paste text** (up to 500 characters) in the main window.
4. **Check spelling:** Click “Check spelling”. Errors will be highlighted:
    - Yellow: spelling (non-word) error
    - Pink: real-word/context error (based on bigram model)
5. **Click on a highlighted error:** Instantly pick a correction or accept the suggested replacement.
6. **Dictionary search:** Enter any substring to browse words in your loaded vocabulary.

---

## Project Structure

- `spell_checker.py` — Main GUI, all logic and design
- `requirements.txt` — Python dependencies

---

## Why is This Special?

- **Originality:** The design, colors, and legend are unique to this group — no generic templates!
- **Educational Value:** Explore NLP concepts (edit distance, bigrams, language modeling) with instant visual feedback.
- **Maximum Usability:** No coding, no terminal commands needed after setup — everything is one click away!

---

## License

MIT

---