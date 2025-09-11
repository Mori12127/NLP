import tkinter as tk
from tkinter import messagebox
import nltk
import re
from collections import Counter
from nltk.util import bigrams
from nltk.metrics import edit_distance

# --- Download corpus if needed ---
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')
from nltk.corpus import brown

# --- Prepare corpus and dictionary ---
corpus_words = [w.lower() for w in brown.words() if w.isalpha()]
VOCAB = set(corpus_words)
WORD_COUNTS = Counter(corpus_words)
BIGRAMS = Counter(bigrams(corpus_words))

def known(words):
    return set(w for w in words if w in VOCAB)

def edit1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    deletes = [L+R[1:] for L,R in splits if R]
    transposes = [L+R[1]+R[0]+R[2:] for L,R in splits if len(R)>1]
    replaces = [L+c+R[1:] for L,R in splits if R for c in letters]
    inserts = [L+c+R for L,R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def candidates(word):
    # Candidates for correction
    return (known([word]) or
            known(edit1(word)) or
            known(e2 for e1 in edit1(word) for e2 in edit1(e1)) or
            [word])

def suggest(word, prev=None):
    sug = []
    for cand in candidates(word):
        dist = edit_distance(word, cand)
        bigram_score = -BIGRAMS[(prev, cand)] if prev else 0
        sug.append((cand, dist, bigram_score))
    sug.sort(key=lambda x: (x[1], x[2], -WORD_COUNTS[x[0]]))
    return sug[:5]

# --- GUI ---
class SpellApp:
    def __init__(self, root):
        self.root = root
        root.title("Spelling Correction GUI")
        # Text field
        self.text = tk.Text(root, height=10, width=60, font=("Arial", 13))
        self.text.pack()
        self.text.bind('<KeyRelease>', self.limit_text)
        # Check button
        self.btn = tk.Button(root, text="Check Spelling", command=self.check)
        self.btn.pack(pady=10)
        # Dictionary list and search
        self.search_entry = tk.Entry(root)
        self.search_entry.pack()
        self.search_btn = tk.Button(root, text="Search Dictionary", command=self.search_dict)
        self.search_btn.pack()
        self.dict_list = tk.Listbox(root, width=30, height=8)
        self.dict_list.pack()
        self.dict_list.insert(tk.END, *sorted(list(VOCAB))[:300])
        # Suggestions
        self.sug_label = tk.Label(root, text="", fg="blue")
        self.sug_label.pack()
        # Click for suggestions
        self.text.tag_config("err", background="yellow")
        self.text.bind("<Button-1>", self.on_click)

    def limit_text(self, event=None):
        content = self.text.get("1.0", tk.END)
        if len(content) > 501:
            self.text.delete("1.0+500c", tk.END)

    def check(self):
        self.text.tag_remove("err", "1.0", tk.END)
        s = self.text.get("1.0", tk.END).lower()
        words = re.findall(r'\b[a-z]+\b', s)
        self.err_pos = []
        prev = None
        for idx, word in enumerate(words):
            if word not in VOCAB:
                match = re.search(r'\b'+re.escape(word)+r'\b', s)
                if match:
                    start = "1.0+%dc" % match.start()
                    end = "1.0+%dc" % match.end()
                    self.text.tag_add("err", start, end)
                    self.err_pos.append((start, end, word, prev))
            prev = word
        if not self.err_pos:
            self.sug_label.config(text="No spelling errors!")
        else:
            word, prev = self.err_pos[0][2], self.err_pos[0][3]
            sug = suggest(word, prev)
            self.sug_label.config(text=f"Suggestions for '{word}': {[s[0] for s in sug]}")

    def on_click(self, event):
        idx = self.text.index(f"@{event.x},{event.y}")
        for start, end, word, prev in self.err_pos:
            if self.text.compare(start, "<=", idx) and self.text.compare(idx, "<", end):
                sug = suggest(word, prev)
                self.sug_label.config(text=f"Suggestions for '{word}': {[s[0] for s in sug]}")
                break

    def search_dict(self):
        term = self.search_entry.get().strip().lower()
        results = [w for w in VOCAB if term in w]
        self.dict_list.delete(0, tk.END)
        for w in sorted(results)[:300]:
            self.dict_list.insert(tk.END, w)

if __name__ == '__main__':
    root = tk.Tk()
    app = SpellApp(root)
    root.mainloop()