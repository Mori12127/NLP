# ===============================
# NLP SpellCorrector — Main File
# “Let there be clarity in this text”
# ===============================

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import nltk
import re
from collections import Counter
from nltk.util import bigrams
from nltk.metrics import edit_distance

# ===============================
# 0. UI COLORS & FONTS
# ===============================
BG_COLOR = "#eaffea"
HEADER_COLOR = "#b9fbc0"
BTN_COLOR = "#aaf6d3"
BTN_HOVER = "#d6fff6"
ERR_COLOR = "#ffe194"
REAL_ERR_COLOR = "#ffb3b3"
TEXT_BG = "#f7fff7"
FONT = "Segoe UI"

# ===============================
# PART 1: BASIC SPELL CHECKER (NON-WORD ERRORS)
# ===============================
# --- Corpus loading, edit distance, candidates, GUI for basic checking ---

VOCAB = set()
WORD_COUNTS = Counter()
BIGRAMS = Counter()
DATASET_NAME = "Brown (NLTK, Science)"

def load_brown():
    """Load default Brown corpus from NLTK (science/news subset)."""
    global VOCAB, WORD_COUNTS, BIGRAMS, DATASET_NAME
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        nltk.download('brown')
    from nltk.corpus import brown
    corpus_words = [w.lower() for w in brown.words() if w.isalpha()]
    VOCAB = set(corpus_words)
    WORD_COUNTS = Counter(corpus_words)
    BIGRAMS = Counter(bigrams(corpus_words))
    DATASET_NAME = "Brown (NLTK, Science)"

def load_custom_dataset(filepath):
    """Load and process user-selected text corpus."""
    global VOCAB, WORD_COUNTS, BIGRAMS, DATASET_NAME
    with open(filepath, encoding='utf-8') as f:
        text = f.read()
    words = [w.lower() for w in re.findall(r'\b[a-z]+\b', text)]
    VOCAB = set(words)
    WORD_COUNTS = Counter(words)
    BIGRAMS = Counter(bigrams(words))
    DATASET_NAME = f"Custom: {filepath.split('/')[-1]}"

def known(words):
    """Return the subset of words that are in the vocabulary."""
    return set(w for w in words if w in VOCAB)

def edit1(word):
    """Generate all words that are one edit away from the input word."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    deletes = [L+R[1:] for L,R in splits if R]
    transposes = [L+R[1]+R[0]+R[2:] for L,R in splits if len(R)>1]
    replaces = [L+c+R[1:] for L,R in splits if R for c in letters]
    inserts = [L+c+R for L,R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def candidates(word):
    """Generate possible spelling corrections for a word."""
    return (known([word]) or
            known(edit1(word)) or
            known(e2 for e1 in edit1(word) for e2 in edit1(e1)) or
            [word])

def suggest(word, prev=None, nextw=None):
    """
    Return top-5 spelling suggestions for a word,
    ranked by edit distance and bigram context.
    """
    sug = []
    for cand in candidates(word):
        dist = edit_distance(word, cand)
        bigram_score = 0
        if prev:
            bigram_score -= BIGRAMS[(prev, cand)]
        if nextw:
            bigram_score -= BIGRAMS[(cand, nextw)]
        sug.append((cand, dist, bigram_score))
    sug.sort(key=lambda x: (x[1], x[2], -WORD_COUNTS[x[0]]))
    return sug[:5]

# ===============================
# PART 2: CONTEXTUAL SPELL CHECKER (REAL-WORD ERRORS, BIGRAMS)
# ===============================
# --- Contextual error detection and GUI for their highlighting/correction ---

def get_real_word_errors(words):
    """
    Return indexes of likely real-word (contextual) errors in sequence of words,
    based on bigram probability.
    """
    errors = []
    for i in range(1, len(words)):
        pair = (words[i-1], words[i])
        if words[i-1] in VOCAB and words[i] in VOCAB:
            score = BIGRAMS[pair]
            alternatives = [w for w, _, _ in suggest(words[i], prev=words[i-1]) if w != words[i]]
            best_alt = None
            best_score = score
            for alt in alternatives:
                alt_score = BIGRAMS[(words[i-1], alt)]
                if alt_score > best_score:
                    best_score = alt_score
                    best_alt = alt
            if best_alt and best_score > score*2 and best_score > 3:
                errors.append((i, words[i], words[i-1], best_alt))
    return errors

# ===============================
# SHARED: GUI COMPONENTS, LOGIC, AND MAIN APP
# ===============================
class RoundedButton(tk.Button):
    """Custom button class for rounded, minty design."""
    def __init__(self, master=None, cnf={}, **kw):
        super().__init__(master, cnf, **kw)
        self["relief"] = "flat"
        self["bd"] = 0
        self["highlightthickness"] = 0
        self["font"] = (FONT, 11, "bold")
        self["bg"] = BTN_COLOR
        self["activebackground"] = BTN_HOVER
        self["fg"] = "#2d4739"
        self["cursor"] = "hand2"
        self["overrelief"] = "flat"

class SpellApp:
    def __init__(self, root):
        self.root = root
        root.title("NLP SpellCorrector")
        root.geometry("830x650")
        root.configure(bg=BG_COLOR)
        self.dataset_path = None

        # --- HEADER & MOTTO ---
        self.header = tk.Label(root, text="NLP SpellCorrector",
                               font=(FONT, 25, "bold"), bg=HEADER_COLOR, fg="#2d4739", pady=10, relief="flat")
        self.header.pack(fill=tk.X, pady=(0, 5))
        self.motto = tk.Label(root, text="Let there be clarity in this text",
                              font=(FONT, 15, "italic"), bg=BG_COLOR, fg="#2d4739")
        self.motto.pack(pady=(0, 13))

        # --- DATASET INFO ---
        self.dataset_label = tk.Label(root, text=f"Current dataset: {DATASET_NAME}",
                                      font=(FONT, 12), bg=BG_COLOR, fg="#2d4739")
        self.dataset_label.pack(pady=(0, 5))

        # --- BUTTONS ---
        self.btn_frame = tk.Frame(root, bg=BG_COLOR)
        self.btn_frame.pack(pady=(0, 5))
        self.load_btn = RoundedButton(self.btn_frame, text="Load dataset", command=self.load_dataset, width=17)
        self.train_btn = RoundedButton(self.btn_frame, text="Train", command=self.train, width=10)
        self.check_btn = RoundedButton(self.btn_frame, text="Check spelling", command=self.check, width=17)
        self.load_btn.grid(row=0, column=0, padx=8, pady=3)
        self.train_btn.grid(row=0, column=1, padx=8, pady=3)
        self.check_btn.grid(row=0, column=2, padx=8, pady=3)

        # --- TEXT FIELD FOR SPELLCHECK ---
        self.text = tk.Text(root, height=9, width=78, font=(FONT, 14), wrap="word",
                            bg=TEXT_BG, relief="flat", highlightthickness=2, highlightbackground=HEADER_COLOR)
        self.text.pack(pady=7)
        self.text.bind('<KeyRelease>', self.limit_text)

        # --- DICTIONARY SEARCH ---
        dict_label = tk.Label(root, text="Dictionary search:", font=(FONT, 11), bg=BG_COLOR, fg="#4bbf8f")
        dict_label.pack()
        self.search_entry = tk.Entry(root, font=(FONT, 11), width=21, relief="flat", bg="#e6ffe6")
        self.search_entry.pack()
        self.search_btn = RoundedButton(root, text="Search", command=self.search_dict, width=13)
        self.search_btn.pack(pady=(2, 3))
        self.dict_list = tk.Listbox(root, width=30, height=5, font=(FONT, 11), bg="#e6ffe6", fg="#00a77d",
                                    highlightthickness=0, relief="flat", selectbackground=HEADER_COLOR)
        self.dict_list.pack(pady=(1, 12))
        self.dict_list.insert(tk.END, *sorted(list(VOCAB))[:300])

        # --- SUGGESTIONS & ERROR HIGHLIGHTING ---
        self.sug_label = tk.Label(root, text="", font=(FONT, 12), fg="#00a77d", bg=BG_COLOR)
        self.sug_label.pack()
        self.text.tag_config("err", background=ERR_COLOR)
        self.text.tag_config("rerr", background=REAL_ERR_COLOR)
        self.text.bind("<Button-1>", self.on_click)

        # --- FOOTER ---
        self.footer = tk.Label(root, text="Created by students", font=(FONT, 10, "italic"),
                               bg=BG_COLOR, fg="#83e8ba")
        self.footer.pack(side=tk.BOTTOM, fill=tk.X, pady=8)

        # --- INTERNAL STATE ---
        self.err_pos = []
        self.rerr_pos = []
        load_brown()

    # ----- DATASET/INTERFACE UPDATES -----
    def update_dataset_label(self):
        self.dataset_label.config(text=f"Current dataset: {DATASET_NAME}")

    def limit_text(self, event=None):
        content = self.text.get("1.0", tk.END)
        if len(content) > 501:
            self.text.delete("1.0+500c", tk.END)

    # ----- BUTTON CALLBACKS (LOAD/TRAIN) -----
    def load_dataset(self):
        path = filedialog.askopenfilename(title="Select text corpus",
                                          filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            self.dataset_path = path
            messagebox.showinfo("Loaded!", f"File {path.split('/')[-1]} selected. Click 'Train' to process it.")
            self.dataset_label.config(text=f"Selected: {path.split('/')[-1]} (not trained)")

    def train(self):
        if not self.dataset_path:
            messagebox.showwarning("No file", "Please select a dataset file first!")
            return
        try:
            load_custom_dataset(self.dataset_path)
            self.update_dataset_label()
            self.dict_list.delete(0, tk.END)
            self.dict_list.insert(tk.END, *sorted(list(VOCAB))[:300])
            messagebox.showinfo("Training finished",
                                f"Corpus {self.dataset_path.split('/')[-1]} has been processed.")
        except Exception as e:
            messagebox.showerror("Training error", str(e))

    # ----- SPELL CHECKING & ERROR HIGHLIGHTING -----
    def check(self):
        self.text.tag_remove("err", "1.0", tk.END)
        self.text.tag_remove("rerr", "1.0", tk.END)
        s = self.text.get("1.0", tk.END).lower()
        words = re.findall(r'\b[a-z]+\b', s)
        self.err_pos = []
        self.rerr_pos = []
        word_positions = []
        pos = 0
        for word in words:
            match = re.search(r'\b'+re.escape(word)+r'\b', s[pos:])
            if match:
                start = pos + match.start()
                end = pos + match.end()
                word_positions.append((start, end, word))
                pos = end

        # --- Non-word errors (PART 1) ---
        prev = None
        for idx, (start, end, word) in enumerate(word_positions):
            if word not in VOCAB:
                self.text.tag_add("err", f"1.0+{start}c", f"1.0+{end}c")
                self.err_pos.append((f"1.0+{start}c", f"1.0+{end}c", word, prev, 
                                     words[idx+1] if idx+1 < len(words) else None))
            prev = word

        # --- Real-word (contextual) errors (PART 2) ---
        rerrs = get_real_word_errors(words)
        for i, w, prev, best in rerrs:
            for j, (start, end, word) in enumerate(word_positions):
                if j == i:
                    self.text.tag_add("rerr", f"1.0+{start}c", f"1.0+{end}c")
                    self.rerr_pos.append((f"1.0+{start}c", f"1.0+{end}c", w, prev, best))
                    break

        # --- Show suggestions ---
        if self.err_pos:
            word, prev, nextw = self.err_pos[0][2], self.err_pos[0][3], self.err_pos[0][4]
            sug = suggest(word, prev, nextw)
            self.sug_label.config(text=f"Error: '{word}'. Suggestions: {[s[0] for s in sug]}")
        elif self.rerr_pos:
            word, prev, best = self.rerr_pos[0][2], self.rerr_pos[0][3], self.rerr_pos[0][4]
            self.sug_label.config(text=f"Contextual error: '{word}' after '{prev}'. More likely: '{best}'")
        else:
            self.sug_label.config(text="No errors found!")

    # ----- CLICKABLE SUGGESTIONS & ERROR CORRECTION -----
    def on_click(self, event):
        idx = self.text.index(f"@{event.x},{event.y}")
        # Non-word error click (PART 1)
        for start, end, word, prev, nextw in self.err_pos:
            if self.text.compare(start, "<=", idx) and self.text.compare(idx, "<", end):
                sug = suggest(word, prev, nextw)
                sel = simpledialog.askstring("Choose suggestion",
                                             f"Suggestions for '{word}':\n" +
                                             "\n".join(f"{i+1}. {s[0]}" for i, s in enumerate(sug)) +
                                             "\nEnter number or fix manually:")
                try:
                    sel_idx = int(sel)-1
                    if 0 <= sel_idx < len(sug):
                        self.text.delete(start, end)
                        self.text.insert(start, sug[sel_idx][0])
                        self.check()
                except:
                    pass
                break
        # Real-word error click (PART 2)
        for start, end, word, prev, best in self.rerr_pos:
            if self.text.compare(start, "<=", idx) and self.text.compare(idx, "<", end):
                if messagebox.askyesno("Replace?", f"Replace '{word}' with '{best}'?"):
                    self.text.delete(start, end)
                    self.text.insert(start, best)
                    self.check()
                break

    # ----- DICTIONARY SEARCH LOGIC -----
    def search_dict(self):
        term = self.search_entry.get().strip().lower()
        results = [w for w in VOCAB if term in w]
        self.dict_list.delete(0, tk.END)
        for w in sorted(results)[:300]:
            self.dict_list.insert(tk.END, w)

# ===============================
# APP LAUNCH
# ===============================
if __name__ == '__main__':
    root = tk.Tk()
    app = SpellApp(root)
    root.mainloop()