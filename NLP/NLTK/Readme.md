# NLTK (Natural Language Toolkit) Fundamentals & Classical NLP

A comprehensive guide and reference covering the foundational concepts, algorithms, and practical workflows for Natural Language Processing (NLP) using **NLTK** based on `main.ipynb`.

---

## Table of Contents
1. [Introduction to NLTK & Architecture](#1-introduction-to-nltk--architecture)
2. [Essential NLTK Resources & Corpus Downloads](#2-essential-nltk-resources--corpus-downloads)
3. [Text Preprocessing Pipeline](#3-text-preprocessing-pipeline)
   * [3.1 Tokenization (Sentence & Word)](#31-tokenization-sentence--word)
   * [3.2 Stopwords & Punctuation Filtering](#32-stopwords--punctuation-filtering)
   * [3.3 Stemming vs. Lemmatization](#33-stemming-vs-lemmatization)
4. [Exploring Linguistic Corpora (The Brown Corpus)](#4-exploring-linguistic-corpora-the-brown-corpus)
5. [Part-of-Speech (POS) Tagging](#5-part-of-speech-pos-tagging)
6. [Chunking & Shallow Parsing (RegexpParser)](#6-chunking--shallow-parsing-regexpparser)
7. [Context-Free Grammars (CFG) & Syntactic Parsing](#7-context-free-grammars-cfg--syntactic-parsing)
8. [Comprehensive Comparison Tables](#8-comprehensive-comparison-tables)
9. [Summary Workflow Diagram](#9-summary-workflow-diagram)

---

## 1. Introduction to NLTK & Architecture

The **Natural Language Toolkit (NLTK)** is the standard educational and research library for classical, rule-based, and statistical Natural Language Processing in Python.

### Core Philosophy
* **Symbolic & Statistical NLP**: Focuses on linguistic structures (syntax trees, grammar rules, morphology, lexicons) rather than end-to-end deep learning embeddings.
* **Modular Pipeline**: Provides discrete tools for tokenization, normalization, POS tagging, chunking, parsing, and corpus analysis.

```python
import nltk
```

---

## 2. Essential NLTK Resources & Corpus Downloads

NLTK separates code logic from linguistic corpora and statistical models. Before running NLP tasks, specific data models must be downloaded:

```python
# Unsupervised Tokenizer Models (Punkt Sentence & Word Tokenizers)
nltk.download('punkt')
nltk.download('punkt_tab')

# Stopwords List (multi-lingual common functional words)
nltk.download('stopwords')

# WordNet Lexical Database (for lemmatization & semantic synsets)
nltk.download('wordnet')

# Averaged Perceptron POS Tagger Models
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# The Brown Corpus (1M-word balanced English corpus across genres)
nltk.download('brown')
```

---

## 3. Text Preprocessing Pipeline

Text normalization transforms messy unstructured text into a standardized sequence of tokens suitable for downstream feature extraction or modeling.

```
Raw Text: "Natural Language Processing with NLTK is amazing! The dogs were running quickly across fields."
    │
    ▼ [1. Sentence & Word Tokenization]
Tokens: ['natural', 'language', 'processing', 'with', 'nltk', 'is', 'amazing', '!', 'the', 'dogs', 'were', 'running', 'quickly', 'across', 'fields', '.']
    │
    ▼ [2. Stopword & Punctuation Removal]
Filtered: ['natural', 'language', 'processing', 'nltk', 'amazing', 'dogs', 'running', 'quickly', 'across', 'fields']
    │
    ├──▶ [3A. Stemming (PorterStemmer)] ──▶ ['natur', 'languag', 'process', 'nltk', 'amaz', 'dog', 'run', 'quickli', 'across', 'field']
    │
    └──▶ [3B. Lemmatization (WordNet)]  ──▶ ['natural', 'language', 'process', 'nltk', 'amaze', 'dog', 'run', 'quickly', 'across', 'field']
```

---

### 3.1 Tokenization (Sentence & Word)
Splitting text into discrete linguistic units (sentences or words) while handling punctuation boundaries:

```python
from nltk.tokenize import sent_tokenize, word_tokenize

raw_text = "Natural Language Processing with NLTK is amazing! The dogs were running quickly across fields."

# 1. Sentence Tokenization (splits on punctuation with abbreviation awareness)
sentences = sent_tokenize(raw_text)
# Output: ['Natural Language Processing with NLTK is amazing!', 'The dogs were running quickly across fields.']

# 2. Word Tokenization (separates words, contractions, and punctuation)
tokens = word_tokenize(raw_text.lower())
```

---

### 3.2 Stopwords & Punctuation Filtering
Removing high-frequency functional words ("is", "with", "the", "were") and punctuation marks that add minimal semantic value:

```python
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

cleaned_tokens = [
    w for w in tokens 
    if w not in stop_words and w not in punctuation
]

print("Cleaned Tokens:", cleaned_tokens)
# Output: ['natural', 'language', 'processing', 'nltk', 'amazing', 'dogs', 'running', 'quickly', 'across', 'fields']
```

---

### 3.3 Stemming vs. Lemmatization

Both methods reduce inflectional forms and derivationally related forms of words to a common base form, but use fundamentally different approaches:

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Stemming (Heuristic suffix-stripping)
stemmed = [stemmer.stem(w) for w in cleaned_tokens]
# Output: ['natur', 'languag', 'process', 'nltk', 'amaz', 'dog', 'run', 'quickli', 'across', 'field']

# Lemmatization (Dictionary lookup with POS tag)
lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in cleaned_tokens]
# Output: ['natural', 'language', 'process', 'nltk', 'amaze', 'dog', 'run', 'quickly', 'across', 'field']
```

* **Stemming (Porter Stemmer)**: A fast, rule-based algorithmic heuristic that strips common suffixes (`-ing`, `-ly`, `-es`, `-tion`). It does **not** check a dictionary and often produces non-real root words (e.g., `"quickly"` $\to$ `"quickli"`, `"amazing"` $\to$ `"amaz"`).
* **Lemmatization (WordNet Lemmatizer)**: A vocabulary and morphological analysis approach that returns the true dictionary lemma (base word). It requires knowing or specifying the Part-of-Speech (`pos='v'` for verb, `pos='n'` for noun).

---

## 4. Exploring Linguistic Corpora (The Brown Corpus)

NLTK includes pre-packaged annotated corpora. The **Brown Corpus** (compiled in 1961 at Brown University) is a standard benchmark containing 1 million words of American English balanced across 15 genres.

```python
from nltk.corpus import brown

# 1. View genre categories
print("Categories:", brown.categories()[:5])
# Output: ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government']

# 2. Extract raw words for a specific genre
news_words = brown.words(categories='news')
print("Total words in news:", len(news_words)) # 100,554 words
print("First 10 words:", news_words[:10])

# 3. Access ground-truth POS tagged sentences
tagged_sents = brown.tagged_sents(categories='news')
print("Sample Tagged Sentence:", tagged_sents[0][:5])
# Output: [('The', 'AT'), ('Fulton', 'NP-TL'), ('County', 'NN-TL'), ('Grand', 'JJ-TL'), ('Jury', 'NN-TL')]
```

---

## 5. Part-of-Speech (POS) Tagging

**Part-of-Speech (POS) Tagging** is the process of labeling each word in a text with its syntactic category (noun, verb, adjective, adverb, preposition) based on both its definition and its grammatical context.

NLTK uses the **Penn Treebank Tagset** and an **Averaged Perceptron Tagger**:

```python
sentence = "The small dog barked at the mailman"
tokens = nltk.word_tokenize(sentence)

pos_tags = nltk.pos_tag(tokens)
print("POS Tags:", pos_tags)
# Output: [('The', 'DT'), ('small', 'JJ'), ('dog', 'NN'), ('barked', 'VBN'), ('at', 'IN'), ('the', 'DT'), ('mailman', 'NN')]
```

### Common Penn Treebank POS Tags Reference

| Tag | Meaning | Example |
| :--- | :--- | :--- |
| `DT` | Determiner | the, a, an, this |
| `JJ` | Adjective | small, quick, green |
| `NN` | Noun, singular | dog, mailman, table |
| `NNS` | Noun, plural | dogs, mailmen, tables |
| `VBD` | Verb, past tense | barked, chased, ran |
| `VBN` | Verb, past participle | barked, eaten, written |
| `VBZ` | Verb, 3rd person singular present | barks, runs, is |
| `IN` | Preposition / Subordinating conjunction | at, in, on, across, with |
| `PRP` | Personal pronoun | he, she, it, they |
| `RB` | Adverb | quickly, slowly, very |

---

## 6. Chunking & Shallow Parsing (RegexpParser)

**Chunking (Shallow Parsing)** identifies non-overlapping constituent groups of tokens (such as Noun Phrases `NP` or Verb Phrases `VP`) without building a full hierarchical parse tree.

### Noun Phrase (NP) Chunking Rule
A regular expression grammar defines what sequence of POS tags constitutes a Noun Phrase:
* **Rule**: `NP: {<DT>?<JJ>*<NN>}`
  * `<DT>?`: Optional determiner (0 or 1)
  * `<JJ>*`: 0 or more adjectives
  * `<NN>`: Ending with a noun

```python
# 1. Define chunk grammar rule
chunk_grammar = r"NP: {<DT>?<JJ>*<NN>}"

# 2. Initialize regex parser
chunk_parser = nltk.RegexpParser(chunk_grammar)

# 3. Parse POS-tagged tokens into a Chunk Tree
chunk_tree = chunk_parser.parse(pos_tags)
print(chunk_tree)
```

### Generated Chunk Tree:
```
(S
  (NP The/DT small/JJ dog/NN)
  barked/VBN
  at/IN
  (NP the/DT mailman/NN))
```

---

## 7. Context-Free Grammars (CFG) & Syntactic Parsing

While chunking extracts flat phrases, **Syntactic Parsing** with **Context-Free Grammars (CFG)** builds a full recursive hierarchical syntax tree representing the formal grammatical structure of a sentence.

### 7.1 Formal Grammar Definition
$$S \to NP \ VP$$
$$NP \to DT \ JJ \ NN \mid DT \ NN \mid PRP$$
$$VP \to VBD \ NP \mid VBD \ PP \mid VBD$$
$$PP \to IN \ NP$$

```python
from nltk import CFG

# 1. Define explicit production rules
toy_grammar = CFG.fromstring("""
  S -> NP VP
  NP -> DT JJ NN | DT NN | PRP
  VP -> VBD NP | VBD PP | VBD
  PP -> IN NP
  DT -> 'the' | 'The' | 'a'
  JJ -> 'small'
  NN -> 'dog' | 'mailman'
  VBD -> 'barked' | 'chased'
  IN -> 'at'
  PRP -> 'he'
""")

# 2. Parse input sentence using Recursive Descent
sentence_tokens = "The small dog chased the mailman".split()
rd_parser = nltk.RecursiveDescentParser(toy_grammar)

print("--- CFG Parse Trees ---")
for tree in rd_parser.parse(sentence_tokens):
    print(tree)
```

### Output Hierarchical Parse Tree:
```
(S
  (NP (DT The) (JJ small) (NN dog))
  (VP (VBD chased) (NP (DT the) (NN mailman))))
```

### Visual Representation of Parse Tree:
```
                  S (Sentence)
          ┌───────┴───────────────────┐
          NP                          VP
   ┌──────┼─────────┐            ┌────┴────────┐
   DT     JJ        NN          VBD            NP
 (The) (small)    (dog)      (chased)     ┌────┴────┐
                                          DT        NN
                                        (the)   (mailman)
```

---

## 8. Comprehensive Comparison Tables

### Comparison 1: Stemming vs. Lemmatization

| Aspect | Stemming (`PorterStemmer`) | Lemmatization (`WordNetLemmatizer`) |
| :--- | :--- | :--- |
| **Approach** | Rule-based heuristic suffix slicing | Morphological analysis + dictionary lookup |
| **Dictionary Check** | ❌ No (blind string chopping) | ✅ Yes (checks WordNet database) |
| **Output Validity** | May produce non-words (`"amaz"`, `"quickli"`) | Always produces valid root lemmas (`"amaze"`, `"quickly"`) |
| **Speed** | Extremely fast ($\mathcal{O}(1)$ string ops) | Slower (requires dictionary search and POS disambiguation) |
| **POS Sensitivity** | Ignores POS context entirely | Uses POS tag (`pos='v'`, `pos='n'`) for accurate root |
| **Primary Use Cases** | Search indexing, Bag-of-Words, TF-IDF | Text analysis, Q&A, Chatbots, Semantic parsing |

---

### Comparison 2: Chunking vs. Full CFG Parsing

| Feature | Chunking (Shallow Parsing) | CFG Parsing (Full Syntactic Parsing) |
| :--- | :--- | :--- |
| **Output Depth** | Flat non-overlapping phrase brackets | Deep recursive hierarchical syntax tree |
| **Grammar Rigidity**| Flexible (captures local patterns via regex) | Strict (entire sentence must conform to production rules) |
| **Computational Cost**| Low ($\mathcal{O}(N)$ over POS sequence) | Higher ($\mathcal{O}(N^3)$ for chart/Earley parsers) |
| **Failure Mode** | Returns unchunked tokens if pattern doesn't match | Throws parsing failure / empty tree if grammar is violated |
| **Primary Use Cases**| Named Entity Recognition (NER), Information Extraction | Grammatical validation, Linguistics research, Compilers |

---

## 9. Summary Workflow Diagram

```
                        ┌──────────────────────────────┐
                        │        Raw Text Input        │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  Tokenization (Words/Sents)  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │ Stopword & Punctuation Clean │
                        └──────────────┬───────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │                                             │
                ▼                                             ▼
  ┌───────────────────────────┐                 ┌───────────────────────────┐
  │         Stemming          │                 │       Lemmatization       │
  │     (Porter Stemmer)      │                 │    (WordNet Lemmatizer)   │
  └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │     POS Tagging (pos_tag)    │
                        └──────────────┬───────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │                                             │
                ▼                                             ▼
  ┌───────────────────────────┐                 ┌───────────────────────────┐
  │   Shallow Chunk Parsing   │                 │    Hierarchical Parsing   │
  │  (RegexpParser - Noun/VP) │                 │  (CFG / RecursiveDescent) │
  └───────────────────────────┘                 └───────────────────────────┘
```
