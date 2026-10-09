# fitzyracing-rank-bm25 (Rank-BM25)

> Part of **[360 Bench](https://github.com/fitzyracing1/360-bench)**, tested fixes for abandoned PyPI packages.

Rank-BM25: a two line search engine.

[![PyPI](https://img.shields.io/pypi/v/fitzyracing-rank-bm25.svg)](https://pypi.org/project/fitzyracing-rank-bm25/)

> **This is a fork of [rank_bm25](https://github.com/dorianbrown/rank_bm25)
> by [Dorian Brown](https://github.com/dorianbrown)** and its contributors, published as a
> drop-in replacement. The upstream package (~8.7M downloads a month, and the engine behind
> LangChain's `BM25Retriever`) has not been released since 0.2.2 (February 2022), and the
> fixes below have sat in open issues and pull requests since 2023–2024. The import name is
> still `rank_bm25`, so no code changes are needed. All credit for the original library goes
> to its author and contributors; it remains available under the same Apache-2.0 license
> (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). To cite it, use the upstream
> [CITATION](CITATION).
>
> If a fixed `rank-bm25` release appears on PyPI, prefer it and switch back.


## What's fixed in this fork

Based on upstream `rank-bm25==0.2.2` (tag [`0.2.2`](https://github.com/dorianbrown/rank_bm25/tree/0.2.2)).
The API is unchanged. Scores are identical to 0.2.2 except in the one case described under
"Half-corpus terms" below.

- **Empty corpus: a clear error instead of `ZeroDivisionError: division by zero`.**
  `BM25Okapi([])` (and `BM25L`, `BM25Plus`, also with a tokenizer or an empty iterator)
  crashed deep inside `_initialize`, which is hard to understand when rank_bm25 sits a few
  layers down, for example in a RAG pipeline whose retriever got no documents. It now raises
  `rank_bm25.EmptyCorpusException: Cannot build a BM25 index from an empty corpus`, as the
  maintainer proposed in [#36](https://github.com/dorianbrown/rank_bm25/issues/36) and
  PR [#45](https://github.com/dorianbrown/rank_bm25/pull/45). The exception subclasses both
  `ValueError` and `ZeroDivisionError`, so existing `except ZeroDivisionError:` code still
  catches it. (Unlike #45 it is defined in `rank_bm25` itself rather than in a new top-level
  `exceptions` module, and it also works for iterable corpora, which `len(corpus)` would not.)

- **Half-corpus terms: `BM25Okapi` no longer gives a term in exactly half the documents an
  idf of 0.** `BM25Okapi` uses idf = log((N − n + 0.5) / (n + 0.5)), which is exactly 0 when
  a term is in n = N/2 of the N documents, and negative when it is in more than half. Negative
  values are replaced by the floor `epsilon * average_idf`, but 0 was not, so a term in
  exactly half the documents counted for *less* than a term in every document, and documents
  that matched it scored 0
  ([#39](https://github.com/dorianbrown/rank_bm25/issues/39),
  [#43](https://github.com/dorianbrown/rank_bm25/issues/43)):

  ```python
  corpus = ["This text contains keyword1 and Keyword2",
            "That is a text that contains keyword1 and term1",
            "Page contains no keywords but contains term1 and term2",
            "This text contains no keywords"]
  BM25Okapi([d.split() for d in corpus]).get_scores("This is a question about keyword1 & term1".split())
  # 0.2.2: array([0.        , 1.52856224, 0.        , 0.        ])
  # fork:  array([0.09779242, 1.60992918, 0.04068347, 0.05242379])
  ```

  The fork gives these terms the same `epsilon * average_idf` floor that terms in more than
  half the documents already get. **Exactly which scores change:**
  - only `BM25Okapi` (`BM25L` and `BM25Plus` are untouched);
  - only for corpora with an even number of documents N, and only for query terms that occur
    in exactly N/2 of them;
  - only when the floor `epsilon * average_idf` is positive (with the default `epsilon=0.25`,
    when the average idf of the vocabulary is positive, which is the normal case). If it is 0
    or negative, which happens in tiny corpora such as 2 documents, these terms keep idf 0,
    exactly as before, because a negative floor would rank matching documents *below*
    non-matching ones;
  - in those cases, documents containing such a term gain `epsilon * average_idf × (the usual
    BM25 term-frequency factor)` per occurrence in the query; no score ever goes down, and
    `bm25.average_idf` is unchanged. Passing `epsilon=0` restores 0.2.2's results exactly.

  A randomized comparison of 20,000 small corpora and queries against 0.2.2 (all three
  algorithms) found no other differences.

  **What this fork deliberately does not change:** PR
  [#40](https://github.com/dorianbrown/rank_bm25/pull/40) switches `BM25Okapi` to a different
  idf formula, log((N + 1) / (n + 0.5)). That removes the problem for every corpus size,
  including 2 documents, but changes *every* `BM25Okapi` score, so it is not a drop-in fix. In
  a 2-document corpus a term in one document therefore still scores 0 here; if you need
  that case, use `BM25Plus` or `BM25L`, whose idf is always positive.

- **Builds from source again.** `setup.py` imported a local `version.py` that wasn't shipped in
  the sdist, so building from the sdist under PEP 517 failed
  ([#56](https://github.com/dorianbrown/rank_bm25/issues/56),
  PR [#57](https://github.com/dorianbrown/rank_bm25/pull/57)). The fork uses a static
  `pyproject.toml`.

Left out on purpose, to keep the fork a drop-in replacement:
- Upstream `master` has two commits after 0.2.2 that were never released:
  [#20](https://github.com/dorianbrown/rank_bm25/pull/20) removes a `q_freq` factor from the
  `BM25L` score (a real correction, but it changes every `BM25L` score) and
  [#23](https://github.com/dorianbrown/rank_bm25/pull/23) rewrites one `log()` in `BM25Plus`
  (can change the last digit of scores). Neither is included.
- PR [#58](https://github.com/dorianbrown/rank_bm25/pull/58) (apply the constructor tokenizer
  to string queries in `get_top_n`, [#38](https://github.com/dorianbrown/rank_bm25/issues/38))
  is a behaviour change the maintainer wanted to hold back on, so it is not included. Tokenize
  your queries yourself, as the README below says.


## Install

```bash
pip install fitzyracing-rank-bm25
```

Python 3.8+; the only dependency is numpy, as before.

### Switching from `rank-bm25`

This distribution installs the same `rank_bm25` module as the original, so your code keeps
doing `from rank_bm25 import BM25Okapi`. **Uninstall the original first**, then install the
fork:

```bash
pip uninstall -y rank-bm25
pip install fitzyracing-rank-bm25
```

The order matters. Both distributions own the same `rank_bm25.py`, so if you install the fork
first and uninstall `rank-bm25` afterwards, pip deletes the shared file and the import breaks.
If that happens, run `pip install --force-reinstall --no-deps fitzyracing-rank-bm25`.

In `requirements.txt` / `pyproject.toml`, replace `rank-bm25` (or `rank_bm25`) with
`fitzyracing-rank-bm25`.

### If you get rank-bm25 through another package

pip cannot replace a dependency with a differently named package. If a dependency requires
`rank-bm25`, install the fork alongside it and then remove the original's files:

```bash
pip install fitzyracing-rank-bm25
pip uninstall -y rank-bm25
pip install --force-reinstall --no-deps fitzyracing-rank-bm25   # restore the file the uninstall removed
```

Afterwards `pip check` reports `<package> requires rank-bm25, which is not installed`; that is
expected. Repeat the steps if a later install pulls `rank-bm25` back in.

**uv users** can do this properly with an override that drops the original:

```toml
# pyproject.toml
[project]
dependencies = ["fitzyracing-rank-bm25", "...the package that depends on rank-bm25..."]

[tool.uv]
override-dependencies = ["rank-bm25; sys_platform == 'never'"]
```

(or `uv pip install --override overrides.txt ...` with that same line in `overrides.txt`).

### Tests

```bash
pip install -e ".[dev]"
pytest
```

`tests/test_fork_fixes.py` fails on rank-bm25 0.2.2 and passes on this fork; it also pins
scores that must stay identical to 0.2.2.


## About Rank-BM25

A collection of algorithms for querying a set of documents and returning the ones most relevant to the query. The most common use case for these algorithms is, as you might have guessed, to create search engines.

So far the algorithms that have been implemented are:
- [x] Okapi BM25
- [x] BM25L
- [x] BM25+
- [ ] BM25-Adpt
- [ ] BM25T 

These algorithms were taken from [this paper](http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf), which gives a nice overview of each method, and also benchmarks them against each other. A nice inclusion is that they compare different kinds of preprocessing like stemming vs no-stemming, stopword removal or not, etc. Great read if you're new to the topic.

## Usage
For this example we'll be using the `BM25Okapi` algorithm, but the others are used in pretty much the same way.

### Initalizing

First thing to do is create an instance of the BM25 class, which reads in a corpus of text and does some indexing on it:
```python
from rank_bm25 import BM25Okapi

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]

tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)
# <rank_bm25.BM25Okapi at 0x1047881d0>
```
Note that this package doesn't do any text preprocessing. If you want to do things like lowercasing, stopword removal, stemming, etc, you need to do it yourself. 

The only requirements is that the class receives a list of lists of strings, which are the document tokens.

### Ranking of documents

Now that we've created our document indexes, we can give it queries and see which documents are the most relevant:
```python
query = "windy London"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)
# array([0.        , 0.93729472, 0.        ])
```
Good to note that we also need to tokenize our query, and apply the same preprocessing steps we did to the documents in order to have an apples-to-apples comparison

Instead of getting the document scores, you can also just retrieve the best documents with
```python
bm25.get_top_n(tokenized_query, corpus, n=1)
# ['It is quite windy in London']
```
And that's pretty much it!
