# Getting started

## Install

```
pip uninstall -y rank-bm25 && pip install fitzyracing-rank-bm25
```

The import name is still `rank_bm25`. It depends on `numpy`.

## Usage

`from rank_bm25 import BM25Okapi, BM25L, BM25Plus`, then `bm25 = BM25Okapi(tokenized_corpus)` and `bm25.get_scores(query)` / `get_top_n`. See the README.

## Develop and test

```
pip install -e . pytest
pytest
```
