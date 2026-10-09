# Changelog

## 0.2.3 (2026-10-09, fitzyracing-rank-bm25 fork)

Based on rank-bm25 0.2.2. Distribution name `fitzyracing-rank-bm25`; the import name is still `rank_bm25`.

- An empty corpus now raises `rank_bm25.EmptyCorpusException` ("Cannot build a BM25 index from an
  empty corpus") instead of a bare `ZeroDivisionError` (upstream #36, PR #45). It subclasses
  `ValueError` and `ZeroDivisionError`, so existing handlers still catch it.
- `BM25Okapi`: a term that occurs in exactly half of the documents gets the existing
  `epsilon * average_idf` floor instead of an idf of 0, when that floor is positive (upstream #39,
  #43). **Score change:** only `BM25Okapi`, only for even-sized corpora, only for query terms in
  exactly N/2 documents, and only upward. All other scores are identical to 0.2.2; `epsilon=0`
  restores 0.2.2 exactly.
- Packaging: `pyproject.toml` replaces `setup.py` and `version.py`, so the sdist builds under
  PEP 517 (upstream #56, PR #57). Python 3.8+.
- Tests: add `tests/test_fork_fixes.py`. CI: test on Python 3.9 to 3.14; remove the upstream
  publish workflow.

## 0.2.2 (2022-02-16) and earlier

See the [upstream releases](https://github.com/dorianbrown/rank_bm25/releases) and
[history](https://github.com/dorianbrown/rank_bm25/commits/master).
