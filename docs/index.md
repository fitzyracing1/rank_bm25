# fitzyracing-rank-bm25

`fitzyracing-rank-bm25` is a drop-in fork of [rank_bm25](https://github.com/dorianbrown/rank_bm25) and part of [360 Bench](https://github.com/fitzyracing1/360-bench). It is based on upstream `rank-bm25==0.2.2`, and the API is unchanged. All credit for the original library goes to its authors and contributors.

## What the fork fixes (summary of [README.md](../README.md))

- An empty corpus raises `rank_bm25.EmptyCorpusException` (a subclass of both `ValueError` and `ZeroDivisionError`) instead of `ZeroDivisionError`.
- `BM25Okapi` no longer gives a term that appears in exactly half the documents an idf of 0.
- Builds from source again.

The full details and links to the upstream issues are in [README.md](../README.md).

Pages: [Getting started](getting-started.md) · [Architecture](architecture.md) · [Contributing](contributing.md)

## CI

`.github/workflows/pytest.yml` runs the test suite. A run is **lit** when it succeeds and **dark** otherwise.

## License

Apache-2.0 (see LICENSE and NOTICE). To cite it, use the upstream CITATION. The original notice is kept: Copyright Dorian Brown and the rank_bm25 contributors.
Modifications: Copyright (c) 2026 Joshua Almeida.
