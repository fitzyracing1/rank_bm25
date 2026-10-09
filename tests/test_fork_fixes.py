"""Regression tests for the fitzyracing-rank-bm25 fork (360 Bench).

Every test in the first two sections fails on rank-bm25 0.2.2 and passes on the fork.
The last section pins scores that must stay exactly as in 0.2.2.
"""
import math

import numpy as np
import pytest

from rank_bm25 import BM25, BM25L, BM25Okapi, BM25Plus

ALGS = [BM25Okapi, BM25L, BM25Plus]


# --- Empty corpus: clear error instead of ZeroDivisionError (upstream #36, PR #45) ---

@pytest.mark.parametrize("alg", ALGS + [BM25])
def test_empty_corpus_raises_clear_error(alg):
    from rank_bm25 import EmptyCorpusException

    with pytest.raises(EmptyCorpusException, match="empty corpus"):
        alg([])


def test_empty_corpus_error_is_still_a_zero_division_error():
    # Code that caught the old ZeroDivisionError keeps working.
    from rank_bm25 import EmptyCorpusException

    assert issubclass(EmptyCorpusException, ZeroDivisionError)
    assert issubclass(EmptyCorpusException, ValueError)
    with pytest.raises(ZeroDivisionError):
        BM25Okapi([])


def test_empty_iterable_corpus_raises_clear_error():
    # Iterables (not only lists) are supported upstream since #17.
    from rank_bm25 import EmptyCorpusException

    with pytest.raises(EmptyCorpusException):
        BM25Okapi(doc for doc in [])


def test_empty_corpus_with_tokenizer_raises_clear_error():
    from rank_bm25 import EmptyCorpusException

    with pytest.raises(EmptyCorpusException):
        BM25Okapi([], tokenizer=str.split)


# --- BM25Okapi: term in exactly half of the documents (upstream #39, #43) ---

ISSUE_39_CORPUS = [
    "This text contains keyword1 and Keyword2",
    "That is a text that contains keyword1 and term1",
    "Page contains no keywords but contains term1 and term2",
    "This text contains no keywords",
]
ISSUE_39_QUERY = "This is a question about keyword1 & term1".split()


def test_half_corpus_term_gets_epsilon_floor_not_zero():
    bm25 = BM25Okapi([d.split() for d in ISSUE_39_CORPUS])
    eps = bm25.epsilon * bm25.average_idf
    assert eps > 0
    # keyword1, term1 and This are each in exactly 2 of 4 documents.
    for word in ("keyword1", "term1", "This"):
        assert bm25.idf[word] == eps
    # Same value as a term in more than half of the documents ("contains": 4 of 4).
    assert bm25.idf["contains"] == eps


def test_issue_39_matching_documents_score_above_zero():
    bm25 = BM25Okapi([d.split() for d in ISSUE_39_CORPUS])
    scores = bm25.get_scores(ISSUE_39_QUERY)
    # 0.2.2 returned [0, 1.5286, 0, 0]: every document matches a query term.
    assert all(s > 0 for s in scores)
    # Doc 1 matches the rare terms "is" and "a" and stays on top.
    assert int(np.argmax(scores)) == 1
    assert bm25.get_batch_scores(ISSUE_39_QUERY, [0, 1, 2, 3]) == pytest.approx(list(scores))


def test_half_corpus_term_ranks_matching_document_first():
    # "red" is in exactly 3 of 6 documents; 0.2.2 scored every document 0 for it.
    corpus = [["red", "apple"], ["green", "apple"], ["red", "car"], ["blue", "car"],
              ["red", "bus"], ["pink", "bike"]]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(["red"])
    assert [i for i, s in enumerate(scores) if s > 0] == [0, 2, 4]


# --- Unchanged behaviour (these pass on 0.2.2 too) ---

README_CORPUS = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?",
]


def test_readme_example_scores_unchanged():
    bm25 = BM25Okapi([doc.split(" ") for doc in README_CORPUS])
    assert bm25.get_scores(["windy", "London"]) == pytest.approx([0.0, 0.93729472, 0.0])


def test_scores_without_half_corpus_terms_are_identical_to_0_2_2():
    # Odd corpus size, so no term can be in exactly half the documents.
    corpus = [d.split() for d in ISSUE_39_CORPUS[:3]]
    bm25 = BM25Okapi(corpus)
    expected = np.array([0.5790074755817026, 0.9730066453562337, 0.0028814303552854763])
    assert np.array_equal(bm25.get_scores(ISSUE_39_QUERY), expected)


def test_two_document_corpus_unchanged():
    # With 2 documents the average idf is <= 0, so the epsilon floor is not positive and
    # a half-corpus term keeps idf 0, exactly as in 0.2.2 (giving it a negative floor would
    # rank matching documents below non-matching ones). See README.
    bm25 = BM25Okapi([["a", "b", "x"], ["c", "d", "x"]])
    assert bm25.idf["a"] == 0.0
    assert bm25.idf["x"] == pytest.approx(-0.08047189562170502)
    assert list(bm25.get_scores(["a"])) == [0.0, 0.0]


@pytest.mark.parametrize("alg", [BM25L, BM25Plus])
def test_other_algorithms_unchanged(alg):
    corpus = [d.split() for d in ISSUE_39_CORPUS]
    a = alg(corpus)
    b = alg(corpus)
    # idf formula untouched: check one value directly.
    n = 2  # keyword1
    if alg is BM25L:
        assert a.idf["keyword1"] == math.log(len(corpus) + 1) - math.log(n + 0.5)
    else:
        assert a.idf["keyword1"] == math.log((len(corpus) + 1) / n)
    assert np.array_equal(a.get_scores(ISSUE_39_QUERY), b.get_scores(ISSUE_39_QUERY))
