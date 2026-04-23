import pickle

import pytest

from src.retrieval.bm25_retriever import BM25Retriever

# Corpus mixing vaccine-related and unrelated content
CORPUS = [
    ("chunk-mmr", "MMR vaccine measles mumps rubella immunization safety children efficacy"),
    ("chunk-dtap", "DTaP diphtheria tetanus pertussis vaccination schedule infants booster"),
    ("chunk-sleep", "infant sleep safety back to sleep SIDS prevention crib mattress"),
    ("chunk-feeding", "breastfeeding benefits nutrition newborn feeding latch colostrum"),
    ("chunk-polio", "polio IPV inactivated poliovirus vaccine dose schedule eradication"),
    ("chunk-dev", "childhood development milestones cognitive motor skills language growth"),
    ("chunk-varicella", "varicella chickenpox vaccine efficacy adverse effects shingles zoster"),
    ("chunk-fever", "fever management acetaminophen ibuprofen pediatric dosing temperature"),
    ("chunk-hepb", "hepatitis B vaccine HBV newborn birth dose immune protection"),
    ("chunk-solid", "solid food introduction baby puree allergen timing six months"),
]


def _build_retriever(corpus: list[tuple[str, str]]) -> BM25Retriever:
    retriever = BM25Retriever.__new__(BM25Retriever)
    tokenized = [text.lower().split() for _, text in corpus]
    chunk_ids = [chunk_id for chunk_id, _ in corpus]
    retriever._build(tokenized, chunk_ids)
    return retriever


def test_retrieve_returns_list_of_tuples():
    retriever = _build_retriever(CORPUS)
    results = retriever.retrieve("MMR vaccine")
    assert isinstance(results, list)
    assert all(
        isinstance(chunk_id, str) and isinstance(score, float)
        for chunk_id, score in results
    )


def test_retrieve_respects_k():
    retriever = _build_retriever(CORPUS)
    assert len(retriever.retrieve("vaccine", k=3)) == 3
    assert len(retriever.retrieve("vaccine", k=5)) == 5


def test_mmr_query_ranks_relevant_chunk_in_top_3():
    retriever = _build_retriever(CORPUS)
    results = retriever.retrieve("MMR measles vaccine", k=3)
    top_ids = [chunk_id for chunk_id, _ in results]
    assert "chunk-mmr" in top_ids


def test_dtap_query_ranks_relevant_chunk_in_top_3():
    retriever = _build_retriever(CORPUS)
    results = retriever.retrieve("DTaP pertussis vaccination infants", k=3)
    top_ids = [chunk_id for chunk_id, _ in results]
    assert "chunk-dtap" in top_ids


def test_varicella_query_ranks_relevant_chunk_in_top_3():
    retriever = _build_retriever(CORPUS)
    results = retriever.retrieve("varicella chickenpox vaccine", k=3)
    top_ids = [chunk_id for chunk_id, _ in results]
    assert "chunk-varicella" in top_ids


def test_polio_query_ranks_relevant_chunk_in_top_3():
    retriever = _build_retriever(CORPUS)
    results = retriever.retrieve("polio IPV poliovirus vaccine", k=3)
    top_ids = [chunk_id for chunk_id, _ in results]
    assert "chunk-polio" in top_ids


def test_hepatitis_b_query_ranks_relevant_chunk_in_top_3():
    retriever = _build_retriever(CORPUS)
    results = retriever.retrieve("hepatitis B HBV vaccine newborn", k=3)
    top_ids = [chunk_id for chunk_id, _ in results]
    assert "chunk-hepb" in top_ids


def test_serialization_roundtrip(tmp_path):
    retriever = _build_retriever(CORPUS)
    index_path = tmp_path / "bm25_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump((retriever.index, retriever.chunk_ids), f)

    loaded = BM25Retriever.__new__(BM25Retriever)
    with open(index_path, "rb") as f:
        loaded.index, loaded.chunk_ids = pickle.load(f)

    assert retriever.retrieve("MMR vaccine", k=3) == loaded.retrieve("MMR vaccine", k=3)


def test_scores_are_non_negative():
    retriever = _build_retriever(CORPUS)
    results = retriever.retrieve("vaccine schedule", k=len(CORPUS))
    assert all(score >= 0.0 for _, score in results)
