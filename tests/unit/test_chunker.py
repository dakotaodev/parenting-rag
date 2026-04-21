import pytest
from src.ingestion.chunker import Chunker

SAMPLE_TEXT = " ".join(["This is a sentence about infant sleep safety." * 10] * 30)


def test_hierarchical_every_child_has_valid_parent():
    chunker = Chunker(method="heirarchial")
    chunks = chunker.chunk_by_method(SAMPLE_TEXT, source="test.txt")

    parent_ids = {c.chunk_id for c in chunks if c.chunk_id and c.chunk_id == c.parent_id}
    children = [c for c in chunks if c.parent_id and c.chunk_id != c.parent_id]

    assert children, "Expected at least one child chunk"
    for child in children:
        assert child.parent_id in parent_ids, (
            f"Child {child.chunk_id!r} references missing parent {child.parent_id!r}"
        )


def test_hierarchical_parents_have_no_foreign_parent():
    chunker = Chunker(method="heirarchial")
    chunks = chunker.chunk_by_method(SAMPLE_TEXT, source="test.txt")

    parents = [c for c in chunks if c.chunk_id and c.chunk_id == c.parent_id]
    assert parents, "Expected at least one parent chunk"


def test_hierarchical_chunk_ids_are_unique():
    chunker = Chunker(method="heirarchial")
    chunks = chunker.chunk_by_method(SAMPLE_TEXT, source="test.txt")

    ids = [c.chunk_id for c in chunks if c.chunk_id]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"


def test_hierarchical_source_propagated():
    chunker = Chunker(method="heirarchial")
    chunks = chunker.chunk_by_method(SAMPLE_TEXT, source="my_doc.txt")

    assert all(c.source == "my_doc.txt" for c in chunks)


def test_fixed_chunks_have_no_parent_id():
    chunker = Chunker(method="fixed")
    chunks = chunker.chunk_by_method(SAMPLE_TEXT, source="test.txt")

    assert chunks, "Expected chunks"
    assert all(c.parent_id is None for c in chunks)
    assert all(c.chunk_id is None for c in chunks)


def test_semantic_chunks_have_no_parent_id():
    chunker = Chunker(method="semantic")
    chunks = chunker.chunk_by_method(SAMPLE_TEXT, source="test.txt")

    assert chunks, "Expected chunks"
    assert all(c.parent_id is None for c in chunks)
