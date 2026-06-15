"""Retriever-focused tests."""

from src.retriever.text_processor import TextProcessor


def test_chunk_overlap_produces_multiple_chunks():
    processor = TextProcessor(chunk_size=30, chunk_overlap=5)
    text = " ".join(["word"] * 50)
    chunks = processor.chunk_text(text, source="test.txt")
    assert len(chunks) > 1
