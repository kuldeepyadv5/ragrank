"""PDF/text extraction and semantic chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from src.config import get_config


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    page: int | None = None
    metadata: dict = field(default_factory=dict)


class TextProcessor:
    """Extract text from files and split into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        cfg = get_config().retriever
        self.chunk_size = chunk_size or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap

    def extract_text(self, file_path: Path | str) -> tuple[str, list[tuple[int, str]]]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(path)
        if suffix in {".txt", ".md", ".csv"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return text, [(1, text)]
        raise ValueError(f"Unsupported file type: {suffix}")

    def extract_from_bytes(
        self, data: bytes, filename: str
    ) -> tuple[str, list[tuple[int, str]]]:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf_bytes(data)
        if suffix in {".txt", ".md", ".csv"}:
            text = data.decode("utf-8", errors="ignore")
            return text, [(1, text)]
        raise ValueError(f"Unsupported file type: {suffix}")

    def _extract_pdf(self, path: Path) -> tuple[str, list[tuple[int, str]]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("Install pypdf for PDF support: pip install pypdf") from exc

        reader = PdfReader(str(path))
        pages: list[tuple[int, str]] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
        full_text = "\n".join(text for _, text in pages)
        return full_text, pages

    def _extract_pdf_bytes(self, data: bytes) -> tuple[str, list[tuple[int, str]]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("Install pypdf for PDF support: pip install pypdf") from exc

        import io

        reader = PdfReader(io.BytesIO(data))
        pages: list[tuple[int, str]] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
        full_text = "\n".join(text for _, text in pages)
        return full_text, pages

    def chunk_text(
        self,
        text: str,
        source: str,
        pages: list[tuple[int, str]] | None = None,
    ) -> list[DocumentChunk]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        chunk_index = 0

        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            chunk_text = normalized[start:end].strip()
            if chunk_text:
                page_num = self._estimate_page(chunk_text, pages) if pages else None
                chunk_id = hashlib.md5(
                    f"{source}:{chunk_index}:{chunk_text[:64]}".encode()
                ).hexdigest()[:12]
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=source,
                        page=page_num,
                        metadata={"chunk_index": chunk_index},
                    )
                )
                chunk_index += 1
            if end >= len(normalized):
                break
            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def process_file(self, file_path: Path | str) -> list[DocumentChunk]:
        path = Path(file_path)
        _, pages = self.extract_text(path)
        full_text = "\n".join(text for _, text in pages)
        return self.chunk_text(full_text, source=path.name, pages=pages)

    def process_upload(self, file_obj: BinaryIO, filename: str) -> list[DocumentChunk]:
        data = file_obj.read()
        _, pages = self.extract_from_bytes(data, filename)
        full_text = "\n".join(text for _, text in pages)
        return self.chunk_text(full_text, source=filename, pages=pages)

    @staticmethod
    def _estimate_page(chunk: str, pages: list[tuple[int, str]]) -> int | None:
        for page_num, page_text in pages:
            if chunk[:80] in re.sub(r"\s+", " ", page_text):
                return page_num
        return pages[0][0] if pages else None
