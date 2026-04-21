from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
import os 
import json
from nltk import sent_tokenize
import nltk
from typing import Literal
import structlog

from src.ingestion.models import DocumentChunk

ChunkMethod = Literal["fixed", "semantic", "heirarchial"]
logger = structlog.get_logger(__name__)

class Chunker:

    def __init__(self, method: ChunkMethod):
        self.method = method
        nltk.download("punkt")

    def chunk_pdf_directory(self, path: str) -> list[DocumentChunk]:
        results = []
        for file in os.listdir(path):
            if file.endswith(".md") or file.endswith(".txt"):
                file_path = os.path.join(path, file)
                with open(file_path, "r") as f:
                    text = f.read()
                results.extend(self.chunk_by_method(text=text, source=file))
        return results

    def chunk_json_directory(self, path: str) -> list[DocumentChunk]:
        results = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        text = data.get("abstract", "")
                    if not text:
                        continue
                    results.extend(self.chunk_by_method(text=text, source=file))
        return results

    def chunk_by_method(self, text: str, source: str) -> list[DocumentChunk]:
        if self.method == "fixed":
            return [DocumentChunk(source=source, content=c) for c in self._chunk_fixed_size(text)]
        elif self.method == "semantic":
            return [DocumentChunk(source=source, content=c) for c in self._chunk_semantic(text)]
        elif self.method == "heirarchial":
            return self._chunk_heirchial(text, source)
        else:
            raise ValueError(f"Unsupported chunking method: {self.method}")

    def _chunk_fixed_size(self, text: str):
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)
    
    def _chunk_semantic(self, text:str) -> list[str]:
        chunks = sent_tokenize(text) 
        avg = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
        logger.info("semantic chunk results", total=len(chunks), avg=avg)
        return chunks
    
    def _chunk_heirchial(self, text: str, source: str):
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=0)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)

        parents: list[str] = parent_splitter.split_text(text)
        results=[]
        for i, parent in enumerate(parents):
            parent_id=f"{source}:parent:{i}"
            results.append(
                DocumentChunk(
                    source=source,
                    content=parent,
                    parent_id=parent_id,
                    chunk_id=parent_id
                )
            )

            children = child_splitter.split_text(parent)
            for j, child in enumerate(children):
                results.append(
                    DocumentChunk(
                        source=source,
                        content=child,
                        chunk_id=f"{source}:child:{i}:{j}",
                        parent_id=parent_id,
                    )
                )
        return results