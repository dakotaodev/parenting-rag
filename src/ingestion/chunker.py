from langchain_text_splitters import MarkdownTextSplitter
import os 
import json
from nltk import sent_tokenize
import nltk
from typing import Literal
import structlog

from src.ingestion.models import DocumentChunk

ChunkMethod = Literal["fixed","semantic"]
logger = structlog.get_logger(__name__)

class Chunker:

    def __init__(self, method: ChunkMethod, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.method = method
        self.splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nltk.download("punkt")

    # create method that applies chunking to all files in a directory
    def chunk_pdf_directory(self, path: str) -> list[DocumentChunk]:
        results=[] 
        for file in os.listdir(path):
            if file.endswith(".md") or file.endswith(".txt"):
                file_path = os.path.join(path, file)
                with open(file_path, "r") as f:
                    text = f.read()
                chunks = self.chunk_by_mehtod(text)
                doc_chunks = [DocumentChunk(source=file, content=chunk) for chunk in chunks]
                results.extend(doc_chunks)
        return results
    
    def chunk_json_directory(self, path: str):
        results=[] 
        # for each file in path and its subdirectories, if file ends with .json, read it and chunk it
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        # load json and extract the abstract field
                        data = json.load(f)
                        text = data.get("abstract", "")
                chunks = self.chunk_by_mehtod(text)
                doc_chunks = [DocumentChunk(source=file, content=chunk) for chunk in chunks]
                results.extend(doc_chunks)
        return results
    
    def chunk_by_mehtod(self, text:str) -> list[str]:
        if self.method == "fixed":
            return self._chunk_fixed_size(text)
        elif self.method == "semantic":
            return self._chunk_semantic(text)
        else:
            return ValueError(f"Error: the specified method is not supported: {self.method}")

    def _chunk_fixed_size(self, text: str):
        return self.splitter.split_text(text)
    
    def _chunk_semantic(self, text:str) -> list[str]:
        chunks = sent_tokenize(text) 
        avg = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
        logger.info("semantic chunk results", total=len(chunks), avg=avg)
        return chunks