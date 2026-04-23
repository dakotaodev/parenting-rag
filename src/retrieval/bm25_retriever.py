import os
import pickle
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from supabase import create_client

load_dotenv()

INDEX_PATH = Path("data/bm25_index.pkl")
_PAGE_SIZE = 1000


class BM25Retriever:
    def __init__(self, force_rebuild: bool = False):
        if not force_rebuild and INDEX_PATH.exists():
            self._load()
        else:
            corpus, chunk_ids = self._fetch_corpus()
            self._build(corpus, chunk_ids)
            self._save()

    def _fetch_corpus(self) -> tuple[list[list[str]], list[str]]:
        client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        texts: list[list[str]] = []
        chunk_ids: list[str] = []
        offset = 0
        while True:
            response = (
                client.table("documents")
                .select("content, metadata")
                .range(offset, offset + _PAGE_SIZE - 1)
                .execute()
            )
            rows = response.data
            if not rows:
                break
            for row in rows:
                chunk_id = (row.get("metadata") or {}).get("chunk_id") or ""
                texts.append(row["content"].lower().split())
                chunk_ids.append(chunk_id)
            if len(rows) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        return texts, chunk_ids

    def _build(self, corpus: list[list[str]], chunk_ids: list[str]) -> None:
        self.chunk_ids = chunk_ids
        self.index = BM25Okapi(corpus)

    def _save(self) -> None:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump((self.index, self.chunk_ids), f)

    def _load(self) -> None:
        with open(INDEX_PATH, "rb") as f:
            self.index, self.chunk_ids = pickle.load(f)

    def retrieve(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        tokens = query.lower().split()
        scores = self.index.get_scores(tokens)
        top_k = np.argsort(scores)[::-1][:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_k]
