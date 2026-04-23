"""
Chunking ablation: fixed / semantic / hierarchical

Compares BM25 retrieval quality across three chunking strategies using 50 parenting
questions. Relevance is determined by keyword overlap between the query and retrieved
chunk (>= 2 content-bearing query terms must appear in the chunk).

Usage:
    uv run python experiments/run_chunking_ablation.py
"""
import csv
import json
import os
from pathlib import Path
from statistics import mean

import nltk
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunker
from src.ingestion.models import DocumentChunk

RAW_JSON_DIR = Path("data/raw/pubmed")
PROCESSED_DIR = Path("data/processed")
OUT_CSV = Path("experiments/chunking_ablation.csv")

STRATEGIES = ["fixed", "semantic", "heirarchial"]

# 25 vaccine / medical queries (golden dataset batch 1)
GOLDEN_QUERIES = [
    "What vaccines are recommended at the 2-month well-child visit?",
    "Is the MMR vaccine safe for infants?",
    "Can the DTaP vaccine cause fever in newborns?",
    "What is the recommended schedule for polio vaccination?",
    "When should a child receive their first hepatitis B vaccine?",
    "What are common side effects of the varicella vaccine?",
    "Does the flu vaccine protect children under 6 months?",
    "What is the Hib vaccine and when is it given?",
    "Are combination vaccines like MMRV safe for toddlers?",
    "How effective is the rotavirus vaccine in preventing diarrhea?",
    "What adverse events are associated with DTaP immunization?",
    "When is the meningococcal vaccine recommended for adolescents?",
    "Can vaccines cause autism in children?",
    "What vaccines are contraindicated in immunocompromised children?",
    "How does maternal vaccination protect newborns from pertussis?",
    "What is the efficacy of the pertussis vaccine in preventing whooping cough?",
    "Are there catch-up schedules for children who missed vaccines?",
    "Is the pneumococcal conjugate vaccine recommended for all children?",
    "What is the recommended HPV vaccine schedule for adolescents?",
    "How does the immune response to vaccines differ in premature infants?",
    "What are the risks of delaying the childhood vaccine schedule?",
    "Can the live attenuated influenza vaccine be given to children with asthma?",
    "What vaccines are given at the 4-month well-child visit?",
    "How does maternal antibody transfer affect vaccine efficacy in newborns?",
    "What is vaccine-preventable disease burden in children under 5?",
]

# 25 informal general parenting queries
INFORMAL_QUERIES = [
    "When do babies start sleeping through the night?",
    "What is the safest sleeping position for a newborn?",
    "When should I introduce solid foods to my baby?",
    "What are signs of RSV in a 3-month-old?",
    "How long should mothers breastfeed?",
    "What are developmental milestones for a 6-month-old?",
    "When do babies typically start crawling?",
    "What foods should be avoided in the first year of life?",
    "When should I call a doctor for infant fever?",
    "What makes a safe sleep environment for infants?",
    "How can I tell if my baby has colic?",
    "What is the recommended age for introducing peanut butter?",
    "How much tummy time does a newborn need each day?",
    "What are signs of dehydration in an infant?",
    "When do babies usually say their first words?",
    "How do I know if my baby has an ear infection?",
    "How much should a newborn sleep in the first week?",
    "What are the signs of a milk protein allergy in infants?",
    "When should children have their first dental visit?",
    "What is the appropriate screen time for children under 2?",
    "How can parents help a colicky baby stop crying?",
    "What are signs of developmental delays in a 12-month-old?",
    "How can I encourage healthy eating habits in toddlers?",
    "What should I do if my baby has a rash?",
    "How do I treat cradle cap in newborns?",
]

ALL_QUERIES = GOLDEN_QUERIES + INFORMAL_QUERIES

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "when", "how",
    "do", "does", "did", "for", "in", "of", "to", "and", "or", "my", "i",
    "if", "can", "should", "at", "with", "be", "by", "from", "that", "this",
}


def _ensure_nltk():
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def load_source_texts() -> list[tuple[str, str]]:
    texts = []
    for root, _, files in os.walk(RAW_JSON_DIR):
        for fname in files:
            if fname.endswith(".json"):
                with open(os.path.join(root, fname)) as f:
                    data = json.load(f)
                abstract = data.get("abstract", "").strip()
                if abstract:
                    texts.append((fname, abstract))
    if PROCESSED_DIR.exists():
        for fname in os.listdir(PROCESSED_DIR):
            if fname.endswith((".md", ".txt")):
                with open(PROCESSED_DIR / fname) as f:
                    text = f.read().strip()
                if text:
                    texts.append((fname, text))
    return texts


def build_corpus(strategy: str, texts: list[tuple[str, str]]) -> list[DocumentChunk]:
    chunker = Chunker(method=strategy)
    chunks = []
    for source, text in texts:
        chunks.extend(chunker.chunk_by_method(text, source=source))
    return chunks


def is_relevant(chunk_text: str, query: str) -> bool:
    """A chunk is relevant when >= 2 content-bearing query terms appear in it."""
    terms = {
        w.lower().rstrip("?.,") for w in query.split()
        if w.lower().rstrip("?.,") not in STOPWORDS and len(w) > 3
    }
    chunk_lower = chunk_text.lower()
    return sum(1 for t in terms if t in chunk_lower) >= 2


def compute_mrr(top_chunks: list[str], query: str) -> float:
    for rank, chunk in enumerate(top_chunks, 1):
        if is_relevant(chunk, query):
            return 1.0 / rank
    return 0.0


def compute_hit_at_5(top_chunks: list[str], query: str) -> float:
    return float(any(is_relevant(c, query) for c in top_chunks[:5]))


def evaluate_strategy(strategy: str, texts: list[tuple[str, str]]) -> dict:
    print(f"  Building {strategy} corpus...", end=" ", flush=True)
    chunks = build_corpus(strategy, texts)
    print(f"{len(chunks)} chunks", flush=True)

    corpus_tokens = [c.content.lower().split() for c in chunks]
    index = BM25Okapi(corpus_tokens)
    avg_words = mean(len(c.content.split()) for c in chunks)

    mrr_scores, hit_scores = [], []
    for query in ALL_QUERIES:
        tokens = query.lower().split()
        scores = index.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        top_chunks = [chunks[i].content for i in top_idx]
        mrr_scores.append(compute_mrr(top_chunks, query))
        hit_scores.append(compute_hit_at_5(top_chunks, query))

    return {
        "strategy": strategy,
        "mrr": round(mean(mrr_scores), 4),
        "hit_at_5": round(mean(hit_scores), 4),
        "avg_chunk_size_words": round(avg_words, 1),
        "num_chunks": len(chunks),
    }


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 60)
    print(f"{'Strategy':<16} {'MRR':>8} {'Hit@5':>8} {'AvgWords':>10} {'Chunks':>8}")
    print("-" * 60)
    for r in rows:
        print(
            f"{r['strategy']:<16} {r['mrr']:>8.4f} {r['hit_at_5']:>8.4f}"
            f" {r['avg_chunk_size_words']:>10.1f} {r['num_chunks']:>8}"
        )
    print("=" * 60)
    best_mrr = max(rows, key=lambda r: r["mrr"])
    best_hit = max(rows, key=lambda r: r["hit_at_5"])
    print(f"Best MRR   → {best_mrr['strategy']} ({best_mrr['mrr']})")
    print(f"Best Hit@5 → {best_hit['strategy']} ({best_hit['hit_at_5']})")


def main():
    _ensure_nltk()
    texts = load_source_texts()
    print(f"Loaded {len(texts)} source documents ({len(ALL_QUERIES)} queries)\n")

    rows = []
    for strategy in STRATEGIES:
        print(f"[{strategy}]")
        row = evaluate_strategy(strategy, texts)
        rows.append(row)
        print(f"  MRR={row['mrr']:.4f}  Hit@5={row['hit_at_5']:.4f}\n")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["strategy", "mrr", "hit_at_5", "avg_chunk_size_words", "num_chunks"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results written to {OUT_CSV}")
    print_summary(rows)


if __name__ == "__main__":
    main()
