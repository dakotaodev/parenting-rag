import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import APIRouter, Request
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langsmith import get_current_run_tree, traceable
from pydantic import BaseModel, Field
from supabase import Client, create_client

load_dotenv()


@lru_cache(maxsize=1)
def _supabase() -> Client:
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


@lru_cache(maxsize=1)
def _llm() -> ChatOllama:
    return ChatOllama(model="mistral")


def _ms(t: float) -> int:
    return round((time.perf_counter() - t) * 1000)


router = APIRouter(
    prefix="/api/v1",
    tags=["v1"]
)


class LatencyBreakdown(BaseModel):
    embed_ms: int
    retrieval_ms: int
    rerank_ms: int
    generation_ms: int
    total_ms: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    request_id: str
    latency_breakdown: LatencyBreakdown


@traceable
@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, request: Request):
    request_id = request.state.request_id
    t_total = time.perf_counter()

    embeddings = OllamaEmbeddings(model="embeddinggemma")
    t0 = time.perf_counter()
    query_embedding = await embeddings.aembed_query(body.question)
    embed_ms = _ms(t0)

    vectorstore = SupabaseVectorStore(
        client=_supabase(),
        embedding=embeddings,
        table_name="documents",
    )
    t0 = time.perf_counter()
    results = await vectorstore.asimilarity_search_by_vector(query_embedding)
    retrieval_ms = _ms(t0)

    # For hierarchical chunks, retrieve parent text instead of child snippet.
    # Fixed/semantic chunks have no parent_id so they pass through unchanged.
    seen_parents: set[str] = set()
    context_parts: list[str] = []
    for doc in results:
        meta = doc.metadata
        parent_id = meta.get("parent_id")
        chunk_id = meta.get("chunk_id")
        is_child = parent_id and parent_id != chunk_id
        if is_child and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            response = (
                _supabase().table("documents")
                .select("content")
                .eq("metadata->>chunk_id", parent_id)
                .limit(1)
                .execute()
            )
            parent_text = response.data[0]["content"] if response.data else None
            context_parts.append(parent_text or doc.page_content)
        elif not is_child:
            context_parts.append(doc.page_content)

    context = "\n\n".join(context_parts)
    sources = list({
        doc.metadata.get("source", doc.metadata.get("file_path", "unknown"))
        for doc in results
    })

    prompt = f"""You are a knowledgeable parenting assistant grounded in peer-reviewed research. \
        Answer the parent's question using only the provided research excerpts. \
        Be warm, clear, and practical. Acknowledge uncertainty where the evidence is mixed or limited. \
        Never provide medical diagnoses or replace the advice of a pediatrician.

        Research excerpts:
        {context}
        """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": body.question}
    ]
    t0 = time.perf_counter()
    llm_response = await _llm().ainvoke(messages)
    generation_ms = _ms(t0)

    total_ms = _ms(t_total)
    breakdown = LatencyBreakdown(
        embed_ms=embed_ms,
        retrieval_ms=retrieval_ms,
        rerank_ms=0,
        generation_ms=generation_ms,
        total_ms=total_ms,
    )

    run = get_current_run_tree()
    if run:
        run.extra = {**(run.extra or {}), "metadata": breakdown.model_dump()}

    return QueryResponse(
        answer=llm_response.content,
        sources=sources,
        request_id=request_id,
        latency_breakdown=breakdown,
    )
