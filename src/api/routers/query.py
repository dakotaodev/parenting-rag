import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import APIRouter, Request
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langsmith import traceable
from pydantic import BaseModel, Field
from supabase import Client, create_client

load_dotenv()


@lru_cache(maxsize=1)
def _supabase() -> Client:
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


@lru_cache(maxsize=1)
def _llm() -> ChatOllama:
    return ChatOllama(model="mistral")

router = APIRouter(
    prefix="/api/v1",
    tags=["v1"]
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    request_id: str


@traceable
@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, request: Request):
    request_id = request.state.request_id

    vectorstore = SupabaseVectorStore(
        client=_supabase(),
        embedding=OllamaEmbeddings(model="embeddinggemma"),
        table_name="documents",
    )
    results = await vectorstore.asimilarity_search(body.question)

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
    response = await _llm().ainvoke(messages)

    return QueryResponse(answer=response.content, sources=sources, request_id=request_id)
