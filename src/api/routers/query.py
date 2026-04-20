import os

from dotenv import load_dotenv
from fastapi import APIRouter, Request
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langsmith import traceable
from pydantic import BaseModel, Field
from supabase import create_client

load_dotenv()
supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
llm = ChatOllama(model="mistral")

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
        client=supabase_client,
        embedding=OllamaEmbeddings(model="embeddinggemma"),
        table_name="documents",
    )
    results = await vectorstore.asimilarity_search(body.question)

    context = "\n\n".join(doc.page_content for doc in results)
    sources = [
        doc.metadata.get("source", doc.metadata.get("file_path", "unknown"))
        for doc in results
    ]

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
    response = await llm.ainvoke(messages)

    return QueryResponse(answer=response.content, sources=sources, request_id=request_id)
