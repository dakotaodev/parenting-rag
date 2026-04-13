from fastapi import APIRouter
from langchain_community.vectorstores import SupabaseVectorStore
from langsmith import traceable
from supabase import create_client
from dotenv import load_dotenv
import os
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

load_dotenv()
supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
llm = ChatOllama(model="mistral")

router = APIRouter(
    prefix="/api/v1",
    tags=["v1"]
)

@traceable
@router.get("/api/v1/query")
async def query(query:str):
    vectorstore = SupabaseVectorStore(
        client=supabase_client,
        embedding=OllamaEmbeddings(model="embeddinggemma"),
        table_name="documents",
    )
    results = vectorstore.similarity_search(query)
    
    context = "\n\n".join(doc.page_content for doc in results)
    print(context)

    prompt = f"""You are a knowledgeable parenting assistant grounded in peer-reviewed research. \
        Answer the parent's question using only the provided research excerpts. \
        Be warm, clear, and practical. Acknowledge uncertainty where the evidence is mixed or limited. \
        Never provide medical diagnoses or replace the advice of a pediatrician.

        Research excerpts:
        {context}
        """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]
    response = llm.invoke(messages)
    return {"response": response.content}