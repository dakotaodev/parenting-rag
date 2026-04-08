from supabase import create_client, Client
from dotenv import load_dotenv
import os

from src.ingestion.models import DocumentChunk

load_dotenv()

class SupabaseClient:

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        self.client: Client = create_client(url, key)

    def upsert_embeddings(self, data: list[DocumentChunk]):
        rows = [
            {"source": c.source, "content": c.content, "embedding": c.embedding}
            for c in data
        ]
        self.client.table("documents").insert(rows).execute()


