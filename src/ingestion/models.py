from pydantic import BaseModel

class DocumentChunk(BaseModel):
    source: str
    content: str
    embedding: list[float] | None = None
    parent_id: str | None = None
    chunk_id: str | None = None