from pydantic import BaseModel

class DocumentChunk(BaseModel):
    source: str
    content: str
    embedding: list[float] | None = None
    