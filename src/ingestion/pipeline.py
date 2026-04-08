from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.chunker import Chunker
from src.ingestion.embedding import Embedder, EmbeddingProvider
from src.ingestion.models import DocumentChunk
from src.clients.supabase import SupabaseClient

RAW_PDFS = "/Users/dakota/repos/parenting-rag/data/raw/pdfs"
RAW_JSONS = "/Users/dakota/repos/parenting-rag/data/raw/pubmed"
PROCESSED_PDFS = "/Users/dakota/repos/parenting-rag/data/processed"

class IngestionPipeline:
    
    def __init__(self, ingestion_dir_path: str, output_dir_path: str, chunk_size=1000, chunk_overlap=200):
        self.pdf_extractor = PDFExtractor(ingestion_dir_path, output_dir_path, chunk_size, chunk_overlap)
        self.chunker = Chunker(chunk_size, chunk_overlap)
        self.embedder = Embedder(provider=EmbeddingProvider.OLLAMA, model="embeddinggemma")
        # supabase client for storing embeddings and metadata in a vector database
        self.supabase_client = SupabaseClient()

    def run(self):
        self.pdf_extractor.extract()
        doc_chunks: list[DocumentChunk] = self.chunker.chunk_pdf_directory(path=PROCESSED_PDFS)
        doc_chunks += self.chunker.chunk_json_directory(path=RAW_JSONS)

        embeddings = self.embedder.embed_batch([c.content for c in doc_chunks])
        for chunk, emb in zip(doc_chunks, embeddings):
            chunk.embedding = emb

        self.supabase_client.upsert_embeddings(data=doc_chunks)


if __name__ == "__main__":
    pipeline = IngestionPipeline(ingestion_dir_path=RAW_PDFS, output_dir_path=PROCESSED_PDFS)
    pipeline.run()