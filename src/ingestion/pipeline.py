from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.chunker import Chunker

RAW_PDFS = "/Users/dakota/repos/parenting-rag/data/raw/pdfs"
RAW_JSONS = "/Users/dakota/repos/parenting-rag/data/raw/pubmed"
PROCESSED_PDFS = "/Users/dakota/repos/parenting-rag/data/processed"

class IngestionPipeline:
    
    def __init__(self, ingestion_dir_path: str, output_dir_path: str, chunk_size=1000, chunk_overlap=200):
        self.pdf_extractor = PDFExtractor(ingestion_dir_path, output_dir_path, chunk_size, chunk_overlap)
        self.chunker = Chunker(chunk_size, chunk_overlap)

    def run(self):
        self.pdf_extractor.extract()
        results = self.chunker.chunk_pdf_directory(path=PROCESSED_PDFS)
        results += self.chunker.chunk_json_directory(path=RAW_JSONS)

        for file, chunks in results:
            print(f"File: {file}, Number of Chunks: {len(chunks)}")

if __name__ == "__main__":
    pipeline = IngestionPipeline(ingestion_dir_path=RAW_PDFS, output_dir_path=PROCESSED_PDFS)
    pipeline.run()