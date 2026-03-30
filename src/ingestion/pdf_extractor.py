import pymupdf4llm
import os
from langchain_text_splitters import MarkdownTextSplitter

class PDFExtractor:

    def __init__(self, ingestion_dir_path: str, output_dir_path: str, chunk_size=1000, chunk_overlap=200):
        self.ingestion_dir_path=ingestion_dir_path
        self.output_dir_path=output_dir_path
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


    def extract(self):
        pdf_files = []
        for pdf in os.listdir(self.ingestion_dir_path):
            if pdf.endswith(".pdf"):
                pdf_files.append(pdf)

        for pdf in pdf_files:
            pdf_path = os.path.join(self.ingestion_dir_path, pdf)
            md_text = pymupdf4llm.to_markdown(pdf_path)

            processed_pdf_path = os.path.join(self.output_dir_path, pdf.replace(".pdf", ".txt"))
            with open(processed_pdf_path, "w") as f:
                f.write(md_text)
