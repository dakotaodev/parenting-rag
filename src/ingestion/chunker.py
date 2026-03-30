from langchain_text_splitters import MarkdownTextSplitter
import os 
import json


class Chunker:

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # create method that applies chunking to all files in a directory
    def chunk_pdf_directory(self, path: str):
        results=[] 
        for file in os.listdir(path):
            if file.endswith(".md") or file.endswith(".txt"):
                file_path = os.path.join(path, file)
                with open(file_path, "r") as f:
                    text = f.read()
                chunks = self.chunk_fixed_size(text)
                results.append((file, chunks))
        return results
    
    def chunk_json_directory(self, path: str):
        results=[] 
        # for each file in path and its subdirectories, if file ends with .json, read it and chunk it
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        # load json and extract the abstract field
                        data = json.load(f)
                        text = data.get("abstract", "")
                chunks = self.chunk_fixed_size(text)
                results.append((file, chunks))
        return results

    def chunk_fixed_size(self, text: str):
        return self.splitter.split_text(text)