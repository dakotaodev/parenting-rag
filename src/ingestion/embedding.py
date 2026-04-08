from langchain_ollama import OllamaEmbeddings
import enum

class EmbeddingProvider(enum.Enum):
    OLLAMA = "ollama"

class Embedder:
    
    def __init__(self, provider: EmbeddingProvider, model: str):
        if provider == EmbeddingProvider.OLLAMA:
            self.embeddings = OllamaEmbeddings(model=model)
    
    def embed(self, text: str):
        return self.embeddings.embed_query(text)
    
    def embed_batch(self, texts: list[str], batch_size=100):
        results = []
        for i in range(0, len(texts), batch_size):
            results.extend(self.embeddings.embed_documents(texts[i:i+batch_size]))
        return results


if __name__ == "__main__":
    embedder = Embedder(provider=EmbeddingProvider.OLLAMA, model="embeddinggemma")
    embedding = embedder.embed("Hello, world!")
    print(embedding)