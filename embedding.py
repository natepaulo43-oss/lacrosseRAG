#embedding using Langchain
#Using ollama so I can run completely locally

from langchain_ollama import OllamaEmbeddings

#nomic was trained with these prefixes on chunks vs questions, retrieval is much worse without them
#the new langchain_ollama class has no setting for this so we add them ourselves
class NomicEmbeddings(OllamaEmbeddings):
    def embed_documents(self, texts):
        return super().embed_documents(["search_document: " + t for t in texts])

    def embed_query(self, text):
        return super().embed_query("search_query: " + text)

def get_embedding_function():
    embeddings = NomicEmbeddings(model = "nomic-embed-text") #dedicaded local embedding model

    return embeddings
