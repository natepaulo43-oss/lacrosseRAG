#embedding using Langchain
#Using ollama so I can run completely locally

from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(model = "nomic-embed-text") #dedicaded local embedding model
        

    return embeddings

