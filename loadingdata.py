#Langchain for document loader

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

def load_documents():
    document_loader = PyPDFDirectoryLoader('data') #loader takes a folder, not a single file
    return document_loader.load()
