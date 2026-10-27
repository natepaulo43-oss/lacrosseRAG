#Create vector to store embeddings in chroma db

from embedding import get_embedding_function
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loadingdata import load_documents
from split import split_documents

CHROMA_PATH = "chroma"

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include = []) #include IDs
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    #only add non existing docs
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids = new_chunk_ids)
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        #if page id is the same as the last one, increment index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        #id looks like "data/collegeRulebook.pdf:6:2" (source:page:chunk)
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        #add chunk to metadata
        chunk.metadata["id"] = chunk_id

        #now if we add new pdfs or new pages we will be able to have a way to check if its in the db or not

    return chunks


#run this file to load the pdf -> split -> store in chroma
if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
