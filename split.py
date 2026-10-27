#split document into smaller chunks

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400, #smaller chunks = each one is about a single rule, so search matches it more precisely
        chunk_overlap = 40,
        length_function=len,
        is_separator_regex = False,
        #split at the rulebook's own structure first, so each rule (SECTION) and example (A.R.) starts its own chunk
        #instead of getting cut off mid-rule like "SECTION 2. The penalty for a technical" | "foul shall be..."
        separators = ["\nSECTION ", "\nA.R. ", "\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    #drop table of contents / index chunks ("Offside ........ 41"), they match keywords but never have answers
    return [chunk for chunk in chunks if len(re.findall(r"\.{5,}", chunk.page_content)) < 2]
