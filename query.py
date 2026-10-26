#create queries

import sys
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from embedding import get_embedding_function
from invokellm import invoke_llm
from vectordb import CHROMA_PATH

ptemplate = """
Answer the question based only on the following context:
{context}

---
Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = embedding_function
    )

    #search database for top 5 results
    results = db.similarity_search_with_score(query_text, k=5)

    content_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(ptemplate)
    prompt = prompt_template.format(context = content_text, question = query_text)

    return invoke_llm(prompt, results)


#usage: python query.py "How long is the shot clock?"
if __name__ == "__main__":
    query_rag(sys.argv[1])
