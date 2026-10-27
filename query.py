#create queries

import sys
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from embedding import get_embedding_function
from invokellm import invoke_llm
from vectordb import CHROMA_PATH

ptemplate = """
Answer the question based only on the following context:
{context}

---
Answer the question based on the above context: {question}

Rules:
- Do not use outside knowledge.
- The question may use different words than the context (a nickname or synonym), so match by meaning.
- If the context answers the question, state the answer plainly and briefly.
- Only if nothing in the context answers the question, reply with exactly: NOT_FOUND
- If the specific thing asked about (an age group, level, team, etc.) is not mentioned in the context, do not explain or guess, reply with only: NOT_FOUND
"""

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = embedding_function
    )

    #search database for top 8 results (chunks are split per rule now and smaller, so 8 x ~300 chars is still less text than before = faster answers)
    results = db.similarity_search_with_score(query_text, k=8)

    content_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(ptemplate)
    prompt = prompt_template.format(context = content_text, question = query_text)

    return invoke_llm(prompt, results)


#usage: python query.py "How long is the shot clock?"
#   or: python query.py            (keeps asking questions until you type exit, good for demos)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_rag(sys.argv[1])
    else:
        print("Lacrosse rulebook chatbot - type a question, or 'exit' to quit")
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if question:
                query_rag(question)
