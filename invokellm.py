from langchain_community.llms.ollama import Ollama

def invoke_llm(prompt, results):
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)
    print(response_text)

    #get original source
    sources = [doc.metadata.get("id",None) for doc, _score in results]
    print(f"Sources: {sources}")

    return response_text
