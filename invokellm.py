from langchain_ollama import OllamaLLM

def invoke_llm(prompt, results):
    model = OllamaLLM(
        model="mistral",
        temperature=0, #temperature 0 = same answer every run, no creative guessing
        keep_alive="1h", #keep model loaded between questions, default unloads after 5 min (slow first answer)
    )
    response_text = model.invoke(prompt)
    print(response_text)

    #get original source
    sources = [doc.metadata.get("id",None) for doc, _score in results]
    print(f"Sources: {sources}")

    return response_text
