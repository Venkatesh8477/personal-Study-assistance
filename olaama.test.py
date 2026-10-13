from langchain_ollama import ChatOllama
llm=ChatOllama(model='qwen2.5:3b',
               temperature=0)
response=llm.invoke(
    "explain operating system in two senteance and example"
)
print(response.content)