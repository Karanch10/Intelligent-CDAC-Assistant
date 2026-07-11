import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found.")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=api_key,
)

vectorstore = Chroma(
    collection_name="cdac_knowledge_base",
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 6}
)

llm = ChatMistralAI(
    model="mistral-small-2506",
    api_key=api_key,
    temperature=0.2,
    streaming=True
)

prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant for C-DAC.

Answer ONLY using the provided context.

If the answer is not available in the context, say:

"I couldn't find that information in the knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""
)

parser = StrOutputParser()

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | parser
)

while True:

    question = input("\nAsk: ")
    if question.lower() == "exit":
        break
    print("\nThinking...\n")
    print("\nAssistant: ", end="", flush=True)
    
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)

    print()