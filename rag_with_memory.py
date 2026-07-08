import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found.")

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

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
    search_kwargs={"k": 3}
)

llm = ChatMistralAI(
    model="mistral-small-2506",
    api_key=api_key,
    temperature=0.2,
    streaming=True
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI assistant for C-DAC.
Use the retrieved context as your primary source of truth.
You also receive the previous conversation history to understand follow-up questions.
If the answer cannot be found in the retrieved context, reply:
"I couldn't find that information in the knowledge base."
Keep answers concise and well formatted.
Context:
{context}
"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "{question}"
        ),
    ]
)

parser = StrOutputParser()

rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["question"]))
    )
    | prompt
    | llm
    | parser
)

rag_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

SESSION_ID  = "karan"

while True:

    question = input("\nAsk: ")
    if question.lower() == "exit":
        break
    print("\nAssistant: ", end="", flush=True)
    config = {
        "configurable": {
            "session_id": SESSION_ID
        }
    }
    for chunk in rag_with_memory.stream(
        {
            "question": question
        },
        config=config,
    ):
        print(chunk, end="", flush=True)
    print()