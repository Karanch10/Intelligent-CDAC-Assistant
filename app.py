import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Intelligent C-DAC Assistant",
    page_icon="🖥️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Design tokens
# ------------------------------------------------------------
# Background : #0B1220  (deep navy, not pure black)
# Surface    : #131B2E
# Surface-2  : #1A2338  (assistant bubble)
# Border     : #26314A
# Text       : #E7ECF6
# Text muted : #8D96AC
# Accent     : #E8A33D  (amber / "certificate seal" gold)
# Accent-2   : #3E7CB1  (steel blue, user bubble)
# Display font: Space Grotesk | Body font: Inter
# Signature element: hexagonal "seal" mark (nods to C-DAC certifications)
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0B1220;
    --surface: #131B2E;
    --surface-2: #1A2338;
    --border: #26314A;
    --text: #E7ECF6;
    --text-muted: #8D96AC;
    --accent: #E8A33D;
    --accent-soft: rgba(232, 163, 61, 0.12);
    --accent-2: #3E7CB1;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background:
        radial-gradient(circle at 15% -10%, rgba(232,163,61,0.07), transparent 40%),
        radial-gradient(circle at 85% 0%, rgba(62,124,177,0.10), transparent 45%),
        var(--bg);
}

.block-container {
    max-width: 760px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

/* ---------- Header ---------- */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 6px;
}

.seal {
    width: 40px;
    height: 40px;
    flex-shrink: 0;
    background: linear-gradient(155deg, var(--accent), #C97F1E);
    clip-path: polygon(50% 0%, 95% 25%, 95% 75%, 50% 100%, 5% 75%, 5% 25%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    color: #0B1220;
    font-size: 15px;
}

.app-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.45rem;
    color: var(--text);
    line-height: 1.15;
}

.app-subtitle {
    font-size: 0.82rem;
    color: var(--text-muted);
    letter-spacing: 0.02em;
}

hr.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 18px 0 22px 0;
}

/* ---------- Greeting bubble ---------- */
.greeting-bubble {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 16px 18px;
    color: var(--text);
    font-size: 0.96rem;
    line-height: 1.5;
    max-width: 88%;
    margin-bottom: 22px;
}

.section-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 10px;
    font-weight: 600;
}

/* ---------- Quick action buttons ---------- */
div[data-testid="stButton"] > button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
    padding: 0.7rem 1rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    width: 100%;
    transition: border-color 0.15s ease, background 0.15s ease, transform 0.1s ease;
    text-align: left !important;
}

div[data-testid="stButton"] > button:hover {
    border-color: var(--accent) !important;
    background: var(--accent-soft) !important;
    color: var(--text) !important;
    transform: translateY(-1px);
}

div[data-testid="stButton"] > button:active {
    transform: translateY(0px);
}

div[data-testid="stButton"] > button:focus:not(:active) {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* ---------- Chat messages ---------- */
div[data-testid="stChatMessage"] {
    background: transparent;
    padding: 4px 0;
}

div[data-testid="stChatMessageContent"] {
    border-radius: 14px;
    padding: 12px 16px;
    font-size: 0.94rem;
    line-height: 1.55;
    border: 1px solid var(--border);
}

/* assistant bubble */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) div[data-testid="stChatMessageContent"] {
    background: var(--surface-2);
    color: var(--text);
}

/* user bubble */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"] {
    background: rgba(62, 124, 177, 0.16);
    border-color: rgba(62, 124, 177, 0.4);
    color: var(--text);
}

/* ---------- Chat input ---------- */
div[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}

div[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# Backend: build the RAG-with-memory chain once per server process
# ============================================================
@st.cache_resource(show_spinner=False)
def load_chain():
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY") or st.secrets["MISTRAL_API_KEY"]
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found.")

    embedding_model = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

    vectorstore = Chroma(
        collection_name="cdac_knowledge_base",
        persist_directory="./chroma_db",
        embedding_function=embedding_model,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatMistralAI(
        model="mistral-small-2506",
        api_key=api_key,
        temperature=0.2,
        streaming=True,
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
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    parser = StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | parser
    )

    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )


rag_with_memory = load_chain()

# ============================================================
# Session state
# ============================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

QUICK_ACTIONS = [
    ("About C-DAC", "Tell me about C-DAC \u2014 what it is, its mission, and what it does."),
    ("PG Certificate Programmes", "What Post Graduate Certificate programmes does C-DAC offer?"),
    ("C-CAT Examination", "Tell me about the C-CAT entrance examination for C-DAC's PG Diploma admissions."),
    ("General C-DAC Question", None),
]

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("**C-DAC Assistant**")
    st.caption("RAG-powered \u00b7 Mistral \u00b7 Chroma")
    if st.button("Start new chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.pending_query = None
        st.rerun()

# ============================================================
# Header
# ============================================================
st.markdown(
    """
    <div class="app-header">
        <div class="seal">C</div>
        <div>
            <div class="app-title">C-DAC Assistant</div>
            <div class="app-subtitle">Admissions \u00b7 Courses \u00b7 C-CAT \u00b7 General enquiries</div>
        </div>
    </div>
    <hr class="divider" />
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Greeting + quick actions (only shown before the first message)
# ============================================================
if not st.session_state.messages and not st.session_state.pending_query:
    st.markdown(
        """
        <div class="greeting-bubble">
            Hello! I'm here to help you with questions about C-DAC \u2014
            courses, admissions, the C-CAT exam, and more. How can I assist you today?
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Quick start</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for i, (label, query) in enumerate(QUICK_ACTIONS):
        col = cols[i % 2]
        if col.button(label, key=f"qa_{i}", use_container_width=True):
            if query:
                st.session_state.pending_query = query
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Sure \u2014 go ahead and type your question about "
                        "C-DAC below, and I'll do my best to help. \U0001F447",
                    }
                )
            st.rerun()

# ============================================================
# Render existing chat history
# ============================================================
for msg in st.session_state.messages:
    avatar = "\U0001F393" if msg["role"] == "assistant" else "\U0001F9D1"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ============================================================
# Helper: run a question through the chain, streaming the answer
# ============================================================
def process(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="\U0001F9D1"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="\U0001F393"):
        placeholder = st.empty()
        full_response = ""
        config = {"configurable": {"session_id": st.session_state.session_id}}
        for chunk in rag_with_memory.stream({"question": question}, config=config):
            full_response += chunk
            placeholder.markdown(full_response + "\u258c")
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


# Handle a query queued by a quick-action button
if st.session_state.pending_query:
    q = st.session_state.pending_query
    st.session_state.pending_query = None
    process(q)

# ============================================================
# Chat input (Streamlit pins this to the bottom of the page)
# ============================================================
user_input = st.chat_input("Type your query...")
if user_input:
    process(user_input)