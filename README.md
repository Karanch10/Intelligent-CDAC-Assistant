# Intelligent C-DAC Knowledge Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** chatbot that provides accurate and context-aware answers to queries related to **C-DAC courses, admissions, placements, eligibility, and FAQs**. The assistant retrieves information from an embedded knowledge base built using official C-DAC webpages and curated FAQs, ensuring grounded and reliable responses.

## 🚀 Live Demo

🔗 **Application:** https://intelligent-cdac-assistant.streamlit.app/

---

## ✨ Features

* 🤖 AI-powered conversational assistant for C-DAC queries
* 📚 Retrieval-Augmented Generation (RAG) architecture
* 🌐 Automated knowledge ingestion from official C-DAC website
* 📝 Support for curated Markdown-based FAQs
* 🔍 Semantic search using vector embeddings
* 💬 Context-aware answer generation using Mistral LLM
* 🎨 Interactive and user-friendly Streamlit interface

---

## 🛠️ Tech Stack

| Category             | Technologies                |
| -------------------- | --------------------------- |
| Programming Language | Python                      |
| LLM                  | Mistral AI                  |
| Framework            | LangChain                   |
| Vector Database      | ChromaDB                    |
| Embeddings           | Mistral Embeddings          |
| Web Scraping         | Requests, BeautifulSoup     |
| Frontend             | Streamlit                   |
| Environment          | Python, Virtual Environment |

---

## 🏗️ System Architecture

```text
                User Query
                     │
                     ▼
             Streamlit Interface
                     │
                     ▼
              LangChain Pipeline
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  Chroma Vector DB         Mistral LLM
          ▲                     │
          │                     ▼
 Semantic Search        Context-Aware Response
          ▲
          │
Knowledge Base
(Website + Curated FAQs)
```

---

## 📖 Knowledge Sources

The chatbot retrieves information from:

* Official C-DAC course pages
* C-DAC admission portal
* About C-DAC page
* C-DAC course flyers
* C-CAT admission information
* Curated FAQ knowledge base (`knowledge.md`)

The knowledge base is embedded into **ChromaDB** using **Mistral Embeddings** for semantic retrieval.

---

## ⚙️ How It Works

1. Scrapes content from official C-DAC webpages.
2. Processes and chunks the extracted text.
3. Generates vector embeddings using Mistral Embeddings.
4. Stores embeddings in ChromaDB.
5. Retrieves the most relevant chunks based on user queries.
6. Passes the retrieved context to the Mistral LLM.
7. Generates accurate, grounded responses.

---

## 📂 Project Structure

```text
Intelligent-CDAC-Assistant/
│
├── app.py                     # Streamlit application
├── chatbot.py                 # RAG pipeline
├── knowledge_builder.py       # Website knowledge ingestion
├── knowledge_builder_pdf.py   # Knowledge from PDFs
├── chroma_db/                 # Vector database
├── requirements.txt
├── .env
└── README.md
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/<your-username>/Intelligent-CDAC-Assistant.git
cd Intelligent-CDAC-Assistant
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file and add your Mistral API key.

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## 📥 Build the Knowledge Base

Run the website ingestion script:

```bash
python knowledge_builder.py
```

```bash
python knowledge_builder_pdf.py
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 💡 Sample Questions

* What courses are offered by C-DAC?
* What documents are required for admission?
* How are placements at C-DAC?
* What are the modules in the PGCP-AC course?
* What are the modules in the PGCP-BDA course?
* What is C-CAT?
* Tell me about C-DAC.

---

## 👨‍💻 Author

**Karan Chougule**

If you found this project useful, consider giving the repository a ⭐ on GitHub!
