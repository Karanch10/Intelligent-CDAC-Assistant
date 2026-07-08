import os
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found.")

pages = [
    {
        "url": "https://www.cdac.in/index.aspx?id=about",
        "page": "About C-DAC",
        "category": "organization"
    },
    {
        "url": "https://cdac.in/index.aspx?id=edu_acts_PGDiplomaAdmission",
        "page": "C-CAT Admission",
        "category": "admission"
    },
    {
        "url": "https://cdac.in/index.aspx?id=DAC&courseid=0",
        "page": "PGCP-AC",
        "category": "course"
    },
    {
        "url": "https://cdac.in/index.aspx?id=DAC&courseid=22",
        "page": "PGCP-ESD",
        "category": "course"
    },
    {
        "url": "https://cdac.in/index.aspx?id=DAC&courseid=26",
        "page": "PGCP-MC",
        "category": "course"
    },
    {
        "url": "https://cdac.in/index.aspx?id=DAC&courseid=30",
        "page": "PGCP-AI",
        "category": "course"
    },
    {
        "url": "https://cdac.in/index.aspx?id=DAC&courseid=65",
        "page": "PGCP-BDA",
        "category": "course"
    },
    {
        "url": "https://cdac.in/index.aspx?id=DAC&courseid=28",
        "page": "PGCP-ITISS",
        "category": "course"
    }
]

embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=api_key
)

vectorstore = Chroma(
    collection_name="cdac_knowledge_base",
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

def fetch_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {url} (status {response.status_code})")
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

for page in pages:

    print(f"Ingesting {page['page']}...")
    try:
        text = fetch_page(page["url"])
    except Exception as e:
        print(f"  Skipped ({e})")
        continue
    chunks = splitter.split_text(text)
    ids = [
        f"{page['page']}_{i}"
        for i in range(len(chunks))
    ]
    metadatas = [
        {
            "source": page["url"],
            "page": page["page"],
            "category": page["category"]
        }
        for _ in chunks
    ]
    vectorstore.add_texts(
        texts=chunks,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Stored {len(chunks)} chunks.")
    time.sleep(1)  
print("\nKnowledge Base Updated Successfully!")