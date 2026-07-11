import os
import re
import io
import time
import unicodedata

import requests
import pdfplumber
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found.")

admission_booklet_pdf = "https://www.cdac.in/index.aspx?id=acts_photo&dynamicid=PDF/Admission%20Booklet%20V1_0.pdf"

course_flyer_pdfs = {
    "bda_course_flyer": (
        "https://www.cdac.in/index.aspx?id=acts_photo&dynamicid=PDF/PGCP_BDA.pdf",
        "PGCP-BDA",
    ),
    "advanced_computing_course_flyer": (
        "https://www.cdac.in/index.aspx?id=acts_photo&dynamicid=PDF/PGCP_AC.pdf",
        "PGCP-AC",
    ),
    "esd_course_flyer": (
        "https://www.cdac.in/index.aspx?id=acts_photo&dynamicid=PDF/PGCP_ESD.pdf",
        "PGCP-ESD",
    ),
    "itiss_course_flyer": (
        "https://www.cdac.in/index.aspx?id=acts_photo&dynamicid=PDF/PGCP_ITISS.pdf",
        "PGCP-ITISS",
    ),
    "ai_course_flyer": (
        "https://www.cdac.in/index.aspx?id=acts_photo&dynamicid=PDF/PGCP_AI.pdf",
        "PGCP-AI",
    ),
}

# ============================================================
# Embeddings + vectorstore (same collection as knowledge_builder.py)
# ============================================================
embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=api_key,
)

vectorstore = Chroma(
    collection_name="cdac_knowledge_base",
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# ============================================================
# Text cleaning (from knowledge_builder_pdf reference)
# ============================================================
def clean_pdf_text(text):
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb00": "ff",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u01a6": "ti",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u0178": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"ACTS C-DAC PUNE.*", "", text)
    text = re.sub(r"published in.*", "", text)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def table_to_text(table):
    """Turn a pdfplumber table (list of rows of cells) into a readable,
    pipe-separated text block so it embeds meaningfully as text."""
    lines = []
    for row in table:
        cells = [(cell or "").strip() for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


# ============================================================
# Course flyers: PyMuPDFLoader (fast, good for text-heavy single-column PDFs)
# ============================================================
def ingest_course_flyer(key, url, title):
    print(f"Ingesting {title} flyer (PDF)...")
    try:
        loader = PyMuPDFLoader(url)
        docs = loader.load()
    except Exception as e:
        print(f"  Skipped ({e})")
        return

    chunks = []
    chunk_pdf_pages = []
    for doc in docs:
        cleaned = clean_pdf_text(doc.page_content)
        if not cleaned:
            continue
        for c in splitter.split_text(cleaned):
            chunks.append(c)
            pdf_page = doc.metadata.get("page")
            chunk_pdf_pages.append(pdf_page + 1 if isinstance(pdf_page, int) else None)

    if not chunks:
        print("  No text extracted.")
        return

    ids = [f"{key}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": url,
            "page": title,
            "category": "course",
            "format": "pdf",
            "pdf_page": chunk_pdf_pages[i],
        }
        for i in range(len(chunks))
    ]

    vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    print(f"  Stored {len(chunks)} chunks.")


# ============================================================
# Admission booklet: pdfplumber (better table extraction)
# ============================================================
def ingest_admission_booklet(url):
    print("Ingesting Admission Booklet (PDF, pdfplumber)...")
    try:
        response = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
        )
        response.raise_for_status()
    except Exception as e:
        print(f"  Skipped ({e})")
        return

    chunks = []
    chunk_pdf_pages = []
    try:
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""

                table_blocks = []
                for table in page.extract_tables():
                    table_text = table_to_text(table)
                    if table_text:
                        table_blocks.append(table_text)

                combined = page_text
                if table_blocks:
                    combined += "\n\n" + "\n\n".join(table_blocks)

                cleaned = clean_pdf_text(combined)
                if not cleaned:
                    continue

                for c in splitter.split_text(cleaned):
                    chunks.append(c)
                    chunk_pdf_pages.append(page_num + 1)
    except Exception as e:
        print(f"  Skipped ({e})")
        return

    if not chunks:
        print("  No text extracted.")
        return

    ids = [f"admission_booklet_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": url,
            "page": "Admission Booklet",
            "category": "admission",
            "format": "pdf",
            "pdf_page": chunk_pdf_pages[i],
        }
        for i in range(len(chunks))
    ]

    vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    print(f"  Stored {len(chunks)} chunks.")


# ============================================================
# Run
# ============================================================
ingest_admission_booklet(admission_booklet_pdf)
time.sleep(1)

for key, (url, title) in course_flyer_pdfs.items():
    ingest_course_flyer(key, url, title)
    time.sleep(1)

print("\nPDF Knowledge Base Updated Successfully!")