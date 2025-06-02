# scripts/ingest.py
import json
import os

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret

# INSTRUCTIONS:
# 1. Download the following HMRC VAT notices as HTML and place them in the data/ directory:
#    - VAT Notice 700 (The VAT Guide)
#    - VAT Notice 700_12 (How to fill in VAT returns)
#    - VAT Notice 700_21 (Keeping VAT records)
#    - VAT Notice 741A (Place of supply of services)
#    - VAT Notice 735 (Domestic reverse charge procedure)
# 2. After downloading, run this script to regenerate embeddings and update the knowledge base.
# 3. The classification API should prioritize:
#    - Post-Brexit rules for cross-border digital services
#    - B2B digital services from EU to UK = reverse charge (out of scope for supplier)
#    - Consistent application of place-of-supply rules

# Only load .env file in development
if os.getenv("ENVIRONMENT") != "production":
    load_dotenv()

# Hardcoded API key for now
OPENAI_API_KEY = "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"

store = InMemoryDocumentStore()

embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small", api_key=Secret.from_token(OPENAI_API_KEY))
writer = DocumentWriter(document_store=store)

files = [
    "data/VAT Notice 700.html",
    "data/VAT Notice 700_12.html",
    "data/VAT Notice 700_21.html",
    "data/VAT Notice 741A.html",
    "data/VAT Notice 735.html",
]


def classify_metadata(section_title, text):
    title = (section_title or "").lower() + " " + (text[:200].lower() if text else "")

    # Business type (B2B vs B2C)
    if any(
        k in title
        for k in [
            "b2b",
            "business to business",
            "vat-registered business",
            "reverse charge",
            "business",
            "company",
            "enterprise",
        ]
    ):
        business_type = "b2b"
    elif any(
        k in title
        for k in ["b2c", "consumer", "private individual", "personal", "individual"]
    ):
        business_type = "b2c"
    else:
        # Default to B2B for business-related content
        business_type = "b2b"

    # Service type (digital vs goods)
    if any(
        k in title
        for k in [
            "digital",
            "online",
            "e-service",
            "e-service",
            "moss",
            "oss",
            "cloud",
            "software",
            "subscription",
            "service",
            "services",
        ]
    ):
        service_type = "digital"
    elif any(
        k in title for k in ["goods", "physical", "tangible", "product", "products"]
    ):
        service_type = "goods"
    else:
        # Default to digital for service-related content
        service_type = "digital"

    # Direction (to/from UK)
    if any(
        k in title
        for k in [
            "to uk",
            "to united kingdom",
            "import",
            "received from",
            "supplied to",
            "sold to",
        ]
    ):
        direction = "to_uk"
    elif any(
        k in title
        for k in ["from uk", "from united kingdom", "export", "supplied to", "sold to"]
    ):
        direction = "from_uk"
    else:
        # Default to to_uk for most content
        direction = "to_uk"

    # Source country (if mentioned)
    source_country = None
    for country in ["ireland", "france", "germany", "spain", "italy", "netherlands"]:
        if f"from {country}" in title:
            source_country = country
            break

    # Brexit status
    if (
        "post-brexit" in title
        or "from 1 january 2021" in title
        or "after 31 december 2020" in title
        or "after brexit" in title
    ):
        brexit_status = "post_brexit"
    else:
        brexit_status = "general"

    # Create metadata with type field
    metadata = {
        "type": f"{business_type}_{service_type}_{direction}",
        "business_type": business_type,
        "service_type": service_type,
        "direction": direction,
        "source_country": source_country,
        "brexit_status": brexit_status,
        "section_title": section_title.strip() if section_title else None,
    }

    return metadata


def is_pre_brexit(text):
    t = text.lower()
    return (
        "before 1 january 2021" in t
        or "before brexit" in t
        or "prior to 1 january 2021" in t
        or "before 31 december 2020" in t
        or "superseded" in t
    )


def semantic_chunk_html(html, file_name):
    soup = BeautifulSoup(html, "html.parser")
    # Find all headers and split by them
    sections = []
    for header in soup.find_all(["h1", "h2", "h3", "h4"]):
        section_title = header.get_text()
        content = []
        for sib in header.find_next_siblings():
            if sib.name and sib.name.startswith("h"):
                break
            content.append(sib.get_text(" ", strip=True))
        section_text = " ".join(content).strip()
        if not section_text or is_pre_brexit(section_text):
            continue  # skip empty or pre-Brexit/superseded
        meta = classify_metadata(section_title, section_text)
        meta["source_file"] = file_name
        sections.append({"content": section_text, "meta": meta})
    return sections


all_chunks = []
for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()
    chunks = semantic_chunk_html(html, os.path.basename(file_path))
    all_chunks.extend(chunks)

print(f"🔎 Created {len(all_chunks)} semantic chunks from {len(files)} VAT notices.")

documents = []
doc_counter = 0
for chunk in all_chunks:
    content = chunk["content"]
    meta = chunk["meta"]
    section_title = (meta.get("section_title") or "").lower()
    if (
        "cookie" in content.lower()
        or "cookie" in section_title
        or len(content.strip()) < 50
    ):
        continue
    doc_counter += 1
    doc = Document(id=f"doc_{doc_counter}", content=content, meta=meta)
    documents.append(doc)

# Create pipeline for embedding and writing
pipeline = Pipeline()
pipeline.add_component("embedder", embedder)
pipeline.add_component("writer", writer)
pipeline.connect("embedder", "writer")

# Process documents in batches
print(f"🔗 Embedding and storing {len(documents)} documents...")
result = pipeline.run({"embedder": {"documents": documents}})

# Save to disk
print("💾 Saving documents to disk...")
all_documents = store.filter_documents()
documents_data = []
for doc in all_documents:
    doc_data = {
        "id": doc.id,
        "content": doc.content,
        "meta": doc.meta,
        "embedding": doc.embedding if doc.embedding is not None else None,
    }
    documents_data.append(doc_data)
with open("vat_documents.json", "w", encoding="utf-8") as f:
    json.dump(documents_data, f, ensure_ascii=False, indent=2)
print(
    f"✔️ Ingestion & indexing complete. 📁 Saved {len(documents_data)} documents to 'vat_documents.json'"
)
