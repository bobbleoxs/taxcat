# app.py
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from joblib import Memory
from pydantic import BaseModel

# Only load .env file in development
if os.getenv("ENVIRONMENT") != "production":
    load_dotenv()

# Initialize joblib memory cache
memory = Memory("cache", verbose=0)

# Get environment variables
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://taxcat.ai")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
OPENAI_API_KEY = "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"

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. Please set it in your environment variables."
    )


class Query(BaseModel):
    text: str


def load_documents_from_json(json_file="vat_documents_enhanced.json"):
    """Load documents from JSON file and return a list of Document objects."""
    print(f"📂 Loading documents from {json_file}...")

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            documents_data = json.load(f)

        documents = []
        for doc_data in documents_data:
            doc = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                meta=doc_data.get("meta", {}),
                embedding=doc_data.get("embedding"),
            )
            documents.append(doc)

        print(f"✅ Loaded {len(documents)} documents from {json_file}")
        return documents

    except FileNotFoundError:
        print(f"❌ Error: {json_file} not found")
        return []
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in {json_file}")
        return []


app = FastAPI(
    title="TaxCat VAT Classification API",
    description="API for classifying UK VAT transactions",
    version="1.0.0",
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if ENVIRONMENT != "production" else None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "🐱 Welcome to TaxCat - UK VAT Classification API",
        "status": "healthy",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "endpoints": {
            "/classify": "POST - Classify VAT transactions",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "TaxCat VAT Classification",
        "environment": ENVIRONMENT,
        "timestamp": "2024-03-19",
    }


# 0) Initialize embedders
text_embedder = OpenAITextEmbedder(model="text-embedding-3-small", api_key=Secret.from_token(OPENAI_API_KEY))
# We'll need a document embedder if/when we add documents to the store
# document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

# 1) Load your store & retriever
store = InMemoryDocumentStore()

# LOAD DOCUMENTS: Load pre-processed documents from JSON
documents = load_documents_from_json()
if documents:
    store.write_documents(documents)
    print(f"📚 Document store populated with {len(documents)} documents")
else:
    print("⚠️  No documents loaded. VAT classification may not work properly.")

retriever = InMemoryEmbeddingRetriever(document_store=store)

# 2) Prompt & generator
prompt_template = [
    ChatMessage.from_user(
        """
    You are an expert in UK VAT.

    Context from retrieved documents:
    {% for doc in documents %}
      Document Content: {{ doc.content }}
      {% if doc.meta and doc.meta.get('url') %}
      Source URL: {{ doc.meta.get('url') }}
      {% endif %}
      ---
    {% endfor %}

    When classifying transactions, pay special attention to:
    - Post-Brexit rules for cross-border and digital services
    - Place-of-supply rules (especially for services and digital products)
    - B2B digital services from the EU to the UK: these are typically subject to the reverse charge (out of scope for the supplier, customer accounts for VAT)
    - Domestic reverse charge rules for construction and other relevant sectors
    - Always provide consistent and correct answers for cross-border, digital, and reverse charge scenarios, referencing HMRC VAT Notice 741A and 735 where relevant

    Based on the context above and your expertise, analyze the following transaction.
    Transaction: {{query}}

    Provide the following:
    1) VAT Rate (e.g., 20%, 5%, 0%, exempt, out_of_scope, needs_review)
    2) Short Rationale
    3) Source URL (if identified from the documents, otherwise N/A)
    4) Confidence (low/medium/high)
    """
    )
]
prompt_builder = ChatPromptBuilder(
    template=prompt_template, required_variables=["query", "documents"]
)
generator = OpenAIChatGenerator(
    model="gpt-4o-mini", api_key=Secret.from_token(OPENAI_API_KEY)
)

# 3) Assemble pipeline
pipe = Pipeline()
pipe.add_component(instance=text_embedder, name="text_embedder")
pipe.add_component(instance=retriever, name="retriever")
pipe.add_component(instance=prompt_builder, name="prompter")
pipe.add_component(instance=generator, name="generator")

# Connect components
pipe.connect("text_embedder.embedding", "retriever.query_embedding")
pipe.connect("retriever.documents", "prompter.documents")
pipe.connect("prompter", "generator")


@memory.cache
def get_classification(text: str) -> str:
    """Cached function to get VAT classification."""
    pipeline_input = {
        "text_embedder": {"text": text},
        "retriever": {"top_k": 5},
        "prompter": {"query": text},
    }
    result = pipe.run(pipeline_input)
    return result["generator"]["replies"][0].text


@app.post("/classify")
def classify(q: Query):
    return {"response": get_classification(q.text)}
