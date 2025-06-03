# app_enhanced.py
import json
import sys

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from joblib import Memory
from pydantic import BaseModel

# Add scripts directory to path for imports
sys.path.append("scripts")
from chunk_analyzer import VATChunkAnalyzer
from query_classifier import VATQueryClassifier

load_dotenv()

# Initialize joblib memory cache
memory = Memory("cache", verbose=0)


class Query(BaseModel):
    text: str


class EnhancedQuery(BaseModel):
    text: str
    use_classification: bool = True  # Option to enable/disable classification filtering


def load_enhanced_documents_from_json(json_file="vat_documents_enhanced.json"):
    """Load enhanced documents with metadata from JSON file and return a list of Document objects."""
    print(f"📂 Loading enhanced documents from {json_file}...")

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

        print(f"✅ Loaded {len(documents)} enhanced documents from {json_file}")
        return documents

    except FileNotFoundError:
        print(
            f"❌ Error: {json_file} not found. Trying fallback to regular documents..."
        )
        return load_documents_from_json("vat_documents.json")
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in {json_file}")
        return []


def load_documents_from_json(json_file="vat_documents.json"):
    """Fallback: Load regular documents from JSON file."""
    print(f"📂 Loading regular documents from {json_file}...")

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
        print(
            f"❌ Error: {json_file} not found. Please run 'python scripts/ingest.py' first."
        )
        return []
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in {json_file}")
        return []


def filter_documents_by_classification(documents, classification_result):
    """Filter documents based on classification metadata"""
    if not classification_result:
        return documents

    # Prepare chunk analyzer for filtering
    chunk_analyzer = VATChunkAnalyzer()

    # Convert documents to chunks format
    chunks = []
    for doc in documents:
        vat_metadata = doc.meta.get("vat_metadata", {})
        chunks.append(
            {
                "content": doc.content,
                "metadata": vat_metadata,
                "id": doc.id,
                "document": doc,  # Keep reference to original document
            }
        )

    # Filter chunks based on classification
    filtered_chunks = chunk_analyzer.filter_chunks_by_classification(
        chunks, classification_result
    )

    # Extract documents from filtered chunks
    filtered_documents = [chunk["document"] for chunk in filtered_chunks]

    print(
        f"🔍 Filtered from {len(documents)} to {len(filtered_documents)} documents based on classification"
    )

    return filtered_documents


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend's URL for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize query classifier
query_classifier = VATQueryClassifier()


@app.get("/")
def root():
    return {
        "message": "🐱 Welcome to TaxCat Enhanced - UK VAT Classification API with Smart Filtering",
        "status": "healthy",
        "version": "2.0.0",
        "features": {
            "query_classification": "Automatic B2B/B2C, service type, location detection",
            "metadata_filtering": "Context-aware document filtering for better accuracy",
            "enhanced_rag": "Targeted retrieval based on transaction characteristics",
        },
        "endpoints": {
            "/classify": "POST - Classify VAT transactions with enhanced filtering",
            "/classify-simple": "POST - Simple classification without filtering",
            "/classification-info": "GET - Get classification system info",
            "/docs": "GET - API Documentation",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "TaxCat Enhanced VAT Classification",
        "timestamp": "2025-01-31",
        "features": {
            "query_classification": True,
            "metadata_filtering": True,
            "enhanced_documents": True,
        },
    }


@app.get("/classification-info")
def classification_info():
    """Get information about the classification system capabilities"""
    return {
        "business_types": ["B2B", "B2C", "unclear"],
        "service_types": ["digital", "professional", "physical", "mixed", "unclear"],
        "geographic_scopes": ["UK_domestic", "EU", "international", "general"],
        "vat_treatments": [
            "20%",
            "0%",
            "exempt",
            "out_of_scope",
            "reverse_charge",
            "general",
        ],
        "supported_locations": {
            "UK": ["uk", "united kingdom", "england", "scotland", "wales"],
            "EU": [
                "eu",
                "ireland",
                "germany",
                "france",
                "italy",
                "spain",
                "netherlands",
            ],
            "US": ["us", "usa", "united states", "america"],
            "International": ["international", "global", "worldwide", "overseas"],
        },
        "example_classifications": [
            {
                "query": "Google Cloud from Ireland to UK",
                "expected": {
                    "business_type": "B2B",
                    "service_type": "digital",
                    "supplier_location": "eu",
                    "customer_location": "uk",
                    "is_cross_border": True,
                },
            }
        ],
    }


# 0) Initialize embedders
text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")

# 1) Load your store & retriever
store = InMemoryDocumentStore()

# LOAD DOCUMENTS: Load enhanced documents with metadata
documents = load_enhanced_documents_from_json()
if documents:
    store.write_documents(documents)
    print(f"📚 Document store populated with {len(documents)} enhanced documents")
else:
    print("⚠️  No documents loaded. VAT classification may not work properly.")

retriever = InMemoryEmbeddingRetriever(document_store=store)

# 2) Enhanced prompt with classification context
prompt_template = [
    ChatMessage.from_user(
        """
    You are an expert in UK VAT with access to HMRC guidance documents.

    {% if classification_context %}
    QUERY CLASSIFICATION CONTEXT:
    - Business Type: {{ classification_context.business_type }}
    - Service Type: {{ classification_context.service_type }}
    - Supplier Location: {{ classification_context.supplier_location }}
    - Customer Location: {{ classification_context.customer_location }}
    - Cross-border Transaction: {{ classification_context.is_cross_border }}
    - Confidence: {{ classification_context.confidence_score }}
    {% if classification_context.ambiguities %}
    - Ambiguities: {{ classification_context.ambiguities | join(", ") }}
    {% endif %}
    {% endif %}

    Context from relevant VAT documents:
    {% for doc in documents %}
      Document Content: {{ doc.content }}
      {% if doc.meta and doc.meta.get('vat_metadata') %}
      Document Metadata:
      - Transaction Types: {{ doc.meta.vat_metadata.transaction_types | join(", ") }}
      - Service Categories: {{ doc.meta.vat_metadata.service_categories | join(", ") }}
      - VAT Treatments: {{ doc.meta.vat_metadata.vat_treatments | join(", ") }}
      {% endif %}
      {% if doc.meta and doc.meta.get('url') %}
      Source URL: {{ doc.meta.get('url') }}
      {% endif %}
      ---
    {% endfor %}

    Based on the classification context and relevant documents above, analyze the following transaction.
    Transaction: {{query}}

    {% if classification_context and classification_context.business_type == 'B2B' and classification_context.service_type == 'digital' and classification_context.is_cross_border %}
    SPECIAL FOCUS: This appears to be a B2B digital service cross-border transaction. Pay particular attention to:
    - Reverse charge procedures (customer may account for VAT)
    - Place of supply rules for digital services
    - Post-Brexit UK-EU specific rules
    {% endif %}

    Provide the following:
    1) VAT Rate (e.g., 20%, 5%, 0%, exempt, out_of_scope)
    2) Short Rationale explaining why this rate applies
    3) Source reference (if identified from the documents, otherwise N/A)
    4) Confidence (low/medium/high)
    {% if classification_context and classification_context.ambiguities %}
    5) Clarifying questions needed: {{ classification_context.ambiguities | join(", ") }}
    {% endif %}

    Be specific about the VAT treatment and reference relevant HMRC guidance.
    """
    )
]
prompt_builder = PromptBuilder(
    template=prompt_template, required_variables=["query", "documents"]
)
generator = OpenAIChatGenerator(
    model="gpt-4o-mini", api_key=Secret.from_env_var("OPENAI_API_KEY")
)

# 3) Assemble enhanced pipeline
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
def get_enhanced_classification(text: str, use_classification: bool = True) -> str:
    """Enhanced cached function to get VAT classification with optional metadata filtering."""

    classification_result = None
    if use_classification:
        # Step 1: Classify the query
        classification = query_classifier.classify_query(text)
        classification_result = {
            "business_type": classification.business_type,
            "service_type": classification.service_type,
            "supplier_location": classification.supplier_location,
            "customer_location": classification.customer_location,
            "is_cross_border": classification.is_cross_border,
            "confidence_score": f"{classification.confidence_score:.2f}",
            "ambiguities": classification.ambiguities,
        }

        # Step 2: Filter documents based on classification
        all_documents = store.filter_documents()
        filtered_documents = filter_documents_by_classification(
            all_documents, classification_result
        )

        # Step 3: Update retriever with filtered documents
        filtered_store = InMemoryDocumentStore()
        filtered_store.write_documents(filtered_documents)
        filtered_retriever = InMemoryEmbeddingRetriever(document_store=filtered_store)

        # Update pipeline with filtered retriever
        pipe.add_component(instance=filtered_retriever, name="retriever")
        pipe.connect("text_embedder.embedding", "retriever.query_embedding")
        pipe.connect("retriever.documents", "prompter.documents")

    # Run pipeline
    pipeline_input = {
        "text_embedder": {"text": text},
        "retriever": {"top_k": 5},
        "prompter": {"query": text, "classification_context": classification_result},
    }

    result = pipe.run(pipeline_input)
    response = result["generator"]["replies"][0].text

    # Add classification info to response if available
    if classification_result:
        response += f"\n\n[Classification: {classification_result['business_type']} {classification_result['service_type']} transaction, Confidence: {classification_result['confidence_score']}]"

    return response


@app.post("/classify")
def classify_enhanced(q: EnhancedQuery):
    """Enhanced VAT classification with optional metadata filtering"""
    response = get_enhanced_classification(q.text, q.use_classification)

    # Also return classification details if requested
    classification_details = None
    if q.use_classification:
        classification = query_classifier.classify_query(q.text)
        classification_details = {
            "business_type": classification.business_type,
            "service_type": classification.service_type,
            "supplier_location": classification.supplier_location,
            "customer_location": classification.customer_location,
            "is_cross_border": classification.is_cross_border,
            "confidence_score": classification.confidence_score,
            "ambiguities": classification.ambiguities,
        }

    return {
        "response": response,
        "classification": classification_details,
        "enhanced_filtering": q.use_classification,
    }


@app.post("/classify-simple")
def classify_simple(q: Query):
    """Simple classification without enhanced filtering (backwards compatibility)"""
    pipeline_input = {
        "text_embedder": {"text": q.text},
        "retriever": {"top_k": 5},
        "prompter": {"query": q.text},
    }
    result = pipe.run(pipeline_input)
    return {"response": result["generator"]["replies"][0].text}


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Enhanced TaxCat API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
