# app.py
import json
import os
import logging # Import logging
from contextlib import asynccontextmanager # For FastAPI lifespan
from typing import Dict, List

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the ingestion script function
try:
    from scripts.enhanced_ingest import process_and_embed_documents
    logger.info("Successfully imported process_and_embed_documents from scripts.enhanced_ingest")
except ImportError as e:
    logger.error(f"Failed to import process_and_embed_documents: {e}", exc_info=True)
    process_and_embed_documents = None

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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
from haystack.core.component import component

# Global variable to hold the document store and retriever
# This will be initialized during app startup via lifespan management
document_store = None
retriever_component = None
pipeline_initialized = False

# Only load .env file in development
if os.getenv("ENVIRONMENT") != "production":
    logger.info("Development environment detected, loading .env file.")
    load_dotenv()

# Initialize joblib memory cache
memory = Memory("cache", verbose=0)

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("CRITICAL: OPENAI_API_KEY environment variable is not set")
    raise ValueError("OPENAI_API_KEY environment variable is required")

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://taxcat.ai")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DOCUMENTS_FILE_PATH = "vat_documents_enhanced.json" # Centralized path


class Query(BaseModel):
    text: str

def load_documents_from_json_to_store(store: InMemoryDocumentStore, json_file=DOCUMENTS_FILE_PATH):
    """Load documents from JSON file into the provided DocumentStore."""
    logger.info(f"📂 Attempting to load documents from {json_file}...")
    if not os.path.exists(json_file):
        logger.warning(f"❌ Document file {json_file} not found. Will attempt to generate it.")
        return False

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            documents_data = json.load(f)
        if not documents_data:
            logger.warning(f"⚠️ No data found in {json_file}. Will attempt to generate it.")
            return False

        documents_to_write = []
        for doc_data in documents_data:
            doc = Document(
                id=doc_data.get("id"),
                content=doc_data["content"],
                meta=doc_data.get("meta", {}),
                embedding=doc_data.get("embedding")
            )
            documents_to_write.append(doc)

        store.write_documents(documents_to_write)
        logger.info(f"✅ Loaded and wrote {len(documents_to_write)} documents to the store from {json_file}")
        return True
    except json.JSONDecodeError:
        logger.error(f"❌ Error: Invalid JSON in {json_file}. Will attempt to regenerate it.")
        return False
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred while loading documents from {json_file}: {e}", exc_info=True)
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global document_store, retriever_component, pipeline_initialized, pipe, text_embedder, prompt_builder, generator
    logger.info("🚀 FastAPI application startup...")

    if not OPENAI_API_KEY:
        logger.error("CRITICAL: OPENAI_API_KEY is not configured. VAT classification will not work.")
        pipeline_initialized = False
        yield
        return

    try:
        # Test the API key with a simple request
        test_embedder = OpenAITextEmbedder(model="text-embedding-3-small", api_key=Secret.from_token(OPENAI_API_KEY))
        test_embedder.run(text="test")
        logger.info("✅ OpenAI API key validated successfully")
    except Exception as e:
        logger.error(f"❌ OpenAI API key validation failed: {e}")
        pipeline_initialized = False
        yield
        return

    logger.info(f"OpenAI API Key loaded, last 4 chars: ...{OPENAI_API_KEY[-4:]}")
    
    # Initialize Document Store
    document_store = InMemoryDocumentStore()

    # First try to load existing documents
    documents_loaded = load_documents_from_json_to_store(document_store, DOCUMENTS_FILE_PATH)
    
    # If loading failed, try to run the ingestion process
    if not documents_loaded and process_and_embed_documents:
        logger.info("🏃 Running document ingestion and embedding process...")
        enhanced_docs, stats = process_and_embed_documents()
        if enhanced_docs is not None:
            logger.info("✅ Document ingestion and embedding process completed successfully.")
            # Try loading the newly generated documents
            documents_loaded = load_documents_from_json_to_store(document_store, DOCUMENTS_FILE_PATH)
        else:
            logger.error("❌ Document ingestion and embedding process failed.")
    
    if document_store.count_documents() > 0:
        logger.info(f"📚 Document store populated with {document_store.count_documents()} documents.")
    else:
        logger.error("❌ Document store is empty. VAT classification will not work properly.")
        pipeline_initialized = False
        yield
        return

    # Initialize Haystack RAG Pipeline components
    try:
        logger.info("⚙️ Initializing Haystack RAG pipeline components...")
        text_embedder = OpenAITextEmbedder(model="text-embedding-3-small", api_key=Secret.from_token(OPENAI_API_KEY))
        retriever_component = InMemoryEmbeddingRetriever(document_store=document_store)
        
        prompt_template_str = """
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
        prompt_builder = PromptBuilder(template=prompt_template_str)
        prompt_to_messages = PromptToMessages()
        generator = OpenAIChatGenerator(model="gpt-4o-mini", api_key=Secret.from_token(OPENAI_API_KEY))

        pipe = Pipeline()
        pipe.add_component(instance=text_embedder, name="text_embedder")
        pipe.add_component(instance=retriever_component, name="retriever")
        pipe.add_component(instance=prompt_builder, name="prompter")
        pipe.add_component(instance=prompt_to_messages, name="prompt_to_messages")
        pipe.add_component(instance=generator, name="generator")

        pipe.connect("text_embedder.embedding", "retriever.query_embedding")
        pipe.connect("retriever.documents", "prompter.documents")
        pipe.connect("prompter.prompt", "prompt_to_messages.prompt")
        pipe.connect("prompt_to_messages.messages", "generator.messages")
        
        pipeline_initialized = True
        logger.info("✅ Haystack RAG pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Haystack RAG pipeline: {e}", exc_info=True)
        pipeline_initialized = False
        yield
        return

    yield

    logger.info("💧 FastAPI application shutdown...")
    # Clean up resources if needed (e.g., close database connections)


app = FastAPI(
    title="TaxCat VAT Classification API",
    description="API for classifying UK VAT transactions",
    version="1.0.0",
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if ENVIRONMENT != "production" else None,
    lifespan=lifespan # Use the lifespan context manager
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://taxcat.ai",
        "https://www.taxcat.ai"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "🐱 Welcome to TaxCat - UK VAT Classification API",
        "status": "healthy" if pipeline_initialized else "initializing",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "documents_in_store": document_store.count_documents() if document_store else 0,
    }

@app.get("/health")
def health_check():
    # Simplified health check for Railway debugging
    logger.info("Simplified /health endpoint hit!")
    return {
        "status": "ok_from_simplified_health_check",
        "message": "Application is running, ignoring pipeline and doc status for this check."
    }

@memory.cache
def get_classification_from_pipeline(text: str) -> str:
    """Cached function to get VAT classification using the RAG pipeline."""
    if not pipeline_initialized or not pipe:
        logger.error("Pipeline not ready for get_classification_from_pipeline.")
        # Fallback response when pipeline isn't ready
        return "VAT classification system is currently initializing or encountered an error. Please try again shortly. If the problem persists, check system logs."
    
    try:
        pipeline_input = {
            "text_embedder": {"text": text},
            "retriever": {"top_k": 5},
            "prompter": {"query": text}, # Ensure prompter also gets the query directly if template needs it
        }
        logger.info(f"Running RAG pipeline with input: {text[:50]}...")
        result = pipe.run(pipeline_input)
        response_text = result["generator"]["replies"][0].content
        logger.info(f"RAG pipeline generated response: {response_text[:100]}...")
        return response_text
    except Exception as e:
        logger.error(f"Error during RAG pipeline execution for query '{text[:50]}...': {e}", exc_info=True)
        return "An error occurred while classifying the transaction. Please check logs or try again."

@app.post("/classify")
def classify(q: Query):
    if not pipeline_initialized:
        logger.warning("Classification attempt while pipeline is not ready.")
        raise HTTPException(status_code=503, detail="The VAT classification system is initializing. Please try again in a moment.")
    
    if document_store and document_store.count_documents() == 0:
        logger.warning("Classification attempt but no documents are in the store.")
        raise HTTPException(status_code=503, detail="The document store is currently empty. Classification may be impaired or based on general knowledge only.")
    
    try:
        response_text = get_classification_from_pipeline(q.text)
        return {"response": response_text, "status": "ok"}
    except Exception as e:
        logger.error(f"Error during classification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while classifying the transaction. Please try again later.")

@component
class PromptToMessages:
    @component.output_types(messages=List[ChatMessage])
    def run(self, prompt: str) -> Dict[str, List[ChatMessage]]:
        print("PromptToMessages.run CALLED with:", prompt)
        result = {"messages": [ChatMessage.from_user(prompt)]}
        print("PromptToMessages.run RETURNING:", result)
        return result

# Ensure the main execution block is only for direct script running, not when imported by uvicorn.
if __name__ == "__main__":
    # This block is NOT executed when Uvicorn runs the app.
    # Uvicorn imports `app` and runs it.
    # For local development testing of this script directly (which is unusual for FastAPI apps):
    logger.info("Running app.py directly (for testing, not for Uvicorn deployment).")
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000) # Example, not standard for deployment script
