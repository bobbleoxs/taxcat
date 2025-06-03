# scripts/enhanced_ingest.py
import json
import os
from dataclasses import asdict

from chunk_analyzer import VATChunkAnalyzer
from dotenv import load_dotenv
from haystack import Pipeline  # pipeline abstraction
from haystack.components.converters import HTMLToDocument  # html → Document
from haystack.components.embedders import OpenAITextEmbedder  # embedding
from haystack.components.preprocessors import (  # clean & split
    DocumentCleaner,
    DocumentSplitter,
)
from haystack.components.writers import DocumentWriter  # write to store
from haystack.document_stores.in_memory import InMemoryDocumentStore  # store
from haystack.utils import Secret
import logging # Import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# INSTRUCTIONS:
# 1. Download the following HMRC VAT notices as HTML and place them in the data/ directory:
#    - VAT Notice 700 (The VAT Guide)
#    - VAT Notice 700_12 (How to fill in VAT returns)
#    - VAT Notice 700_21 (Keeping VAT records)
#    - VAT Notice 741A (Place of supply of services):
#      https://www.gov.uk/guidance/vat-place-of-supply-of-services-notice-741a
#    - VAT Notice 735 (Domestic reverse charge procedure):
#      https://www.gov.uk/guidance/the-vat-domestic-reverse-charge-procedure-notice-735
# 2. After downloading, run this script to regenerate embeddings and update the knowledge base.
# 3. The classification API should prioritize:
#    - Post-Brexit rules for cross-border digital services
#    - B2B digital services from EU to UK = reverse charge (out of scope for supplier)
#    - Consistent application of place-of-supply rules

# Only load .env file in development
if os.getenv("ENVIRONMENT") != "production":
    logger.info("Development environment detected, loading .env file.")
    load_dotenv()

# Hardcoded API key for now -  THIS IS FOR TEMPORARY DEBUGGING.
# Ideally, this should be handled by environment variables passed at runtime.
OPENAI_API_KEY = "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"


def enhance_document_with_metadata(doc, chunk_analyzer):
    """
    Enhance a document chunk with metadata analysis

    Args:
        doc: Haystack Document object
        chunk_analyzer: VATChunkAnalyzer instance

    Returns:
        Document with enhanced metadata
    """
    # Analyze the chunk content
    metadata = chunk_analyzer.analyze_chunk(doc.content)

    # Add metadata to the document's meta dictionary
    doc.meta.update(
        {
            "vat_metadata": asdict(metadata),
            "transaction_types": metadata.transaction_types,
            "geographic_scope": metadata.geographic_scope,
            "service_categories": metadata.service_categories,
            "vat_treatments": metadata.vat_treatments,
            "confidence_score": metadata.confidence_score,
        }
    )

    return doc


def process_and_embed_documents():
    """Main function to run the enhanced ingestion pipeline with metadata analysis."""
    try:
        logger.info("🚀 Starting Enhanced VAT Document Ingestion with Metadata Analysis...")
        # Ensure API key is available, even if hardcoded for now
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set. Ingestion cannot proceed.")
            return None, None # Or raise an error
        logger.info(f"DEBUG: Using OpenAI API Key ending with: {OPENAI_API_KEY[-4:]}")

        # 1) Init document store and chunk analyzer
        store = InMemoryDocumentStore()
        chunk_analyzer = VATChunkAnalyzer()

        # 2) Define components
        converter = HTMLToDocument()
        cleaner = DocumentCleaner()
        splitter = DocumentSplitter(split_length=150, split_overlap=20)
        # Ensure Secret.from_token is used correctly with the API key
        embedder = OpenAITextEmbedder(model="text-embedding-3-small", api_key=Secret.from_token(OPENAI_API_KEY))
        writer = DocumentWriter(document_store=store)

        # 3) Build and run indexing pipeline
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("converter", converter)
        indexing_pipeline.add_component("cleaner", cleaner)
        indexing_pipeline.add_component("splitter", splitter)
        indexing_pipeline.add_component("embedder", embedder)
        indexing_pipeline.add_component("writer", writer)

        indexing_pipeline.connect("converter.documents", "cleaner.documents")
        indexing_pipeline.connect("cleaner.documents", "splitter.documents")
        indexing_pipeline.connect("splitter.documents", "embedder.documents")
        indexing_pipeline.connect("embedder.documents", "writer.documents")

        files = [
            "data/VAT Notice 700.html",
            "data/VAT Notice 700_12.html",
            "data/VAT Notice 700_21.html",
            "data/VAT Notice 741A.html",
            "data/VAT Notice 735.html",
        ]
        # Check if files exist
        for f_path in files:
            if not os.path.exists(f_path):
                logger.warning(f"File not found: {f_path}. Skipping.")
                # Decide if this is critical, for now, we'll let it try and fail if converter needs it

        logger.info("📚 Processing documents through pipeline...")
        indexing_pipeline.run(data={"converter": {"sources": files}})

        logger.info("🔍 Analyzing chunks and adding metadata...")
        all_documents = store.filter_documents()
        enhanced_documents = []
        metadata_stats = {
            "total_chunks": 0,
            "b2b_chunks": 0,
            "b2c_chunks": 0,
            "digital_chunks": 0,
            "professional_chunks": 0,
            "physical_chunks": 0,
            "reverse_charge_chunks": 0,
            "high_confidence_chunks": 0,
        }

        for doc in all_documents:
            enhanced_doc = enhance_document_with_metadata(doc, chunk_analyzer)
            enhanced_documents.append(enhanced_doc)
            metadata_stats["total_chunks"] += 1
            vat_meta = enhanced_doc.meta.get("vat_metadata", {})
            if "B2B" in vat_meta.get("transaction_types", []): metadata_stats["b2b_chunks"] += 1
            if "B2C" in vat_meta.get("transaction_types", []): metadata_stats["b2c_chunks"] += 1
            if "digital" in vat_meta.get("service_categories", []): metadata_stats["digital_chunks"] += 1
            if "professional" in vat_meta.get("service_categories", []): metadata_stats["professional_chunks"] += 1
            if "physical" in vat_meta.get("service_categories", []): metadata_stats["physical_chunks"] += 1
            if "reverse_charge" in vat_meta.get("vat_treatments", []): metadata_stats["reverse_charge_chunks"] += 1
            if vat_meta.get("confidence_score", 0) >= 0.8: metadata_stats["high_confidence_chunks"] += 1

        logger.info("💾 Saving enhanced documents to disk...")
        documents_data = []
        for doc in enhanced_documents:
            doc_data = {
                "id": doc.id,
                "content": doc.content,
                "meta": doc.meta,
                "embedding": doc.embedding if doc.embedding is not None else None,
            }
            documents_data.append(doc_data)

        output_json_path = "vat_documents_enhanced.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(documents_data, f, ensure_ascii=False, indent=2)

        # 7) Print statistics
        logger.info("\n Enhanced Ingestion Statistics:")
        logger.info("=" * 50)
        logger.info(f"Total Chunks Processed: {metadata_stats['total_chunks']}")
        logger.info(f"B2B Transaction Chunks: {metadata_stats['b2b_chunks']}")
        logger.info(f"B2C Transaction Chunks: {metadata_stats['b2c_chunks']}")
        logger.info(f"Digital Service Chunks: {metadata_stats['digital_chunks']}")
        logger.info(f"Professional Service Chunks: {metadata_stats['professional_chunks']}")
        logger.info(f"Physical Goods Chunks: {metadata_stats['physical_chunks']}")
        logger.info(f"Reverse Charge Chunks: {metadata_stats['reverse_charge_chunks']}")
        logger.info(f"High Confidence Chunks (≥0.8): {metadata_stats['high_confidence_chunks']}")

        logger.info("\n✅ Enhanced ingestion & indexing complete.")
        logger.info(f"📁 Saved {len(documents_data)} enhanced documents to '{output_json_path}'")
        return enhanced_documents, metadata_stats

    except Exception as e:
        logger.error(f"Error during document ingestion: {e}", exc_info=True)
        # Depending on the desired behavior, you might want to re-raise the exception
        # or return a specific value indicating failure.
        return None, None


# Remove demo_metadata_filtering or ensure it's not called directly on import
# def demo_metadata_filtering(enhanced_documents, chunk_analyzer):
#     ...

if __name__ == "__main__":
    # This part is for direct script execution (e.g., local testing)
    # It will not run when the module is imported by FastAPI.
    enhanced_docs, stats = process_and_embed_documents()
    if enhanced_docs:
        logger.info("Script executed directly: Ingestion successful.")
        # Optionally, run demo filtering if needed for direct testing
        # chunk_analyzer = VATChunkAnalyzer()
        # demo_metadata_filtering(enhanced_docs, chunk_analyzer)
    else:
        logger.error("Script executed directly: Ingestion failed.")
