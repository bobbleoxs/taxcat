# scripts/enhanced_ingest.py
import json
import os
from dataclasses import asdict

from chunk_analyzer import VATChunkAnalyzer
from dotenv import load_dotenv
from haystack import Pipeline  # pipeline abstraction
from haystack.components.converters import HTMLToDocument  # html → Document
from haystack.components.embedders import OpenAIDocumentEmbedder  # embedding
from haystack.components.preprocessors import (  # clean & split
    DocumentCleaner,
    DocumentSplitter,
)
from haystack.components.writers import DocumentWriter  # write to store
from haystack.document_stores.in_memory import InMemoryDocumentStore  # store
from haystack.utils import Secret

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
    load_dotenv()

# Hardcoded API key for now
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


def run_enhanced_ingest():
    """Run the enhanced ingestion pipeline with metadata analysis"""

    print("🚀 Starting Enhanced VAT Document Ingestion with Metadata Analysis...")

    # 1) Init document store and chunk analyzer
    store = InMemoryDocumentStore()
    chunk_analyzer = VATChunkAnalyzer()

    # 2) Define components
    converter = HTMLToDocument()
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter(split_length=150, split_overlap=20)
    embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small", api_key=Secret.from_token(OPENAI_API_KEY))
    writer = DocumentWriter(document_store=store)

    # 3) Build and run indexing pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", converter)
    indexing_pipeline.add_component("cleaner", cleaner)
    indexing_pipeline.add_component("splitter", splitter)
    indexing_pipeline.add_component("embedder", embedder)
    indexing_pipeline.add_component("writer", writer)

    # Connect the components
    indexing_pipeline.connect("converter.documents", "cleaner.documents")
    indexing_pipeline.connect("cleaner.documents", "splitter.documents")
    indexing_pipeline.connect("splitter.documents", "embedder.documents")
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    # 4) Point at your HTML files (including new VAT notices)
    files = [
        "data/VAT Notice 700.html",
        "data/VAT Notice 700_12.html",
        "data/VAT Notice 700_21.html",
        "data/VAT Notice 741A.html",  # Place of supply of services
        "data/VAT Notice 735.html",  # Domestic reverse charge procedure
    ]

    print("📚 Processing documents through pipeline...")
    indexing_pipeline.run(data={"converter": {"sources": files}})

    # 5) Get all documents from store and enhance with metadata
    print("🔍 Analyzing chunks and adding metadata...")
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
        # Enhance document with metadata
        enhanced_doc = enhance_document_with_metadata(doc, chunk_analyzer)
        enhanced_documents.append(enhanced_doc)

        # Update statistics
        metadata_stats["total_chunks"] += 1
        vat_meta = enhanced_doc.meta.get("vat_metadata", {})

        if "B2B" in vat_meta.get("transaction_types", []):
            metadata_stats["b2b_chunks"] += 1
        if "B2C" in vat_meta.get("transaction_types", []):
            metadata_stats["b2c_chunks"] += 1
        if "digital" in vat_meta.get("service_categories", []):
            metadata_stats["digital_chunks"] += 1
        if "professional" in vat_meta.get("service_categories", []):
            metadata_stats["professional_chunks"] += 1
        if "physical" in vat_meta.get("service_categories", []):
            metadata_stats["physical_chunks"] += 1
        if "reverse_charge" in vat_meta.get("vat_treatments", []):
            metadata_stats["reverse_charge_chunks"] += 1
        if vat_meta.get("confidence_score", 0) >= 0.8:
            metadata_stats["high_confidence_chunks"] += 1

    # 6) Save the enhanced documents to disk for persistence
    print("💾 Saving enhanced documents to disk...")
    documents_data = []

    for doc in enhanced_documents:
        doc_data = {
            "id": doc.id,
            "content": doc.content,
            "meta": doc.meta,
            "embedding": doc.embedding if doc.embedding is not None else None,
        }
        documents_data.append(doc_data)

    with open("vat_documents_enhanced.json", "w", encoding="utf-8") as f:
        json.dump(documents_data, f, ensure_ascii=False, indent=2)

    # 7) Print statistics
    print("\n📊 Enhanced Ingestion Statistics:")
    print("=" * 50)
    print(f"Total Chunks Processed: {metadata_stats['total_chunks']}")
    print(f"B2B Transaction Chunks: {metadata_stats['b2b_chunks']}")
    print(f"B2C Transaction Chunks: {metadata_stats['b2c_chunks']}")
    print(f"Digital Service Chunks: {metadata_stats['digital_chunks']}")
    print(f"Professional Service Chunks: {metadata_stats['professional_chunks']}")
    print(f"Physical Goods Chunks: {metadata_stats['physical_chunks']}")
    print(f"Reverse Charge Chunks: {metadata_stats['reverse_charge_chunks']}")
    print(f"High Confidence Chunks (≥0.8): {metadata_stats['high_confidence_chunks']}")

    print("\n✅ Enhanced ingestion & indexing complete.")
    print(
        f"📁 Saved {len(documents_data)} enhanced documents to 'vat_documents_enhanced.json'"
    )

    # 8) Show some example enhanced chunks
    print("\n🔍 Sample Enhanced Chunks:")
    print("=" * 50)

    # Show a few examples of enhanced chunks
    sample_count = min(3, len(enhanced_documents))
    for i in range(sample_count):
        doc = enhanced_documents[i]
        vat_meta = doc.meta.get("vat_metadata", {})

        print(f"\n📄 Sample Chunk {i+1}:")
        print(f"Content: {doc.content[:150]}...")
        print(f"Transaction Types: {vat_meta.get('transaction_types', [])}")
        print(f"Geographic Scope: {vat_meta.get('geographic_scope', [])}")
        print(f"Service Categories: {vat_meta.get('service_categories', [])}")
        print(f"VAT Treatments: {vat_meta.get('vat_treatments', [])}")
        print(f"Confidence: {vat_meta.get('confidence_score', 0):.2f}")
        print("-" * 50)

    return enhanced_documents, metadata_stats


# Demo filtering function
def demo_metadata_filtering(enhanced_documents, chunk_analyzer):
    """Demonstrate how metadata filtering works"""

    print("\n🎯 Demo: Metadata-Based Filtering")
    print("=" * 50)

    # Prepare chunks for filtering
    chunks = []
    for doc in enhanced_documents:
        chunks.append(
            {
                "content": doc.content,
                "metadata": doc.meta.get("vat_metadata", {}),
                "id": doc.id,
            }
        )

    # Test different query classifications
    test_classifications = [
        {
            "name": "B2B Digital EU→UK",
            "classification": {
                "business_type": "B2B",
                "service_type": "digital",
                "supplier_location": "eu",
                "customer_location": "uk",
                "is_cross_border": True,
            },
        },
        {
            "name": "B2C Physical UK Domestic",
            "classification": {
                "business_type": "B2C",
                "service_type": "physical",
                "supplier_location": "uk",
                "customer_location": "uk",
                "is_cross_border": False,
            },
        },
    ]

    for test in test_classifications:
        print(f"\n🔍 Filter Test: {test['name']}")
        print(f"Classification: {test['classification']}")

        # Filter chunks
        filtered_chunks = chunk_analyzer.filter_chunks_by_classification(
            chunks, test["classification"]
        )

        print(f"📊 Results: {len(filtered_chunks)} relevant chunks found")

        # Show top 2 most relevant chunks
        top_chunks = filtered_chunks[:2]
        for i, chunk in enumerate(top_chunks, 1):
            print(f"\nTop {i} (Score: {chunk.get('relevance_score', 0)}):")
            print(f"Content: {chunk['content'][:100]}...")
            print(f"Metadata: {chunk['metadata']}")

        print("-" * 50)


if __name__ == "__main__":
    # Run enhanced ingestion
    enhanced_docs, stats = run_enhanced_ingest()

    # Demo the filtering
    chunk_analyzer = VATChunkAnalyzer()
    demo_metadata_filtering(enhanced_docs, chunk_analyzer)
