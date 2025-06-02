#!/usr/bin/env python3
"""
Test script to verify document ingestion and retrieval functionality.
This helps debug what chunks are being retrieved for different queries.
"""

from dotenv import load_dotenv
from haystack import Pipeline
from haystack.components.converters import HTMLToDocument
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Load environment variables
load_dotenv()


def setup_document_store():
    """Set up and populate the document store with VAT documents."""
    print("🔄 Setting up document store...")

    # Initialize document store
    store = InMemoryDocumentStore()

    # Define ingestion components
    converter = HTMLToDocument()
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter(split_length=150, split_overlap=20)
    document_embedder = OpenAIDocumentEmbedder()
    writer = DocumentWriter(document_store=store)

    # Build ingestion pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=converter, name="converter")
    indexing_pipeline.add_component(instance=cleaner, name="cleaner")
    indexing_pipeline.add_component(instance=splitter, name="splitter")
    indexing_pipeline.add_component(instance=document_embedder, name="embedder")
    indexing_pipeline.add_component(instance=writer, name="writer")

    # Connect ingestion pipeline
    indexing_pipeline.connect("converter.documents", "cleaner.documents")
    indexing_pipeline.connect("cleaner.documents", "splitter.documents")
    indexing_pipeline.connect("splitter.documents", "embedder.documents")
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    # Process VAT documents
    vat_files = [
        "data/VAT Notice 700.html",
        "data/VAT Notice 700_12.html",
        "data/VAT Notice 700_21.html",
    ]

    print("📚 Ingesting VAT documents...")
    indexing_pipeline.run(data={"converter": {"sources": vat_files}})
    print("✔️ Document ingestion complete!")

    return store


def setup_retrieval_pipeline(document_store):
    """Set up the retrieval pipeline."""
    text_embedder = OpenAITextEmbedder()
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)

    # Build retrieval pipeline
    pipeline = Pipeline()
    pipeline.add_component(instance=text_embedder, name="text_embedder")
    pipeline.add_component(instance=retriever, name="retriever")

    # Connect pipeline
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    return pipeline


def test_retrieval(pipeline, query, top_k=5):
    """Test retrieval for a specific query."""
    print(f"\n🔍 Testing query: '{query}'")
    print("=" * 60)

    # Run retrieval
    result = pipeline.run(
        {"text_embedder": {"text": query}, "retriever": {"top_k": top_k}}
    )

    documents = result["retriever"]["documents"]

    if not documents:
        print("❌ No documents retrieved!")
        return

    print(f"📄 Retrieved {len(documents)} documents:")

    for i, doc in enumerate(documents, 1):
        print(f"\n--- Document {i} ---")
        print(f"📊 Score: {doc.score:.4f}")
        print(
            f"📝 Content: {doc.content[:200]}{'...' if len(doc.content) > 200 else ''}"
        )

        # Show metadata if available
        if doc.meta:
            print(f"🏷️  Metadata: {doc.meta}")

        print("-" * 40)


def main():
    """Main testing function."""
    print("🐱 TaxCat - Document Retrieval Test")
    print("=" * 40)

    # Set up document store and retrieval
    document_store = setup_document_store()
    retrieval_pipeline = setup_retrieval_pipeline(document_store)

    # Get document count
    all_docs = document_store.filter_documents()
    print(f"📊 Total documents in store: {len(all_docs)}")

    # Test queries
    test_queries = [
        "office supplies",
        "VAT rate 20%",
        "exempt goods",
        "zero rate",
        "restaurant food",
        "books and magazines",
        "medical equipment",
        "construction services",
    ]

    print("\n🧪 Running test queries...")

    for query in test_queries:
        test_retrieval(retrieval_pipeline, query, top_k=3)
        input("\nPress Enter to continue to next query...")

    # Interactive mode
    print("\n🔧 Interactive mode - Enter your own queries:")
    print("(Type 'quit' to exit)")

    while True:
        query = input("\n💬 Enter query: ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            break

        if query:
            test_retrieval(retrieval_pipeline, query, top_k=5)


if __name__ == "__main__":
    main()
