# debug_rag_pipeline.py
import json
import sys
from typing import Dict, List

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from haystack.core.component import component

# Add scripts directory to path for imports
sys.path.append("scripts")

from chunk_analyzer import VATChunkAnalyzer
from query_classifier import VATQueryClassifier


def classify_query(query: str) -> dict:
    """Classify the query into business type, service type, etc."""
    query_lower = query.lower()

    # Business type - default to B2B for business-related queries
    if any(k in query_lower for k in ["b2c", "consumer", "individual", "personal"]):
        business_type = "b2c"
    else:
        business_type = "b2b"  # Default to B2B for business queries

    # Service type
    if any(
        k in query_lower
        for k in ["digital", "software", "cloud", "subscription", "online", "e-service"]
    ):
        service_type = "digital"
    elif any(k in query_lower for k in ["goods", "physical", "tangible"]):
        service_type = "goods"
    else:
        service_type = "unknown"

    # Direction
    if "to uk" in query_lower or "to united kingdom" in query_lower:
        direction = "to_uk"
    elif "from uk" in query_lower or "from united kingdom" in query_lower:
        direction = "from_uk"
    else:
        direction = "unknown"

    # Source country
    source_country = None
    for country in ["ireland", "france", "germany", "spain", "italy", "netherlands"]:
        if f"from {country}" in query_lower:
            source_country = country
            break

    return {
        "business_type": business_type,
        "service_type": service_type,
        "direction": direction,
        "source_country": source_country,
    }


def load_documents_from_json(json_file="vat_documents.json"):
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
        print(
            f"❌ Error: {json_file} not found. Please run 'python scripts/ingest.py' first."
        )
        return []
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in {json_file}")
        return []


def debug_pipeline(query: str):
    # Initialize components
    text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
    store = InMemoryDocumentStore()
    documents = load_documents_from_json()
    if documents:
        store.write_documents(documents)
        print(f"📚 Document store populated with {len(documents)} documents")
    else:
        print("⚠️  No documents loaded. VAT classification may not work properly.")
        return

    # Classify query
    print("\n1️⃣ Step 1: Classifying query...")
    query_classification = classify_query(query)
    print(f"Query classification: {json.dumps(query_classification, indent=2)}")

    # Build metadata filter
    filters = None
    if query_classification["service_type"] != "unknown":
        # Construct the type string
        type_parts = [
            query_classification["business_type"],
            query_classification["service_type"],
            query_classification["direction"],
        ]
        type_str = "_".join(type_parts)
        filters = {"type": type_str}  # Remove meta. prefix for in-memory store
        print(f"DEBUG: Using filter: {filters}")

    retriever = InMemoryEmbeddingRetriever(document_store=store)
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
    prompt_builder = PromptBuilder(template=prompt_template[0].content, required_variables=["query", "documents"])
    prompt_to_messages = PromptToMessages()
    generator = OpenAIChatGenerator(
        model="gpt-4o-mini", api_key=Secret.from_env_var("OPENAI_API_KEY")
    )

    # Create pipeline
    pipe = Pipeline()
    pipe.add_component(instance=text_embedder, name="text_embedder")
    pipe.add_component(instance=retriever, name="retriever")
    pipe.add_component(instance=prompt_builder, name="prompter")
    pipe.add_component(instance=prompt_to_messages, name="prompt_to_messages")
    pipe.add_component(instance=generator, name="generator")

    # Connect components
    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompter.documents")
    pipe.connect("prompter.prompt", "prompt_to_messages.prompt")
    pipe.connect("prompt_to_messages.messages", "generator.messages")

    # Run pipeline with debug output
    print(f"\n🔍 Debugging query: {query}")
    print("\n2️⃣ Step 2: Generating query embedding...")
    embedding_result = text_embedder.run(text=query)
    print("✅ Query embedding generated")

    print("\n3️⃣ Step 3: Retrieving relevant documents...")
    retrieval_result = retriever.run(
        query_embedding=embedding_result["embedding"],
        top_k=5,
        filters=filters if filters else None,
    )
    retrieved_docs = retrieval_result["documents"]
    print(f"📄 Retrieved {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\nDocument {i}:")
        print(f"ID: {doc.id}")
        print(f"Meta: {json.dumps(doc.meta, indent=2)}")
        print(f"Content preview: {doc.content[:200]}...")

    print("\n4️⃣ Step 4: Building prompt...")
    prompt_result = prompt_builder.run(query=query, documents=retrieved_docs)
    print("📝 Final prompt:")
    print(prompt_result["prompt"])

    print("\n5️⃣ Step 5: Generating response...")
    response = generator.run(messages=prompt_result["prompt"])
    print("\n🎯 Final response:")
    print(response["replies"][0].text)


def load_enhanced_documents():
    """Load the enhanced documents with metadata"""
    print("📂 Loading enhanced documents...")
    with open("vat_documents_enhanced.json", "r", encoding="utf-8") as f:
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

    print(f"✅ Loaded {len(documents)} enhanced documents")
    return documents


def analyze_classification(query: str, classifier: VATQueryClassifier) -> Dict:
    """Analyze query classification"""
    print(f"\n🔍 STEP 1: Query Classification for '{query}'")
    print("=" * 60)

    classification = classifier.classify_query(query)
    classification_dict = {
        "business_type": classification.business_type,
        "service_type": classification.service_type,
        "supplier_location": classification.supplier_location,
        "customer_location": classification.customer_location,
        "is_cross_border": classification.is_cross_border,
        "confidence_score": classification.confidence_score,
        "ambiguities": classification.ambiguities,
    }

    print(f"Business Type: {classification_dict['business_type']}")
    print(f"Service Type: {classification_dict['service_type']}")
    print(f"Supplier Location: {classification_dict['supplier_location']}")
    print(f"Customer Location: {classification_dict['customer_location']}")
    print(f"Cross-border: {classification_dict['is_cross_border']}")
    print(f"Confidence: {classification_dict['confidence_score']:.2f}")
    print(f"Ambiguities: {classification_dict['ambiguities']}")

    return classification_dict


def analyze_metadata_filtering(
    documents: List[Document], classification: Dict, chunk_analyzer: VATChunkAnalyzer
) -> List[Dict]:
    """Analyze metadata filtering step"""
    print("\n📊 STEP 2: Metadata Filtering Analysis")
    print("=" * 60)

    # Convert documents to chunks format
    chunks = []
    for doc in documents:
        vat_metadata = doc.meta.get("vat_metadata", {})
        chunks.append(
            {
                "content": doc.content,
                "metadata": vat_metadata,
                "id": doc.id,
                "document": doc,
            }
        )

    print(f"Total documents before filtering: {len(chunks)}")

    # Analyze sample metadata
    print("\n📋 Sample Metadata Analysis (first 5 docs):")
    for i, chunk in enumerate(chunks[:5]):
        metadata = chunk["metadata"]
        print(
            f"Doc {i+1}: {metadata.get('transaction_types', [])} | {metadata.get('service_categories', [])} | {metadata.get('geographic_scope', [])} | {metadata.get('vat_treatments', [])}"
        )

    # Check specifically for B2B digital metadata
    b2b_count = sum(
        1 for chunk in chunks if "B2B" in chunk["metadata"].get("transaction_types", [])
    )
    digital_count = sum(
        1
        for chunk in chunks
        if "digital" in chunk["metadata"].get("service_categories", [])
    )
    eu_count = sum(
        1 for chunk in chunks if "EU" in chunk["metadata"].get("geographic_scope", [])
    )
    international_count = sum(
        1
        for chunk in chunks
        if "international" in chunk["metadata"].get("geographic_scope", [])
    )
    reverse_charge_count = sum(
        1
        for chunk in chunks
        if "reverse_charge" in chunk["metadata"].get("vat_treatments", [])
    )

    print("\n🏷️ Metadata Statistics:")
    print(f"B2B chunks: {b2b_count}")
    print(f"Digital chunks: {digital_count}")
    print(f"EU chunks: {eu_count}")
    print(f"International chunks: {international_count}")
    print(f"Reverse charge chunks: {reverse_charge_count}")

    # Apply filtering
    filtered_chunks = chunk_analyzer.filter_chunks_by_classification(
        chunks, classification
    )
    print(
        f"\n🔍 After filtering: {len(filtered_chunks)} chunks (reduced by {len(chunks) - len(filtered_chunks)})"
    )

    # Analyze filtered chunks
    print("\n📋 Top 10 Filtered Chunks (by relevance score):")
    for i, chunk in enumerate(filtered_chunks[:10]):
        relevance = chunk.get("relevance_score", 0)
        metadata = chunk["metadata"]
        content_preview = chunk["content"][:100].replace("\n", " ")
        print(
            f"Chunk {i+1} (Score: {relevance:.1f}): {metadata.get('transaction_types', [])} | {metadata.get('service_categories', [])} | {metadata.get('vat_treatments', [])}"
        )
        print(f"   Content: {content_preview}...")
        print()

    return filtered_chunks


def analyze_embedding_retrieval(filtered_chunks: List[Dict], query: str) -> List[Dict]:
    """Analyze embedding-based retrieval"""
    print("\n🔎 STEP 3: Embedding Retrieval Analysis")
    print("=" * 60)

    # Create a fresh document store with filtered documents
    filtered_store = InMemoryDocumentStore()
    filtered_documents = [chunk["document"] for chunk in filtered_chunks]
    filtered_store.write_documents(filtered_documents)

    # Create embedder and retriever
    text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
    retriever = InMemoryEmbeddingRetriever(document_store=filtered_store)

    # Get query embedding
    print(f"Getting embedding for query: '{query}'")
    query_result = text_embedder.run(text=query)
    query_embedding = query_result["embedding"]

    # Retrieve top chunks
    retrieval_result = retriever.run(query_embedding=query_embedding, top_k=5)
    retrieved_docs = retrieval_result["documents"]

    print("\n📄 Top 5 Retrieved Documents (by embedding similarity):")
    for i, doc in enumerate(retrieved_docs):
        metadata = doc.meta.get("vat_metadata", {})
        score = doc.score if hasattr(doc, "score") else "N/A"
        content_preview = doc.content[:100].replace("\n", " ")
        print(
            f"Doc {i+1} (Similarity: {score}): {metadata.get('transaction_types', [])} | {metadata.get('service_categories', [])} | {metadata.get('vat_treatments', [])}"
        )
        print(f"   Content: {content_preview}...")
        print()

    return retrieved_docs


def analyze_prompt_construction(
    retrieved_docs: List[Document], query: str, classification: Dict
):
    """Analyze the final prompt construction"""
    print("\n📝 STEP 4: Prompt Construction Analysis")
    print("=" * 60)

    print(f"Query: {query}")
    print("Classification Context:")
    for key, value in classification.items():
        print(f"  {key}: {value}")

    print(f"\nDocuments in prompt context ({len(retrieved_docs)} docs):")
    for i, doc in enumerate(retrieved_docs):
        metadata = doc.meta.get("vat_metadata", {})
        print(f"Doc {i+1}: {metadata}")
        print(f"   Content: {doc.content[:150]}...")
        print()

    # Check for conflicting advice
    b2b_docs = [
        doc
        for doc in retrieved_docs
        if "B2B" in doc.meta.get("vat_metadata", {}).get("transaction_types", [])
    ]
    b2c_docs = [
        doc
        for doc in retrieved_docs
        if "B2C" in doc.meta.get("vat_metadata", {}).get("transaction_types", [])
    ]
    reverse_charge_docs = [
        doc
        for doc in retrieved_docs
        if "reverse_charge"
        in doc.meta.get("vat_metadata", {}).get("vat_treatments", [])
    ]

    print("⚠️ Potential Conflicts:")
    print(f"B2B-specific docs: {len(b2b_docs)}")
    print(f"B2C-specific docs: {len(b2c_docs)}")
    print(f"Reverse charge docs: {len(reverse_charge_docs)}")

    if b2b_docs and b2c_docs:
        print("🚨 WARNING: Mixed B2B and B2C guidance in retrieved docs!")

    if (
        classification["business_type"] == "B2B"
        and classification["is_cross_border"]
        and len(reverse_charge_docs) == 0
    ):
        print(
            "🚨 WARNING: B2B cross-border query but no reverse charge docs retrieved!"
        )


def debug_rag_pipeline(query1: str, query2: str):
    """Main debugging function"""
    print("🔬 RAG Pipeline Debugging Tool")
    print("=" * 80)

    # Initialize components
    classifier = VATQueryClassifier()
    chunk_analyzer = VATChunkAnalyzer()
    documents = load_enhanced_documents()

    # Debug both queries
    for query in [query1, query2]:
        print("\n" + "🧪" * 40)
        print(f"DEBUGGING QUERY: '{query}'")
        print("🧪" * 40)

        # Step 1: Classification
        classification = analyze_classification(query, classifier)

        # Step 2: Metadata filtering
        filtered_chunks = analyze_metadata_filtering(
            documents, classification, chunk_analyzer
        )

        # Step 3: Embedding retrieval
        retrieved_docs = analyze_embedding_retrieval(filtered_chunks, query)

        # Step 4: Prompt construction
        analyze_prompt_construction(retrieved_docs, query, classification)

        print("\n" + "✅" * 40)
        print(f"COMPLETED DEBUGGING FOR: '{query}'")
        print("✅" * 40)


@component
class PromptToMessages:
    def run(self, prompt: str):
        return {"messages": [ChatMessage.from_user(prompt)]}


if __name__ == "__main__":
    # Test queries
    queries = [
        "Google Cloud from Ireland to UK",
        "Software subscription from Ireland to UK",
    ]

    for query in queries:
        print("\n" + "=" * 80)
        debug_pipeline(query)
        print("=" * 80)
