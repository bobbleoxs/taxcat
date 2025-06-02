# demo_enhanced_system.py
import json
import sys
from typing import Dict, List

# Add scripts directory to path
sys.path.append("scripts")

from chunk_analyzer import VATChunkAnalyzer
from query_classifier import VATQueryClassifier


def create_sample_vat_chunks():
    """Create sample VAT guidance chunks for demonstration"""
    return [
        {
            "id": "chunk_1",
            "content": """
            For digital services supplied to business customers in other EU member states,
            the place of supply is where the customer belongs. The customer will account
            for VAT through the reverse charge procedure. This means the UK supplier does
            not charge VAT on the invoice.
            """,
            "source": "VAT Notice 741A - Place of supply of services",
        },
        {
            "id": "chunk_2",
            "content": """
            When you sell goods to consumers in the UK, you must charge the standard rate
            of VAT at 20%. This applies to most retail sales within the United Kingdom.
            The consumer pays the VAT as part of the purchase price.
            """,
            "source": "VAT Notice 700 - The VAT Guide",
        },
        {
            "id": "chunk_3",
            "content": """
            Professional services such as consultancy and legal advice supplied to
            taxable persons are generally subject to the reverse charge when the
            customer is in another member state. The place of supply is where the
            customer belongs.
            """,
            "source": "VAT Notice 741A - Place of supply of services",
        },
        {
            "id": "chunk_4",
            "content": """
            Zero-rated supplies include most food, books, and children's clothing.
            These are taxable supplies but charged at 0% VAT rate. This applies to
            consumers and businesses alike within the UK.
            """,
            "source": "VAT Notice 700 - The VAT Guide",
        },
        {
            "id": "chunk_5",
            "content": """
            Software and digital downloads supplied to UK consumers are standard rated
            at 20%. This includes mobile apps, e-books, and streaming services. The
            place of supply for digital services to consumers is where the supplier
            belongs.
            """,
            "source": "VAT Notice 741A - Place of supply of services",
        },
        {
            "id": "chunk_6",
            "content": """
            Export of goods to countries outside the UK can be zero-rated provided you
            obtain and keep valid evidence of export. This applies to shipments to both
            EU and non-EU countries for goods leaving the UK.
            """,
            "source": "VAT Notice 703 - Exports and Removals",
        },
    ]


def enhance_chunks_with_metadata(chunks: List[Dict]) -> List[Dict]:
    """Add metadata to chunks using the analyzer"""
    analyzer = VATChunkAnalyzer()
    enhanced_chunks = []

    for chunk in chunks:
        # Analyze the chunk content
        metadata = analyzer.analyze_chunk(chunk["content"])

        # Add metadata to chunk
        enhanced_chunk = chunk.copy()
        enhanced_chunk["metadata"] = {
            "transaction_types": metadata.transaction_types,
            "geographic_scope": metadata.geographic_scope,
            "service_categories": metadata.service_categories,
            "vat_treatments": metadata.vat_treatments,
            "confidence_score": metadata.confidence_score,
        }

        enhanced_chunks.append(enhanced_chunk)

    return enhanced_chunks


def demo_filtering_system():
    """Demonstrate the enhanced filtering system"""

    print("🎯 Enhanced VAT Document Chunking & Filtering Demo")
    print("=" * 70)

    # Step 1: Create and enhance sample chunks
    print("\n📝 Step 1: Creating sample VAT guidance chunks...")
    sample_chunks = create_sample_vat_chunks()
    enhanced_chunks = enhance_chunks_with_metadata(sample_chunks)

    print(f"Created {len(enhanced_chunks)} enhanced chunks with metadata")

    # Step 2: Show sample chunk with metadata
    print("\n📄 Step 2: Sample Enhanced Chunk")
    print("-" * 50)
    sample_chunk = enhanced_chunks[0]
    print(f"Content: {sample_chunk['content'].strip()[:100]}...")
    print(f"Source: {sample_chunk['source']}")
    print("Metadata:")
    for key, value in sample_chunk["metadata"].items():
        print(f"  {key}: {value}")

    # Step 3: Test different query classifications
    print("\n🔍 Step 3: Testing Query Classification & Filtering")
    print("=" * 70)

    classifier = VATQueryClassifier()
    analyzer = VATChunkAnalyzer()

    test_queries = [
        {
            "query": "Google Cloud from Ireland to UK",
            "description": "B2B Digital Service EU→UK",
        },
        {
            "query": "Personal Spotify subscription",
            "description": "B2C Digital Service",
        },
        {
            "query": "UK retail clothing sales to consumers",
            "description": "B2C Physical Goods UK Domestic",
        },
    ]

    for test in test_queries:
        print(f"\n🔎 Testing Query: '{test['query']}'")
        print(f"Expected: {test['description']}")
        print("-" * 50)

        # Classify the query
        classification = classifier.classify_query(test["query"])
        classification_dict = {
            "business_type": classification.business_type,
            "service_type": classification.service_type,
            "supplier_location": classification.supplier_location,
            "customer_location": classification.customer_location,
            "is_cross_border": classification.is_cross_border,
            "confidence_score": classification.confidence_score,
            "ambiguities": classification.ambiguities,
        }

        print(
            f"Classification: {classification.business_type} {classification.service_type}"
        )
        print(f"Cross-border: {classification.is_cross_border}")
        print(f"Confidence: {classification.confidence_score:.2f}")

        # Filter chunks before and after
        print("\n📊 Chunk Filtering Results:")
        print(f"Total chunks available: {len(enhanced_chunks)}")

        # Show filtering in action
        filtered_chunks = analyzer.filter_chunks_by_classification(
            enhanced_chunks, classification_dict
        )
        print(f"Relevant chunks after filtering: {len(filtered_chunks)}")

        # Show top 2 most relevant chunks
        top_chunks = filtered_chunks[:2]
        for i, chunk in enumerate(top_chunks, 1):
            relevance_score = chunk.get("relevance_score", 0)
            print(f"\n  📋 Chunk {i} (Relevance Score: {relevance_score}):")
            print(f"     Content: {chunk['content'].strip()[:80]}...")
            print(
                f"     Metadata: {chunk['metadata']['transaction_types']} | {chunk['metadata']['service_categories']} | {chunk['metadata']['vat_treatments']}"
            )

        print("\n" + "=" * 50)

    # Step 4: Show metadata statistics
    print("\n📊 Step 4: Metadata Statistics Across All Chunks")
    print("-" * 50)

    stats = {
        "total_chunks": len(enhanced_chunks),
        "b2b_chunks": 0,
        "b2c_chunks": 0,
        "digital_chunks": 0,
        "professional_chunks": 0,
        "physical_chunks": 0,
        "reverse_charge_chunks": 0,
        "uk_domestic_chunks": 0,
        "eu_chunks": 0,
    }

    for chunk in enhanced_chunks:
        metadata = chunk["metadata"]
        if "B2B" in metadata.get("transaction_types", []):
            stats["b2b_chunks"] += 1
        if "B2C" in metadata.get("transaction_types", []):
            stats["b2c_chunks"] += 1
        if "digital" in metadata.get("service_categories", []):
            stats["digital_chunks"] += 1
        if "professional" in metadata.get("service_categories", []):
            stats["professional_chunks"] += 1
        if "physical" in metadata.get("service_categories", []):
            stats["physical_chunks"] += 1
        if "reverse_charge" in metadata.get("vat_treatments", []):
            stats["reverse_charge_chunks"] += 1
        if "UK_domestic" in metadata.get("geographic_scope", []):
            stats["uk_domestic_chunks"] += 1
        if "EU" in metadata.get("geographic_scope", []):
            stats["eu_chunks"] += 1

    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    # Step 5: Save enhanced chunks (demo)
    print("\n💾 Step 5: Saving Enhanced Chunks (Demo)")
    print("-" * 50)

    output_data = {
        "total_chunks": len(enhanced_chunks),
        "enhanced_chunks": enhanced_chunks,
        "metadata_stats": stats,
        "demo_timestamp": "2025-01-31",
    }

    with open("demo_enhanced_chunks.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("✅ Saved enhanced chunks demo to 'demo_enhanced_chunks.json'")
    print(f"📁 File contains {len(enhanced_chunks)} chunks with full metadata")

    # Step 6: Show integration example
    print("\n🔗 Step 6: Integration with FastAPI")
    print("-" * 50)
    print("The enhanced system integrates with your FastAPI app as follows:")
    print()
    print("1. Query comes in: 'Google Cloud from Ireland to UK'")
    print("2. Query Classifier detects: B2B digital EU→UK cross-border")
    print("3. Chunk Filter finds: Only B2B digital reverse charge chunks")
    print("4. RAG Retrieval gets: Targeted, relevant VAT guidance")
    print("5. LLM Response: Accurate advice about reverse charge procedure")
    print()
    print("Benefits:")
    print("✅ No contradictory B2B vs B2C guidance mixed together")
    print("✅ No irrelevant domestic vs international rules")
    print("✅ Focused on specific VAT treatments (reverse charge)")
    print("✅ Higher accuracy and confidence in responses")


if __name__ == "__main__":
    demo_filtering_system()
