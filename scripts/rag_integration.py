# scripts/rag_integration.py
import json
import os
from typing import Any, Dict, List

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from query_classifier import VATQueryClassification, VATQueryClassifier


class VATRAGSystem:
    """
    VAT RAG system that uses query classification to provide targeted,
    context-aware VAT advice based on transaction characteristics.
    """

    def __init__(self, documents_file: str = "vat_documents.json"):
        self.classifier = VATQueryClassifier()
        self.document_store = InMemoryDocumentStore()
        self._load_documents(documents_file)
        self._setup_rag_pipeline()

    def _load_documents(self, documents_file: str):
        """Load pre-processed VAT documents into the document store"""
        if not os.path.exists(documents_file):
            print(f"⚠️  Warning: {documents_file} not found. Run ingest.py first.")
            return

        with open(documents_file, "r", encoding="utf-8") as f:
            documents_data = json.load(f)

        from haystack import Document

        documents = []
        for doc_data in documents_data:
            doc = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                meta=doc_data.get("meta", {}),
                embedding=doc_data.get("embedding"),
            )
            documents.append(doc)

        self.document_store.write_documents(documents)
        print(f"📚 Loaded {len(documents)} VAT documents")

    def _setup_rag_pipeline(self):
        """Setup the RAG pipeline with query classification integration"""

        # Components
        text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
        retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store, top_k=5
        )

        # Create context-aware prompt template
        prompt_template = """
        You are a VAT (Value Added Tax) expert assistant. Based on the classified query information and relevant VAT documents, provide accurate, specific guidance.

        **Query Classification:**
        - Business Type: {{ classification.business_type }}
        - Service Type: {{ classification.service_type }}
        - Supplier Location: {{ classification.supplier_location }}
        - Customer Location: {{ classification.customer_location }}
        - Cross-border Transaction: {{ classification.is_cross_border }}
        - Confidence Score: {{ classification.confidence_score }}
        {% if classification.ambiguities %}
        - Ambiguities: {{ classification.ambiguities | join(", ") }}
        {% endif %}

        **User Query:** {{ query }}

        **Relevant VAT Documentation:**
        {% for document in documents %}
        {{ document.content }}
        ---
        {% endfor %}

        **Instructions:**
        1. Provide specific VAT treatment based on the transaction characteristics
        2. Reference relevant VAT notices and regulations
        3. If this is a cross-border transaction, clearly state the place of supply rules
        4. For B2B digital services from EU to UK, mention reverse charge procedures if applicable
        5. If there are ambiguities, ask clarifying questions
        6. Always be specific about registration requirements and thresholds where relevant

        **Response:**
        """

        prompt_builder = PromptBuilder(template=prompt_template)
        generator = OpenAIGenerator(
            model="gpt-4", generation_kwargs={"max_tokens": 1000}
        )

        # Build pipeline
        self.rag_pipeline = Pipeline()
        self.rag_pipeline.add_component("text_embedder", text_embedder)
        self.rag_pipeline.add_component("retriever", retriever)
        self.rag_pipeline.add_component("prompt_builder", prompt_builder)
        self.rag_pipeline.add_component("llm", generator)

        # Connect components
        self.rag_pipeline.connect(
            "text_embedder.embedding", "retriever.query_embedding"
        )
        self.rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

    def get_filtered_query(
        self, original_query: str, classification: VATQueryClassification
    ) -> str:
        """
        Create an enhanced query for document retrieval based on classification
        """
        enhanced_terms = [original_query]

        # Add business type specific terms
        if classification.business_type == "B2B":
            enhanced_terms.extend(
                ["business to business", "commercial", "reverse charge"]
            )
        elif classification.business_type == "B2C":
            enhanced_terms.extend(
                ["business to consumer", "consumer sales", "end consumer"]
            )

        # Add service type specific terms
        if classification.service_type == "digital":
            enhanced_terms.extend(
                [
                    "digital services",
                    "electronically supplied services",
                    "place of supply",
                ]
            )
        elif classification.service_type == "professional":
            enhanced_terms.extend(
                ["professional services", "consulting", "place of customer"]
            )
        elif classification.service_type == "physical":
            enhanced_terms.extend(["goods", "imports", "customs", "physical delivery"])

        # Add location specific terms
        if classification.is_cross_border:
            enhanced_terms.extend(["cross-border", "international", "place of supply"])

        if (
            classification.supplier_location == "uk"
            and classification.customer_location == "eu"
        ):
            enhanced_terms.extend(["UK to EU", "post-Brexit", "export"])
        elif (
            classification.supplier_location == "eu"
            and classification.customer_location == "uk"
        ):
            enhanced_terms.extend(["EU to UK", "import", "reverse charge"])

        return " ".join(enhanced_terms)

    def answer_vat_query(self, query: str) -> Dict[str, Any]:
        """
        Process a VAT query and return comprehensive advice with classification context
        """
        # Step 1: Classify the query
        classification = self.classifier.classify_query(query)

        # Step 2: Create enhanced query for retrieval
        enhanced_query = self.get_filtered_query(query, classification)

        # Step 3: Run RAG pipeline
        try:
            result = self.rag_pipeline.run(
                {
                    "text_embedder": {"text": enhanced_query},
                    "prompt_builder": {
                        "query": query,
                        "classification": {
                            "business_type": classification.business_type,
                            "service_type": classification.service_type,
                            "supplier_location": classification.supplier_location
                            or "Not specified",
                            "customer_location": classification.customer_location
                            or "Not specified",
                            "is_cross_border": classification.is_cross_border,
                            "confidence_score": f"{classification.confidence_score:.2f}",
                            "ambiguities": classification.ambiguities,
                        },
                    },
                }
            )

            response = result["llm"]["replies"][0]

        except Exception as e:
            response = f"Error generating response: {str(e)}"

        return {
            "query": query,
            "classification": self.classifier.to_json(classification),
            "enhanced_query": enhanced_query,
            "response": response,
            "confidence": classification.confidence_score,
        }

    def suggest_clarifying_questions(
        self, classification: VATQueryClassification
    ) -> List[str]:
        """Generate clarifying questions based on identified ambiguities"""
        questions = []

        if classification.business_type == "unclear":
            questions.append(
                "Are you asking about a business-to-business (B2B) or business-to-consumer (B2C) transaction?"
            )

        if classification.service_type == "unclear":
            questions.append(
                "What type of service or goods are being supplied? (e.g., digital services, consulting, physical products)"
            )

        if not classification.supplier_location:
            questions.append("Where is the supplier located/registered for VAT?")

        if not classification.customer_location:
            questions.append("Where is the customer located?")

        if "VAT registration status may affect treatment" in classification.ambiguities:
            questions.append(
                "What is the VAT registration status of both the supplier and customer?"
            )

        if "Registration thresholds may apply" in classification.ambiguities:
            questions.append(
                "What is the value/volume of the transactions in question?"
            )

        return questions


# Example usage
if __name__ == "__main__":
    # Initialize the VAT RAG system
    print("🚀 Initializing VAT RAG System...")
    vat_system = VATRAGSystem()

    # Test queries
    test_queries = [
        "My UK software company sells SaaS subscriptions to German businesses. Do I need to charge VAT?",
        "I'm buying professional consulting services from a French company. What's the VAT treatment?",
        "We import physical products from China to our UK warehouse. How does VAT apply?",
    ]

    print("\n🔍 VAT Query Processing Examples:")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 50)

        result = vat_system.answer_vat_query(query)

        # Parse classification for display
        classification_data = json.loads(result["classification"])

        print("🏷️  Classification:")
        print(f"   Business Type: {classification_data['business_type']}")
        print(f"   Service Type: {classification_data['service_type']}")
        print(f"   Cross-border: {classification_data['is_cross_border']}")
        print(f"   Confidence: {classification_data['confidence_score']:.2f}")

        if classification_data["ambiguities"]:
            print(f"⚠️  Ambiguities: {', '.join(classification_data['ambiguities'])}")

        print("\n🤖 Response:")
        print(
            result["response"][:500] + "..."
            if len(result["response"]) > 500
            else result["response"]
        )

        print("\n" + "=" * 60)
