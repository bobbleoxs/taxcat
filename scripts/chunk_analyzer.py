# scripts/chunk_analyzer.py
import re
from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk"""

    transaction_types: List[str]  # B2B, B2C, general
    geographic_scope: List[str]  # UK_domestic, EU, international
    service_categories: List[str]  # digital, professional, physical, general
    vat_treatments: List[str]  # 20%, 0%, exempt, out_of_scope, reverse_charge
    confidence_score: float  # Overall confidence in the metadata extraction


class VATChunkAnalyzer:
    """Analyzes VAT guidance chunks to extract relevant metadata for filtering"""

    def __init__(self):
        # Transaction type indicators
        self.transaction_patterns = {
            "B2B": [
                "business customer",
                "taxable person",
                "registered business",
                "vat registered",
                "business to business",
                "b2b",
                "commercial customer",
                "business purchaser",
                "registered for vat",
                "business registration",
                "business supplies",
                "reverse charge",
                "place of customer",
                "customer registration",
            ],
            "B2C": [
                "consumer",
                "end user",
                "individual",
                "personal use",
                "final consumer",
                "non-business",
                "private individual",
                "household",
                "personal customer",
                "consumer sales",
                "retail sales",
                "end consumer",
                "personal purchase",
            ],
        }

        # Geographic scope indicators
        self.geographic_patterns = {
            "UK_domestic": [
                "within the uk",
                "domestic supply",
                "uk supply",
                "within united kingdom",
                "uk customer",
                "uk business",
                "domestic transaction",
                "uk resident",
                "within britain",
                "domestic sales",
                "uk to uk",
            ],
            "EU": [
                "european union",
                "eu member state",
                "intra-eu",
                "eu supply",
                "within the eu",
                "eu customer",
                "eu business",
                "member state",
                "intra-community",
                "eu transaction",
                "european",
                "continental europe",
            ],
            "international": [
                "third country",
                "non-eu",
                "outside the eu",
                "export",
                "import",
                "international supply",
                "overseas",
                "foreign country",
                "worldwide",
                "global",
                "cross-border",
                "outside uk",
                "rest of world",
            ],
        }

        # Service category indicators
        self.service_patterns = {
            "digital": [
                "digital services",
                "electronic services",
                "electronically supplied",
                "software",
                "app",
                "digital content",
                "streaming",
                "download",
                "cloud services",
                "saas",
                "online platform",
                "web service",
                "digital product",
                "electronic supply",
                "telecommunications",
                "broadcasting",
                "e-book",
                "digital media",
                "internet service",
            ],
            "professional": [
                "professional services",
                "consulting",
                "advisory",
                "legal services",
                "accounting",
                "audit",
                "management consultancy",
                "training",
                "education",
                "design",
                "marketing",
                "research",
                "analysis",
                "expertise",
                "professional advice",
                "consultancy service",
            ],
            "physical": [
                "goods",
                "tangible goods",
                "physical products",
                "merchandise",
                "inventory",
                "equipment",
                "machinery",
                "materials",
                "supplies",
                "manufacturing",
                "retail goods",
                "physical delivery",
                "shipping",
                "warehouse",
                "stock",
                "tangible property",
            ],
        }

        # VAT treatment indicators
        self.vat_treatment_patterns = {
            "20%": [
                "standard rate",
                "20%",
                "20 percent",
                "twenty percent",
                "standard rated",
                "standard vat",
                "full rate",
            ],
            "0%": [
                "zero rate",
                "0%",
                "0 percent",
                "zero percent",
                "zero rated",
                "zero-rated",
                "nil rate",
                "zero vat",
            ],
            "exempt": [
                "exempt",
                "exemption",
                "vat exempt",
                "exempt supply",
                "exempt from vat",
                "exempted",
                "vat exemption",
            ],
            "out_of_scope": [
                "out of scope",
                "outside the scope",
                "not within scope",
                "outside scope of vat",
                "out-of-scope",
                "non-taxable",
            ],
            "reverse_charge": [
                "reverse charge",
                "reverse charging",
                "customer accounts",
                "customer to account",
                "reverse vat",
                "reverse charge procedure",
                "customer responsible for vat",
            ],
        }

    def analyze_chunk(self, chunk_content: str) -> ChunkMetadata:
        """
        Analyze a chunk of VAT guidance and extract metadata

        Args:
            chunk_content: The text content of the chunk

        Returns:
            ChunkMetadata object with extracted information
        """
        content_lower = chunk_content.lower()

        # Detect transaction types
        transaction_types = self._detect_transaction_types(content_lower)

        # Detect geographic scope
        geographic_scope = self._detect_geographic_scope(content_lower)

        # Detect service categories
        service_categories = self._detect_service_categories(content_lower)

        # Detect VAT treatments
        vat_treatments = self._detect_vat_treatments(content_lower)

        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            transaction_types, geographic_scope, service_categories, vat_treatments
        )

        # If no specific categories found, add "general"
        if not transaction_types:
            transaction_types = ["general"]
        if not geographic_scope:
            geographic_scope = ["general"]
        if not service_categories:
            service_categories = ["general"]
        if not vat_treatments:
            vat_treatments = ["general"]

        return ChunkMetadata(
            transaction_types=transaction_types,
            geographic_scope=geographic_scope,
            service_categories=service_categories,
            vat_treatments=vat_treatments,
            confidence_score=confidence_score,
        )

    def _detect_transaction_types(self, content: str) -> List[str]:
        """Detect B2B, B2C, or general transaction types"""
        detected = set()

        for transaction_type, patterns in self.transaction_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    detected.add(transaction_type)

        return list(detected)

    def _detect_geographic_scope(self, content: str) -> List[str]:
        """Detect geographic scope (UK_domestic, EU, international)"""
        detected = set()

        for scope, patterns in self.geographic_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    detected.add(scope)

        return list(detected)

    def _detect_service_categories(self, content: str) -> List[str]:
        """Detect service categories (digital, professional, physical)"""
        detected = set()

        for category, patterns in self.service_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    detected.add(category)

        return list(detected)

    def _detect_vat_treatments(self, content: str) -> List[str]:
        """Detect VAT treatments mentioned in the text"""
        detected = set()

        for treatment, patterns in self.vat_treatment_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    detected.add(treatment)

        # Also look for percentage patterns
        percentage_matches = re.findall(r"(\d+)%", content)
        for match in percentage_matches:
            if match in ["20", "5", "0"]:
                detected.add(f"{match}%")

        return list(detected)

    def _calculate_confidence(
        self,
        transaction_types: List[str],
        geographic_scope: List[str],
        service_categories: List[str],
        vat_treatments: List[str],
    ) -> float:
        """Calculate confidence score based on detected metadata"""
        score = 0.0
        total_categories = 4

        # Each category contributes to confidence
        if transaction_types:
            score += 0.25
        if geographic_scope:
            score += 0.25
        if service_categories:
            score += 0.25
        if vat_treatments:
            score += 0.25

        # Bonus for specific (non-general) detection
        specific_count = 0
        if transaction_types and "general" not in transaction_types:
            specific_count += 1
        if geographic_scope and "general" not in geographic_scope:
            specific_count += 1
        if service_categories and "general" not in service_categories:
            specific_count += 1
        if vat_treatments and "general" not in vat_treatments:
            specific_count += 1

        # Boost score for specific detections
        specificity_bonus = (specific_count / total_categories) * 0.2

        return min(1.0, score + specificity_bonus)

    def filter_chunks_by_classification(
        self, chunks: List[Dict], classification: Dict
    ) -> List[Dict]:
        """
        Filter chunks based on query classification results

        Args:
            chunks: List of chunks with metadata
            classification: Query classification results

        Returns:
            Filtered list of chunks most relevant to the classification
        """
        filtered_chunks = []

        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            # Check if chunk is relevant based on classification
            relevance_score = 0

            # Business type matching (most important)
            business_type = classification.get("business_type", "").upper()
            if business_type in ["B2B", "B2C"]:
                chunk_transaction_types = metadata.get("transaction_types", [])
                if business_type in chunk_transaction_types:
                    relevance_score += 3
                elif "general" in chunk_transaction_types:
                    relevance_score += 0.5  # Much lower score for general

            # Service type matching
            service_type = classification.get("service_type", "")
            if service_type in ["digital", "professional", "physical"]:
                chunk_service_categories = metadata.get("service_categories", [])
                if service_type in chunk_service_categories:
                    relevance_score += 2
                elif "general" in chunk_service_categories:
                    relevance_score += 0.3  # Much lower score for general

            # Geographic matching (important for cross-border)
            is_cross_border = classification.get("is_cross_border", False)
            chunk_geographic_scope = metadata.get("geographic_scope", [])
            if is_cross_border:
                if (
                    "international" in chunk_geographic_scope
                    or "EU" in chunk_geographic_scope
                ):
                    relevance_score += 2
                elif "general" in chunk_geographic_scope:
                    relevance_score += 0.3
            else:
                if "UK_domestic" in chunk_geographic_scope:
                    relevance_score += 2
                elif "general" in chunk_geographic_scope:
                    relevance_score += 0.3

            # Special boost for reverse charge content (important for B2B digital cross-border)
            vat_treatments = metadata.get("vat_treatments", [])
            if (
                business_type == "B2B"
                and service_type == "digital"
                and is_cross_border
                and "reverse_charge" in vat_treatments
            ):
                relevance_score += 3  # Major boost for reverse charge

            # Only include chunks with meaningful relevance (threshold of 2.0)
            # This ensures we get chunks that match at least one major criterion
            if relevance_score >= 2.0:
                chunk["relevance_score"] = relevance_score
                filtered_chunks.append(chunk)

        # Sort by relevance score (highest first)
        filtered_chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # If we get too few results, lower the threshold slightly
        if len(filtered_chunks) < 10 and len(chunks) > 50:
            print(
                f"⚠️  Only {len(filtered_chunks)} chunks with high relevance, including more..."
            )
            for chunk in chunks:
                if chunk not in filtered_chunks:
                    metadata = chunk.get("metadata", {})
                    # Calculate a basic relevance score for backup chunks
                    basic_score = 0
                    if business_type in metadata.get("transaction_types", []):
                        basic_score += 1
                    if service_type in metadata.get("service_categories", []):
                        basic_score += 1
                    if basic_score >= 1:  # At least one match
                        chunk["relevance_score"] = basic_score
                        filtered_chunks.append(chunk)
                        if len(filtered_chunks) >= 50:  # Cap at 50 chunks
                            break

            # Re-sort with the additional chunks
            filtered_chunks.sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )

        return filtered_chunks


# Example usage and testing
if __name__ == "__main__":
    analyzer = VATChunkAnalyzer()

    # Test with sample VAT guidance chunks
    test_chunks = [
        """
        For digital services supplied to business customers in other EU member states,
        the place of supply is where the customer belongs. The customer will account
        for VAT through the reverse charge procedure.
        """,
        """
        When you sell goods to consumers in the UK, you must charge the standard rate
        of VAT at 20%. This applies to most retail sales within the United Kingdom.
        """,
        """
        Professional services such as consultancy and legal advice supplied to
        taxable persons are generally subject to the reverse charge when the
        customer is in another member state.
        """,
        """
        Zero-rated supplies include most food, books, and children's clothing.
        These are taxable supplies but charged at 0% VAT rate.
        """,
    ]

    print("🔍 VAT Chunk Analysis Examples:")
    print("=" * 60)

    for i, chunk in enumerate(test_chunks, 1):
        print(f"\n📄 Chunk {i}:")
        print(f"Text: {chunk.strip()}")
        print("-" * 40)

        metadata = analyzer.analyze_chunk(chunk)

        print(f"🏷️  Transaction Types: {metadata.transaction_types}")
        print(f"🌍 Geographic Scope: {metadata.geographic_scope}")
        print(f"🔧 Service Categories: {metadata.service_categories}")
        print(f"💰 VAT Treatments: {metadata.vat_treatments}")
        print(f"📊 Confidence: {metadata.confidence_score:.2f}")

        print("\n📋 Metadata JSON:")
        print(asdict(metadata))

        print("\n" + "=" * 60)
