# scripts/query_classifier.py
import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class VATQueryClassification:
    """Structured classification of a VAT query"""

    business_type: str  # "B2B", "B2C", "unclear"
    service_type: str  # "digital", "professional", "physical", "mixed", "unclear"
    supplier_location: Optional[str]
    customer_location: Optional[str]
    is_cross_border: bool
    ambiguities: List[str]
    confidence_score: float
    key_indicators: Dict[str, List[str]]


class VATQueryClassifier:
    """Classifies VAT queries to provide structured data for RAG filtering"""

    def __init__(self):
        # B2B indicators
        self.b2b_patterns = {
            "company_suffixes": [
                "ltd",
                "limited",
                "corp",
                "corporation",
                "llc",
                "plc",
                "inc",
                "incorporated",
                "gmbh",
                "sa",
                "srl",
                "bv",
            ],
            "business_terms": [
                "company",
                "business",
                "enterprise",
                "firm",
                "organization",
                "organisation",
                "corporate",
                "commercial",
            ],
            "b2b_language": [
                "subscription",
                "license",
                "licensing",
                "procurement",
                "vendor",
                "supplier",
                "client",
                "invoice",
                "purchase order",
                "contract",
            ],
            "tax_identifiers": [
                "vat number",
                "tax id",
                "ein",
                "business registration",
                "company registration",
            ],
            "b2b_brands": [
                "google cloud",
                "microsoft azure",
                "amazon aws",
                "salesforce",
                "slack",
                "zoom",
                "adobe creative suite",
                "office 365",
                "microsoft 365",
            ],
        }

        # B2C indicators
        self.b2c_patterns = {
            "personal_terms": [
                "personal",
                "individual",
                "consumer",
                "customer",
                "private",
                "household",
                "my",
                "i am",
                "i'm",
            ],
            "consumer_language": [
                "buy",
                "purchase",
                "shop",
                "order",
                "personal use",
                "home",
                "family",
            ],
            "consumer_brands": [
                "spotify",
                "netflix",
                "apple music",
                "amazon prime",
                "disney plus",
                "personal subscription",
            ],
        }

        # Service type indicators
        self.service_patterns = {
            "digital": [
                "software",
                "app",
                "digital",
                "online",
                "cloud",
                "saas",
                "platform",
                "streaming",
                "download",
                "e-book",
                "digital content",
                "web service",
                "api",
                "hosting",
                "email",
                "telecommunication",
                "internet",
                "website",
                "domain",
                "ssl certificate",
                # Specific digital services/brands
                "azure",
                "aws",
                "google cloud",
                "spotify",
                "netflix",
                "adobe",
                "creative suite",
                "office 365",
                "microsoft 365",
                "subscription service",
                "digital service",
            ],
            "professional": [
                "consulting",
                "advisory",
                "legal",
                "accounting",
                "audit",
                "tax advice",
                "professional service",
                "expertise",
                "consultation",
                "training",
                "education",
                "marketing",
                "design",
                "development",
                "research",
                "analysis",
            ],
            "physical": [
                "goods",
                "product",
                "merchandise",
                "item",
                "shipping",
                "delivery",
                "warehouse",
                "inventory",
                "manufacturing",
                "equipment",
                "material",
                "tangible",
                "physical",
                "hardware",
                "machinery",
            ],
        }

        # Location indicators - improved with specific EU countries
        self.location_patterns = {
            "uk": [
                "uk",
                "united kingdom",
                "england",
                "scotland",
                "wales",
                "northern ireland",
                "britain",
                "gb",
            ],
            "eu": [
                "eu",
                "european union",
                "europe",
                "germany",
                "france",
                "italy",
                "spain",
                "netherlands",
                "belgium",
                "ireland",
                "austria",
                "denmark",
                "sweden",
                "poland",
            ],
            "us": ["us", "usa", "united states", "america", "american"],
            "international": [
                "international",
                "global",
                "worldwide",
                "overseas",
                "foreign",
                "abroad",
            ],
        }

        # Cross-border indicators
        self.cross_border_terms = [
            "cross-border",
            "international",
            "export",
            "import",
            "foreign",
            "overseas",
            "eu to uk",
            "uk to eu",
            "non-eu",
            "third country",
            "intra-eu",
        ]

    def classify_query(self, query: str) -> VATQueryClassification:
        """
        Classify a VAT query and return structured classification data

        Args:
            query: The user's VAT question/query

        Returns:
            VATQueryClassification object with structured data
        """
        query_lower = query.lower()

        # Detect business type (B2B vs B2C)
        business_type, b2b_indicators, b2c_indicators = self._classify_business_type(
            query_lower
        )

        # Extract service type
        service_type, service_indicators = self._classify_service_type(query_lower)

        # Identify locations
        supplier_location, customer_location, location_indicators = (
            self._extract_locations(query_lower)
        )

        # Check for cross-border transaction
        is_cross_border = self._detect_cross_border(
            query_lower, supplier_location, customer_location
        )

        # Identify ambiguities
        ambiguities = self._identify_ambiguities(
            query_lower,
            business_type,
            service_type,
            supplier_location,
            customer_location,
        )

        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            business_type,
            service_type,
            supplier_location,
            customer_location,
            ambiguities,
        )

        # Compile key indicators
        key_indicators = {
            "b2b_indicators": b2b_indicators,
            "b2c_indicators": b2c_indicators,
            "service_indicators": service_indicators,
            "location_indicators": location_indicators,
        }

        return VATQueryClassification(
            business_type=business_type,
            service_type=service_type,
            supplier_location=supplier_location,
            customer_location=customer_location,
            is_cross_border=is_cross_border,
            ambiguities=ambiguities,
            confidence_score=confidence_score,
            key_indicators=key_indicators,
        )

    def _classify_business_type(self, query: str) -> Tuple[str, List[str], List[str]]:
        """Classify whether transaction is B2B or B2C"""
        b2b_indicators = []
        b2c_indicators = []

        # Check for B2B patterns
        for category, patterns in self.b2b_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    b2b_indicators.append(f"{category}: {pattern}")

        # Check for B2C patterns
        for category, patterns in self.b2c_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    b2c_indicators.append(f"{category}: {pattern}")

        # Improved business type determination with context weighting
        b2b_score = len(b2b_indicators)
        b2c_score = len(b2c_indicators)

        # Apply context-based weighting
        # Strong B2C indicators
        if any("personal" in indicator for indicator in b2c_indicators):
            b2c_score += 2  # Personal is a strong B2C signal
        if any("individual" in indicator for indicator in b2c_indicators):
            b2c_score += 2
        if any("consumer_brands" in indicator for indicator in b2c_indicators):
            b2c_score += 1  # Consumer brands like Spotify

        # Strong B2B indicators
        if "business" in query.lower():
            b2b_score += 2  # "business" is a strong B2B signal
        if "company" in query.lower():
            b2b_score += 2
        if any("b2b_brands" in indicator for indicator in b2b_indicators):
            b2b_score += 2  # Enterprise brands like Google Cloud

        # Special case: if "personal" appears with consumer services, it's likely B2C
        if "personal" in query.lower() and any(
            service in query.lower()
            for service in ["spotify", "netflix", "personal subscription"]
        ):
            b2c_score += 3

        # Determine business type based on weighted scores
        if b2b_score > b2c_score:
            return "B2B", b2b_indicators, b2c_indicators
        elif b2c_score > b2b_score:
            return "B2C", b2b_indicators, b2c_indicators
        else:
            return "unclear", b2b_indicators, b2c_indicators

    def _classify_service_type(self, query: str) -> Tuple[str, List[str]]:
        """Classify the type of service/goods"""
        service_indicators = {}

        for service_type, patterns in self.service_patterns.items():
            matches = [pattern for pattern in patterns if pattern in query]
            if matches:
                service_indicators[service_type] = matches

        # Determine primary service type
        if len(service_indicators) == 0:
            return "unclear", []
        elif len(service_indicators) == 1:
            service_type = list(service_indicators.keys())[0]
            return service_type, service_indicators[service_type]
        else:
            # Multiple types detected
            return "mixed", [f"{k}: {v}" for k, v in service_indicators.items()]

    def _extract_locations(
        self, query: str
    ) -> Tuple[Optional[str], Optional[str], Dict[str, List[str]]]:
        """Extract supplier and customer locations"""
        location_indicators = {}

        for region, patterns in self.location_patterns.items():
            matches = [pattern for pattern in patterns if pattern in query]
            if matches:
                location_indicators[region] = matches

        # Try to identify supplier vs customer based on context
        supplier_location = None
        customer_location = None

        # Look for directional indicators - improved pattern
        if "from" in query and "to" in query:
            # Try to parse "from X to Y" pattern with better regex
            from_match = re.search(r"from\s+([^to]+?)\s+to\s+([^,.!?]+)", query)
            if from_match:
                supplier_loc = from_match.group(1).strip()
                customer_loc = from_match.group(2).strip()

                supplier_location = self._match_location_to_region(supplier_loc)
                customer_location = self._match_location_to_region(customer_loc)

        # If we can't determine direction, just note locations found
        if not supplier_location and not customer_location and location_indicators:
            regions = list(location_indicators.keys())
            if len(regions) == 1:
                supplier_location = regions[0]
            elif len(regions) >= 2:
                supplier_location = regions[0]
                customer_location = regions[1]

        return supplier_location, customer_location, location_indicators

    def _match_location_to_region(self, location: str) -> Optional[str]:
        """Match a location string to a known region"""
        location_lower = location.lower()
        for region, patterns in self.location_patterns.items():
            if any(pattern in location_lower for pattern in patterns):
                return region
        return location_lower if location_lower else None

    def _detect_cross_border(
        self,
        query: str,
        supplier_location: Optional[str],
        customer_location: Optional[str],
    ) -> bool:
        """Detect if this is a cross-border transaction"""
        # Check for explicit cross-border terms
        if any(term in query for term in self.cross_border_terms):
            return True

        # Check if supplier and customer are in different regions
        if (
            supplier_location
            and customer_location
            and supplier_location != customer_location
        ):
            return True

        return False

    def _identify_ambiguities(
        self,
        query: str,
        business_type: str,
        service_type: str,
        supplier_location: Optional[str],
        customer_location: Optional[str],
    ) -> List[str]:
        """Identify areas that need clarification"""
        ambiguities = []

        if business_type == "unclear":
            ambiguities.append(
                "Business type (B2B vs B2C) unclear - need clarification on customer type"
            )

        if service_type == "unclear":
            ambiguities.append(
                "Service type unclear - need clarification on what is being supplied"
            )

        if service_type == "mixed":
            ambiguities.append(
                "Multiple service types detected - need clarification on primary service"
            )

        if not supplier_location:
            ambiguities.append("Supplier location not specified")

        if not customer_location:
            ambiguities.append("Customer location not specified")

        # Check for potential VAT registration ambiguity
        if "vat" in query and "registration" in query:
            ambiguities.append("VAT registration status may affect treatment")

        # Check for threshold-related queries
        if any(term in query for term in ["threshold", "limit", "registration"]):
            ambiguities.append(
                "Registration thresholds may apply based on transaction volume"
            )

        return ambiguities

    def _calculate_confidence(
        self,
        business_type: str,
        service_type: str,
        supplier_location: Optional[str],
        customer_location: Optional[str],
        ambiguities: List[str],
    ) -> float:
        """Calculate confidence score for the classification"""
        score = 1.0

        # Reduce confidence for unclear classifications
        if business_type == "unclear":
            score -= 0.3
        if service_type == "unclear":
            score -= 0.2
        if service_type == "mixed":
            score -= 0.1
        if not supplier_location:
            score -= 0.15
        if not customer_location:
            score -= 0.15

        # Reduce confidence based on number of ambiguities
        score -= len(ambiguities) * 0.05

        return max(0.0, min(1.0, score))

    def to_json(self, classification: VATQueryClassification) -> str:
        """Convert classification to JSON string"""
        return json.dumps(asdict(classification), indent=2, ensure_ascii=False)


# Example usage and testing
if __name__ == "__main__":
    classifier = VATQueryClassifier()

    # Test queries
    test_queries = [
        "My UK company provides digital marketing services to a German business. What's the VAT treatment?",
        "I'm an individual buying software from a US company for personal use.",
        "We're a Ltd company importing physical goods from France to the UK.",
        "Professional consulting services from UK to EU client company",
        "Personal purchase of digital content from online platform",
    ]

    print("🔍 VAT Query Classification Test Results:")
    print("=" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)

        classification = classifier.classify_query(query)
        print(f"Business Type: {classification.business_type}")
        print(f"Service Type: {classification.service_type}")
        print(f"Supplier Location: {classification.supplier_location}")
        print(f"Customer Location: {classification.customer_location}")
        print(f"Cross-border: {classification.is_cross_border}")
        print(f"Confidence: {classification.confidence_score:.2f}")

        if classification.ambiguities:
            print(f"Ambiguities: {', '.join(classification.ambiguities)}")
