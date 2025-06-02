# VAT Query Classification System

A comprehensive system for classifying VAT (Value Added Tax) queries and providing structured data for targeted document retrieval and advice generation.

## Overview

This system analyzes user VAT queries to automatically detect:

- **Business Type**: B2B vs B2C transactions
- **Service Type**: Digital, professional, or physical goods/services
- **Locations**: Supplier and customer locations
- **Cross-border**: Whether it's an international transaction
- **Ambiguities**: Areas that need clarification

The classification data is then used to filter and enhance RAG (Retrieval-Augmented Generation) responses, ensuring users get the most relevant VAT guidance.

## Features

### 🔍 Query Classification
- Detects B2B indicators: company suffixes (Ltd, Corp), business language, tax identifiers
- Identifies B2C patterns: personal terms, consumer language
- Extracts service types: digital services, professional consulting, physical goods
- Parses locations from directional language ("from UK to Germany")
- Flags cross-border transactions automatically

### 🎯 Smart RAG Integration
- Enhances queries with relevant terms based on classification
- Provides context-aware prompting for targeted advice
- Filters document retrieval based on transaction characteristics
- Suggests clarifying questions for ambiguous queries

### 🚀 API Ready
- FastAPI endpoints for real-time classification
- Batch processing capabilities
- Health checks and monitoring
- Structured JSON responses

## Installation

```bash
# Install dependencies
pip install fastapi uvicorn haystack-ai openai pydantic

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### 1. Basic Classification

```python
from scripts.query_classifier import VATQueryClassifier

classifier = VATQueryClassifier()
result = classifier.classify_query(
    "My UK company provides digital marketing services to a German business. What's the VAT treatment?"
)

print(f"Business Type: {result.business_type}")  # B2B
print(f"Service Type: {result.service_type}")    # mixed (digital + professional)
print(f"Cross-border: {result.is_cross_border}") # True
```

### 2. RAG Integration

```python
from scripts.rag_integration import VATRAGSystem

# Initialize (requires VAT documents to be ingested first)
vat_system = VATRAGSystem()

# Get comprehensive advice
advice = vat_system.answer_vat_query(
    "We're importing software licenses from a US company. How does VAT apply?"
)

print(advice["response"])
```

### 3. API Server

```bash
# Start the API server
python scripts/vat_api.py

# Test classification endpoint
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"query": "My UK company sells software to German businesses"}'
```

## Classification Output

The system returns structured JSON with detailed classification:

```json
{
  "business_type": "B2B",
  "service_type": "digital",
  "supplier_location": "uk",
  "customer_location": "eu",
  "is_cross_border": true,
  "ambiguities": [],
  "confidence_score": 0.95,
  "key_indicators": {
    "b2b_indicators": ["company_suffixes: company"],
    "service_indicators": ["software", "digital"],
    "location_indicators": {"uk": ["uk"], "eu": ["german"]}
  }
}
```

## Business Type Detection

### B2B Indicators
- **Company Suffixes**: Ltd, Limited, Corp, LLC, PLC, Inc, GmbH
- **Business Terms**: company, business, enterprise, firm, corporate
- **B2B Language**: subscription, license, vendor, supplier, invoice
- **Tax Identifiers**: VAT number, tax ID, business registration

### B2C Indicators
- **Personal Terms**: personal, individual, consumer, private
- **Consumer Language**: buy, purchase, shop, personal use, home

## Service Type Classification

### Digital Services
- Software, apps, cloud services, SaaS platforms
- Streaming, downloads, e-books, digital content
- Web services, APIs, hosting, domains, SSL certificates

### Professional Services
- Consulting, advisory, legal, accounting, audit
- Training, education, marketing, design, research

### Physical Goods
- Products, merchandise, equipment, machinery
- Shipping, delivery, manufacturing, inventory

## Location Detection

The system recognizes major regions and can parse directional language:

- **UK**: United Kingdom, England, Scotland, Wales, Britain
- **EU**: European Union, Germany, France, Italy, Spain, Netherlands
- **US**: United States, America, USA
- **International**: Global, worldwide, overseas, foreign

### Directional Parsing
- "from UK to Germany" → supplier: uk, customer: eu
- "German company selling to UK business" → supplier: eu, customer: uk

## Cross-Border Detection

Automatically flags transactions as cross-border based on:
- Explicit terms: "cross-border", "international", "export/import"
- Different supplier/customer locations
- EU-UK post-Brexit indicators

## Ambiguity Handling

The system identifies common ambiguities and suggests clarifications:

- **Business Type Unclear**: Need B2B vs B2C clarification
- **Service Type Mixed**: Multiple service types detected
- **Missing Locations**: Supplier or customer location not specified
- **VAT Registration**: Registration status may affect treatment
- **Thresholds**: Volume thresholds may apply

## API Endpoints

### POST `/classify`
Classify a single VAT query:

```bash
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"query": "UK Ltd selling software to French business"}'
```

### POST `/advice`
Get comprehensive VAT advice with RAG:

```bash
curl -X POST "http://localhost:8000/advice" \
     -H "Content-Type: application/json" \
     -d '{"query": "Cross-border digital services from UK to EU"}'
```

### POST `/classify/batch`
Process multiple queries:

```bash
curl -X POST "http://localhost:8000/classify/batch" \
     -H "Content-Type: application/json" \
     -d '{"queries": [{"query": "..."}, {"query": "..."}]}'
```

### GET `/examples`
Get example queries for testing:

```bash
curl "http://localhost:8000/examples"
```

## Integration with RAG

The classification system enhances RAG retrieval by:

1. **Query Enhancement**: Adding relevant terms based on classification
   - B2B → "business to business", "reverse charge"
   - Digital → "electronically supplied services", "place of supply"
   - Cross-border → "international", "place of supply rules"

2. **Context-Aware Prompting**: Providing classification context to the LLM
3. **Targeted Retrieval**: Filtering documents based on transaction type
4. **Ambiguity Handling**: Asking clarifying questions when needed

## VAT Knowledge Areas

The system is designed to handle key VAT scenarios:

### Post-Brexit Rules
- UK-EU digital services (reverse charge procedures)
- Place of supply changes
- New registration requirements

### Digital Services
- B2B digital services (customer location rules)
- B2C digital services (supplier location rules)
- Electronically supplied services definitions

### Cross-Border Transactions
- Import/export VAT treatment
- Intra-EU vs third country rules
- Registration thresholds by country

## Example Classifications

### B2B Digital Service (UK→EU)
```
Query: "My UK software company sells SaaS to German businesses"
→ B2B, digital, cross-border, reverse charge applicable
```

### B2C Physical Import
```
Query: "Individual buying electronics from China for personal use"
→ B2C, physical, cross-border, import VAT applicable
```

### Professional Service (EU→UK)
```
Query: "French consultant providing services to UK company"
→ B2B, professional, cross-border, reverse charge applicable
```

## Error Handling

The system gracefully handles:
- Unclear business types (defaults to "unclear")
- Multiple service types (flags as "mixed")
- Missing location information (notes in ambiguities)
- Low confidence scores (suggests clarification)

## Performance

- Classification: ~100ms per query
- Batch processing: ~50ms per query in batches
- RAG integration: ~2-3 seconds (depends on LLM)
- Memory usage: ~50MB base + documents

## Future Enhancements

- Machine learning model training on classified examples
- Support for more countries and tax jurisdictions
- Integration with real-time tax rate APIs
- Automated document updates from government sources
- Multi-language support for international queries

## Contributing

To add new patterns or improve classification:

1. Update pattern dictionaries in `VATQueryClassifier.__init__()`
2. Add test cases to verify new patterns work
3. Update confidence scoring if needed
4. Test with real-world examples

## License

This VAT classification system is designed for educational and development purposes. Always consult qualified tax professionals for official VAT advice.
