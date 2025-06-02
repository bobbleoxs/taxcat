# scripts/vat_api.py
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from query_classifier import VATQueryClassifier
from rag_integration import VATRAGSystem

app = FastAPI(
    title="VAT Classification API",
    description="Classify VAT queries and provide structured data for RAG filtering",
    version="1.0.0",
)

# Initialize the systems
classifier = VATQueryClassifier()
try:
    rag_system = VATRAGSystem()
    rag_available = True
except Exception as e:
    print(f"⚠️  RAG system not available: {e}")
    rag_available = False


class VATQuery(BaseModel):
    query: str


class ClassificationResponse(BaseModel):
    business_type: str
    service_type: str
    supplier_location: Optional[str]
    customer_location: Optional[str]
    is_cross_border: bool
    ambiguities: List[str]
    confidence_score: float
    key_indicators: Dict[str, List[str]]


class VATAdviceResponse(BaseModel):
    query: str
    classification: ClassificationResponse
    enhanced_query: str
    response: str
    confidence: float
    clarifying_questions: List[str]


@app.get("/")
async def root():
    return {
        "message": "VAT Classification API",
        "endpoints": {
            "classify": "/classify - Classify a VAT query",
            "advice": "/advice - Get VAT advice with classification",
            "health": "/health - Health check",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_system_available": rag_available,
        "services": {
            "classifier": "available",
            "rag": "available" if rag_available else "unavailable",
        },
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_vat_query(query: VATQuery):
    """
    Classify a VAT query and return structured data about the transaction type.

    This endpoint analyzes the query to determine:
    - Business type (B2B vs B2C)
    - Service type (digital, professional, physical)
    - Supplier and customer locations
    - Cross-border transaction flag
    - Ambiguities that need clarification
    """
    try:
        classification = classifier.classify_query(query.query)

        return ClassificationResponse(
            business_type=classification.business_type,
            service_type=classification.service_type,
            supplier_location=classification.supplier_location,
            customer_location=classification.customer_location,
            is_cross_border=classification.is_cross_border,
            ambiguities=classification.ambiguities,
            confidence_score=classification.confidence_score,
            key_indicators=classification.key_indicators,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/advice", response_model=VATAdviceResponse)
async def get_vat_advice(query: VATQuery):
    """
    Get comprehensive VAT advice using query classification and RAG.

    This endpoint:
    1. Classifies the query to understand transaction characteristics
    2. Uses RAG to retrieve relevant VAT documentation
    3. Provides targeted advice based on the classification
    4. Suggests clarifying questions if ambiguities exist
    """
    if not rag_available:
        raise HTTPException(
            status_code=503,
            detail="RAG system not available. Please run ingest.py first to load VAT documents.",
        )

    try:
        # Get classification first
        classification = classifier.classify_query(query.query)

        # Get comprehensive advice
        result = rag_system.answer_vat_query(query.query)

        # Get clarifying questions
        clarifying_questions = rag_system.suggest_clarifying_questions(classification)

        return VATAdviceResponse(
            query=query.query,
            classification=ClassificationResponse(
                business_type=classification.business_type,
                service_type=classification.service_type,
                supplier_location=classification.supplier_location,
                customer_location=classification.customer_location,
                is_cross_border=classification.is_cross_border,
                ambiguities=classification.ambiguities,
                confidence_score=classification.confidence_score,
                key_indicators=classification.key_indicators,
            ),
            enhanced_query=result["enhanced_query"],
            response=result["response"],
            confidence=result["confidence"],
            clarifying_questions=clarifying_questions,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Advice generation failed: {str(e)}"
        )


@app.post("/classify/batch")
async def classify_batch_queries(queries: List[VATQuery]):
    """
    Classify multiple VAT queries in batch.

    Useful for processing multiple queries efficiently.
    """
    try:
        results = []
        for query in queries:
            classification = classifier.classify_query(query.query)
            results.append(
                {
                    "query": query.query,
                    "classification": ClassificationResponse(
                        business_type=classification.business_type,
                        service_type=classification.service_type,
                        supplier_location=classification.supplier_location,
                        customer_location=classification.customer_location,
                        is_cross_border=classification.is_cross_border,
                        ambiguities=classification.ambiguities,
                        confidence_score=classification.confidence_score,
                        key_indicators=classification.key_indicators,
                    ),
                }
            )
        return {"results": results}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch classification failed: {str(e)}"
        )


@app.get("/examples")
async def get_example_queries():
    """
    Get example VAT queries for testing the classification system.
    """
    return {
        "examples": [
            {
                "category": "B2B Digital Services",
                "query": "My UK software company provides SaaS services to German businesses. What's the VAT treatment?",
                "expected_classification": {
                    "business_type": "B2B",
                    "service_type": "digital",
                    "is_cross_border": True,
                },
            },
            {
                "category": "B2C Physical Goods",
                "query": "I'm an individual buying electronics from a Chinese retailer for personal use.",
                "expected_classification": {
                    "business_type": "B2C",
                    "service_type": "physical",
                    "is_cross_border": True,
                },
            },
            {
                "category": "Professional Services",
                "query": "Our company is hiring a French consultant for business advisory services.",
                "expected_classification": {
                    "business_type": "B2B",
                    "service_type": "professional",
                    "is_cross_border": True,
                },
            },
            {
                "category": "Domestic Transaction",
                "query": "UK Ltd company selling software licenses to another UK business.",
                "expected_classification": {
                    "business_type": "B2B",
                    "service_type": "digital",
                    "is_cross_border": False,
                },
            },
        ]
    }


if __name__ == "__main__":
    print("🚀 Starting VAT Classification API...")
    print("📝 Available endpoints:")
    print("   - POST /classify - Classify a single VAT query")
    print("   - POST /advice - Get comprehensive VAT advice")
    print("   - POST /classify/batch - Classify multiple queries")
    print("   - GET /examples - Get example queries")
    print("   - GET /health - Health check")
    print("\n💡 Example usage:")
    print('   curl -X POST "http://localhost:8000/classify" \\')
    print('        -H "Content-Type: application/json" \\')
    print(
        '        -d \'{"query": "My UK company sells software to German businesses"}\''
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
