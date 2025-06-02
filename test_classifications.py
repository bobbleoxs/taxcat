#!/usr/bin/env python3
"""
Comprehensive VAT Classification Test Script
Tests various UK VAT scenarios and edge cases against the TaxCat API.
"""

import re
import time
from typing import Any, Dict

import requests

# API configuration
API_BASE_URL = "http://127.0.0.1:8000"
CLASSIFY_ENDPOINT = f"{API_BASE_URL}/classify"


def extract_vat_info(response_text: str) -> Dict[str, str]:
    """Extract VAT rate, rationale, and confidence from API response."""

    # Extract VAT rate
    rate_match = re.search(
        r"VAT Rate[:\*]*[:\s]*([0-9]+%|exempt|zero|out_of_scope)",
        response_text,
        re.IGNORECASE,
    )
    vat_rate = rate_match.group(1) if rate_match else "Unknown"

    # Extract confidence
    confidence_match = re.search(
        r"Confidence[:\*]*[:\s]*(low|medium|high)", response_text, re.IGNORECASE
    )
    confidence = confidence_match.group(1) if confidence_match else "Unknown"

    # Extract rationale (first sentence or up to 100 chars)
    rationale_match = re.search(
        r"Short Rationale[:\*]*[:\s]*([^3]+?)(?=3\)|Source|$)",
        response_text,
        re.IGNORECASE | re.DOTALL,
    )
    if rationale_match:
        rationale = rationale_match.group(1).strip()
        # Take first sentence or truncate to reasonable length
        sentences = rationale.split(".")
        rationale = sentences[0][:120] + ("..." if len(sentences[0]) > 120 else "")
    else:
        rationale = "No rationale found"

    return {"rate": vat_rate, "confidence": confidence, "rationale": rationale}


def test_vat_classification(query: str, expected_rate: str = None) -> Dict[str, Any]:
    """Test a single VAT classification."""

    try:
        response = requests.post(
            CLASSIFY_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json={"text": query},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            vat_info = extract_vat_info(result["response"])

            return {
                "query": query,
                "expected": expected_rate,
                "status": "SUCCESS",
                "rate": vat_info["rate"],
                "confidence": vat_info["confidence"],
                "rationale": vat_info["rationale"],
                "full_response": result["response"],
            }
        else:
            return {
                "query": query,
                "expected": expected_rate,
                "status": "ERROR",
                "error": f"HTTP {response.status_code}: {response.text}",
                "rate": "N/A",
                "confidence": "N/A",
                "rationale": "N/A",
            }

    except requests.exceptions.RequestException as e:
        return {
            "query": query,
            "expected": expected_rate,
            "status": "ERROR",
            "error": str(e),
            "rate": "N/A",
            "confidence": "N/A",
            "rationale": "N/A",
        }


def print_test_result(result: Dict[str, Any], test_num: int):
    """Print formatted test result."""

    print(f"\n{'='*80}")
    print(f"TEST {test_num}: {result['query']}")
    print(f"{'='*80}")

    if result["expected"]:
        print(f"Expected Rate: {result['expected']}")

    print(f"Status: {result['status']}")

    if result["status"] == "SUCCESS":
        print(f"VAT Rate: {result['rate']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Rationale: {result['rationale']}")

        # Check if result matches expectation
        if result["expected"]:
            rate_clean = result["rate"].lower().replace("%", "")
            expected_clean = result["expected"].lower().replace("%", "")
            match = rate_clean == expected_clean
            print(f"Match Expected: {'✅ YES' if match else '❌ NO'}")
    else:
        print(f"Error: {result['error']}")

    print("-" * 80)


def check_api_health():
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy and running")
            return True
        else:
            print(f"❌ API health check failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False


def main():
    """Run comprehensive VAT classification tests."""

    print("🐱 TaxCat - Comprehensive VAT Classification Tests")
    print("=" * 60)

    # Check API health first
    if not check_api_health():
        print("\n❌ API is not available. Please start the server with:")
        print("uvicorn app:app --reload --port 8000")
        return

    # Define test cases covering various VAT scenarios
    test_cases = [
        # Zero-rated (0%) - Books, food, children's clothing
        ("Purchase of 10 children's books for £50", "0%"),
        ("Weekly grocery shopping for bread, milk, and vegetables £35", "0%"),
        ("Children's clothing - school uniforms for £80", "0%"),
        ("Prescription medicines from pharmacy £25", "0%"),
        ("Newspapers and magazines subscription £15", "0%"),
        # Reduced rate (5%) - Domestic fuel, children's car seats
        ("Domestic electricity bill for home £120", "5%"),
        ("Gas heating bill for residential property £200", "5%"),
        ("Child car seat for toddler £150", "5%"),
        # Standard rate (20%) - Most goods and services
        ("Office supplies including pens, paper, and staplers £100", "20%"),
        ("Restaurant meal for business lunch £45", "20%"),
        ("Computer software license for business £500", "20%"),
        ("Digital book download from Amazon £12", "20%"),
        ("Mobile phone and monthly contract £80", "20%"),
        ("Professional consulting services £2000", "20%"),
        ("Hotel accommodation for business trip £150", "20%"),
        # Edge cases - Books vs digital books
        ("Physical paperback novel from bookstore £8", "0%"),
        ("E-book download of same novel £6", "20%"),
        # Services vs goods
        ("Accounting services for year-end accounts £800", "20%"),
        ("Purchase of accounting software £300", "20%"),
        # Export transactions
        ("Export of goods to France £5000", "0%"),
        ("Services provided to client in Germany £3000", "0%"),
        # Exempt items
        ("Private health insurance premium £100", "exempt"),
        ("University tuition fees £9000", "exempt"),
        ("Bank loan interest charges £50", "exempt"),
        # Mixed and complex scenarios
        ("Construction services for new office building £50000", "20%"),
        ("Medical equipment for hospital £10000", "0%"),
        ("Charity donation to registered charity £500", "exempt"),
    ]

    print(f"\n🧪 Running {len(test_cases)} VAT classification tests...\n")

    results = []
    success_count = 0
    match_count = 0

    for i, (query, expected) in enumerate(test_cases, 1):
        print(f"Running test {i}/{len(test_cases)}...")
        result = test_vat_classification(query, expected)
        results.append(result)

        print_test_result(result, i)

        if result["status"] == "SUCCESS":
            success_count += 1

            # Check if result matches expectation
            if expected:
                rate_clean = result["rate"].lower().replace("%", "")
                expected_clean = expected.lower().replace("%", "")
                if rate_clean == expected_clean:
                    match_count += 1

        # Small delay between requests
        time.sleep(0.5)

    # Print summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests: {len(test_cases)}")
    print(f"Successful Calls: {success_count}")
    print(f"Failed Calls: {len(test_cases) - success_count}")
    print(f"Matches Expected: {match_count}")
    print(f"Success Rate: {success_count/len(test_cases)*100:.1f}%")
    if success_count > 0:
        print(f"Accuracy Rate: {match_count/success_count*100:.1f}%")

    # Show confidence distribution
    confidence_counts = {}
    for result in results:
        if result["status"] == "SUCCESS":
            conf = result["confidence"]
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

    if confidence_counts:
        print("\nConfidence Distribution:")
        for conf, count in confidence_counts.items():
            print(f"  {conf}: {count} tests")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
