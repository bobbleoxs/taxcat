# test_specific_queries.py
import json
import sys

# Add scripts directory to path
sys.path.append("scripts")

from query_classifier import VATQueryClassifier


def test_specific_queries():
    """Test the classifier with specific problematic queries"""

    classifier = VATQueryClassifier()

    # Test queries from user
    test_cases = [
        {
            "id": 1,
            "query": "Google Cloud from Ireland to UK",
            "expected": "B2B digital services from EU to UK",
        },
        {
            "id": 2,
            "query": "Microsoft Azure from Ireland to UK",
            "expected": "B2B digital services from EU to UK",
        },
        {
            "id": 3,
            "query": "Digital service from EU to UK",
            "expected": "B2B digital services from EU to UK",
        },
        {
            "id": 4,
            "query": "Software subscription from Ireland to UK",
            "expected": "B2B digital services from EU to UK",
        },
        {
            "id": 5,
            "query": "Personal Spotify subscription from Ireland",
            "expected": "B2C digital service",
        },
        {
            "id": 6,
            "query": "Adobe Creative Suite for my design business from US",
            "expected": "B2B digital service from US",
        },
    ]

    print("🔍 Testing Specific VAT Query Classifications")
    print("=" * 60)

    for test_case in test_cases:
        print(f"\n📋 Test {test_case['id']}: {test_case['query']}")
        print(f"Expected: {test_case['expected']}")
        print("-" * 50)

        # Classify the query
        classification = classifier.classify_query(test_case["query"])

        # Convert to JSON for display
        classification_json = classifier.to_json(classification)

        # Parse back to dict for easier analysis
        classification_dict = json.loads(classification_json)

        # Show key results
        print(f"✅ Business Type: {classification_dict['business_type']}")
        print(f"✅ Service Type: {classification_dict['service_type']}")
        print(f"✅ Supplier Location: {classification_dict['supplier_location']}")
        print(f"✅ Customer Location: {classification_dict['customer_location']}")
        print(f"✅ Cross-border: {classification_dict['is_cross_border']}")
        print(f"✅ Confidence: {classification_dict['confidence_score']:.2f}")

        if classification_dict["ambiguities"]:
            print(f"⚠️  Ambiguities: {', '.join(classification_dict['ambiguities'])}")

        # Show key indicators
        print("\n🔎 Key Indicators Found:")
        for indicator_type, indicators in classification_dict["key_indicators"].items():
            if indicators:
                print(f"  {indicator_type}: {indicators}")

        # Full JSON output
        print("\n📄 Full JSON Output:")
        print(json.dumps(classification_dict, indent=2))

        # Analysis
        print("\n📊 Analysis:")

        # Check if classification matches expectations
        if test_case["id"] in [1, 2, 3, 4]:  # Should be B2B digital EU->UK
            correct_business = classification_dict["business_type"] == "B2B"
            correct_service = classification_dict["service_type"] == "digital"
            correct_locations = (
                classification_dict["supplier_location"] == "eu"
                and classification_dict["customer_location"] == "uk"
            )
            correct_cross_border = classification_dict["is_cross_border"]

            all_correct = (
                correct_business
                and correct_service
                and correct_locations
                and correct_cross_border
            )

            print(
                f"  Business Type Correct: {'✅' if correct_business else '❌'} (Expected B2B, Got {classification_dict['business_type']})"
            )
            print(
                f"  Service Type Correct: {'✅' if correct_service else '❌'} (Expected digital, Got {classification_dict['service_type']})"
            )
            print(
                f"  Locations Correct: {'✅' if correct_locations else '❌'} (Expected EU->UK, Got {classification_dict['supplier_location']}→{classification_dict['customer_location']})"
            )
            print(
                f"  Cross-border Correct: {'✅' if correct_cross_border else '❌'} (Expected True, Got {classification_dict['is_cross_border']})"
            )
            print(f"  Overall: {'✅ PASS' if all_correct else '❌ FAIL'}")

        elif test_case["id"] == 5:  # Should be B2C
            correct_business = classification_dict["business_type"] == "B2C"
            correct_service = classification_dict["service_type"] == "digital"

            print(
                f"  Business Type Correct: {'✅' if correct_business else '❌'} (Expected B2C, Got {classification_dict['business_type']})"
            )
            print(
                f"  Service Type Correct: {'✅' if correct_service else '❌'} (Expected digital, Got {classification_dict['service_type']})"
            )
            print(
                f"  Overall: {'✅ PASS' if correct_business and correct_service else '❌ FAIL'}"
            )

        elif test_case["id"] == 6:  # Should be B2B from US
            correct_business = classification_dict["business_type"] == "B2B"
            correct_service = classification_dict["service_type"] == "digital"
            correct_supplier = classification_dict["supplier_location"] == "us"

            print(
                f"  Business Type Correct: {'✅' if correct_business else '❌'} (Expected B2B, Got {classification_dict['business_type']})"
            )
            print(
                f"  Service Type Correct: {'✅' if correct_service else '❌'} (Expected digital, Got {classification_dict['service_type']})"
            )
            print(
                f"  Supplier Location Correct: {'✅' if correct_supplier else '❌'} (Expected US, Got {classification_dict['supplier_location']})"
            )
            print(
                f"  Overall: {'✅ PASS' if correct_business and correct_service and correct_supplier else '❌ FAIL'}"
            )

        print("\n" + "=" * 60)


if __name__ == "__main__":
    test_specific_queries()
