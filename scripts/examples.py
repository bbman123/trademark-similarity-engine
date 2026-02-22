"""
Example usage demonstrating all features of the Trademark Similarity Engine
"""

import sys
from pathlib import Path

# Add project root to path (script lives in scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.svm_classifier import HybridSimilarityClassifier
from src.retrieval import TrademarkRetriever
from src.linguistic_features import LinguisticFeatureExtractor


def example_1_basic_similarity():
    """Example 1: Basic similarity checking"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Similarity Checking")
    print("=" * 80)
    
    # Initialize classifier
    classifier = HybridSimilarityClassifier()
    
    # Test pairs
    test_pairs = [
        ("SuperCoffee", "Super Coffee"),
        ("TechSmart", "SmartTech"),
        ("Premium Gold", "Golden Premium"),
        ("Fresh Market", "Old Store"),
    ]
    
    for mark1, mark2 in test_pairs:
        label, prob, _ = classifier.predict(mark1, mark2)
        
        print(f"\n'{mark1}' vs '{mark2}'")
        print(f"  → Prediction: {'Similar ⚠️' if label == 1 else 'Dissimilar ✓'}")
        print(f"  → Probability: {prob:.2%}")


def example_2_detailed_analysis():
    """Example 2: Detailed analysis with explanations"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Detailed Analysis with Explanations")
    print("=" * 80)
    
    classifier = HybridSimilarityClassifier()
    
    mark1 = "TechSmart"
    mark2 = "SmartTech"
    
    label, prob, details = classifier.predict(mark1, mark2, return_details=True)
    
    print(f"\nComparing: '{mark1}' vs '{mark2}'")
    print(f"\n🎯 Prediction: {'Similar' if label == 1 else 'Dissimilar'}")
    print(f"📊 Probability: {prob:.2%}")
    
    print(f"\n🔍 CNN Embedding Similarity: {details['cnn_features']['embedding_similarity']:.3f}")
    
    print("\n📋 Linguistic Features:")
    ling = details['linguistic_features']
    print(f"  • Jaro-Winkler: {ling['jaro_winkler']:.3f}")
    print(f"  • Semantic (EN): {ling['semantic_similarity_en']:.3f}")
    print(f"  • Semantic (HA): {ling['semantic_similarity_ha']:.3f}")
    print(f"  • Semantic (YO): {ling['semantic_similarity_yo']:.3f}")
    print(f"  • Soundex Match: {ling['soundex_match']}")
    print(f"  • Synonym Overlap: {ling['synonym_overlap_score']:.3f}")
    
    print("\n💡 Key Factors:")
    for factor in details['key_factors']:
        print(f"  • {factor}")


def example_3_linguistic_features():
    """Example 3: Extracting linguistic features"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Linguistic Feature Extraction")
    print("=" * 80)
    
    extractor = LinguisticFeatureExtractor()
    
    mark1 = "Premium Coffee"
    mark2 = "Quality Cafe"
    
    features = extractor.extract_all_features(mark1, mark2)
    
    print(f"\nAnalyzing: '{mark1}' vs '{mark2}'")
    print("\n📊 All Features:")
    for feature_name, value in features.items():
        if isinstance(value, float):
            print(f"  • {feature_name}: {value:.3f}")
        else:
            print(f"  • {feature_name}: {value}")
    
    # Synonym/Antonym demonstration
    print(f"\n🔍 Synonym Analysis:")
    syn1 = extractor.get_synonyms("premium")
    syn2 = extractor.get_synonyms("quality")
    print(f"  • 'premium' synonyms: {syn1}")
    print(f"  • 'quality' synonyms: {syn2}")
    print(f"  • Overlap: {syn1 & syn2}")


def example_4_batch_processing():
    """Example 4: Batch processing multiple pairs"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Batch Processing")
    print("=" * 80)
    
    classifier = HybridSimilarityClassifier()
    
    # Large batch of test pairs
    test_pairs = [
        ("BrandA", "BrandB"),
        ("SuperMarket", "Mega Market"),
        ("Tech Solutions", "Digital Solutions"),
        ("Fresh Foods", "Natural Foods"),
        ("Gold Premium", "Silver Standard"),
    ]
    
    print(f"\nProcessing {len(test_pairs)} pairs...\n")
    
    results = classifier.batch_predict(test_pairs, return_details=False)
    
    for (mark1, mark2), (label, prob, _) in zip(test_pairs, results):
        risk = "HIGH" if prob >= 0.7 else ("MEDIUM" if prob >= 0.5 else "LOW")
        emoji = "🔴" if risk == "HIGH" else ("🟡" if risk == "MEDIUM" else "🟢")
        
        print(f"{emoji} {mark1:20s} vs {mark2:20s} | {prob:6.1%} | {risk}")


def example_5_retrieval_system():
    """Example 5: Candidate retrieval system"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Retrieval System (Candidate Selection)")
    print("=" * 80)
    
    # Create sample trademark database
    sample_trademarks = [
        {"text": "SuperCoffee", "id": "TM001", "class": "30"},
        {"text": "MegaCoffee", "id": "TM002", "class": "30"},
        {"text": "Coffee Plus", "id": "TM003", "class": "30"},
        {"text": "TechSmart", "id": "TM004", "class": "9"},
        {"text": "SmartTech", "id": "TM005", "class": "9"},
        {"text": "Digital Solutions", "id": "TM006", "class": "42"},
        {"text": "Tech Solutions", "id": "TM007", "class": "42"},
        {"text": "Fresh Market", "id": "TM008", "class": "35"},
        {"text": "Premium Foods", "id": "TM009", "class": "29"},
        {"text": "Golden Coffee", "id": "TM010", "class": "30"},
    ]
    
    # Initialize and index
    retriever = TrademarkRetriever()
    print(f"\nIndexing {len(sample_trademarks)} trademarks...")
    retriever.index_trademarks(sample_trademarks)
    
    # Query
    query = "Super Coffee"
    print(f"\nQuery: '{query}'")
    print("\nTop 5 Similar Candidates:")
    
    candidates = retriever.retrieve_hybrid(
        query_text=query,
        top_k=5,
        threshold=0.3
    )
    
    for i, candidate in enumerate(candidates, 1):
        text = candidate['text']
        combined = candidate['scores']['combined']
        embedding = candidate['scores']['embedding']
        phonetic = "✓" if candidate['scores']['phonetic_match'] else "✗"
        
        print(f"\n{i}. {text}")
        print(f"   Combined Score: {combined:.3f}")
        print(f"   Embedding: {embedding:.3f} | Phonetic: {phonetic}")


def example_6_multilingual():
    """Example 6: Multilingual analysis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Multilingual Analysis (EN/HA/YO)")
    print("=" * 80)
    
    extractor = LinguisticFeatureExtractor()
    
    # English pair
    mark1_en = "Premium Coffee"
    mark2_en = "Quality Coffee"
    
    print(f"\nEnglish Comparison:")
    print(f"  '{mark1_en}' vs '{mark2_en}'")
    
    # Get local equivalents
    mark1_ha = extractor.get_local_equivalents(mark1_en, 'ha')
    mark2_ha = extractor.get_local_equivalents(mark2_en, 'ha')
    mark1_yo = extractor.get_local_equivalents(mark1_en, 'yo')
    mark2_yo = extractor.get_local_equivalents(mark2_en, 'yo')
    
    print(f"\nHausa Equivalents:")
    print(f"  '{mark1_ha}' vs '{mark2_ha}'")
    
    print(f"\nYoruba Equivalents:")
    print(f"  '{mark1_yo}' vs '{mark2_yo}'")
    
    # Compute similarities
    features = extractor.extract_all_features(mark1_en, mark2_en)
    
    print(f"\nSemantic Similarities:")
    print(f"  English: {features['semantic_similarity_en']:.3f}")
    print(f"  Hausa:   {features['semantic_similarity_ha']:.3f}")
    print(f"  Yoruba:  {features['semantic_similarity_yo']:.3f}")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("TRADEMARK SIMILARITY ENGINE - EXAMPLES")
    print("=" * 80)
    
    try:
        example_1_basic_similarity()
        example_2_detailed_analysis()
        example_3_linguistic_features()
        example_4_batch_processing()
        example_5_retrieval_system()
        example_6_multilingual()
        
        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
