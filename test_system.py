"""
Test script to validate the modular implementation
"""

import sys
from pathlib import Path
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "=" * 80)
    print("TEST 1: Module Imports")
    print("=" * 80)
    
    try:
        from src import config
        print("✓ config module")
        
        from src import cache_manager
        print("✓ cache_manager module")
        
        from src import cnn_encoder
        print("✓ cnn_encoder module")
        
        from src import linguistic_features
        print("✓ linguistic_features module")
        
        from src import svm_classifier
        print("✓ svm_classifier module")
        
        from src import retrieval
        print("✓ retrieval module")
        
        from src import api_service
        print("✓ api_service module")
        
        print("\n✅ All modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_config():
    """Test configuration"""
    print("\n" + "=" * 80)
    print("TEST 2: Configuration")
    print("=" * 80)
    
    try:
        from src.config import config
        
        print(f"✓ BASE_DIR: {config.BASE_DIR}")
        print(f"✓ MODEL_DIR: {config.MODEL_DIR}")
        print(f"✓ CNN_MODEL_PATH: {config.CNN_MODEL_PATH}")
        print(f"✓ SVM_MODEL_PATH: {config.SVM_MODEL_PATH}")
        print(f"✓ Languages: {config.LANGUAGES}")
        
        print("\n✅ Configuration loaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_cnn_encoder():
    """Test CNN encoder"""
    print("\n" + "=" * 80)
    print("TEST 3: CNN Encoder")
    print("=" * 80)
    
    try:
        from src.cnn_encoder import CNNEncoder
        
        print("Loading CNN encoder...")
        encoder = CNNEncoder()
        
        # Test encoding
        test_texts = ["SuperCoffee", "TechSmart"]
        print(f"\nTesting with: {test_texts}")
        
        embeddings = encoder.get_embeddings(test_texts, use_cache=False)
        print(f"✓ Embeddings shape: {embeddings.shape}")
        
        # Test single embedding
        single_emb = encoder.get_single_embedding("TestMark")
        print(f"✓ Single embedding shape: {single_emb.shape}")
        
        # Test similarity
        sim = encoder.compute_similarity("SuperCoffee", "Super Coffee")
        print(f"✓ Similarity score: {sim:.3f}")
        
        print("\n✅ CNN encoder working!")
        return True
        
    except Exception as e:
        print(f"\n❌ CNN encoder test failed: {e}")
        traceback.print_exc()
        return False


def test_linguistic_features():
    """Test linguistic feature extractor"""
    print("\n" + "=" * 80)
    print("TEST 4: Linguistic Features")
    print("=" * 80)
    
    try:
        from src.linguistic_features import LinguisticFeatureExtractor
        
        print("Initializing linguistic feature extractor...")
        extractor = LinguisticFeatureExtractor()
        
        # Test feature extraction
        mark1 = "Premium Coffee"
        mark2 = "Quality Cafe"
        print(f"\nTesting with: '{mark1}' vs '{mark2}'")
        
        features = extractor.extract_all_features(mark1, mark2)
        print(f"✓ Extracted {len(features)} features")
        
        # Test specific features
        print(f"\nKey features:")
        print(f"  • Synonym overlap: {features['synonym_overlap_score']:.3f}")
        print(f"  • Jaro-Winkler: {features['jaro_winkler']:.3f}")
        print(f"  • Semantic (EN): {features['semantic_similarity_en']:.3f}")
        print(f"  • Soundex match: {features['soundex_match']}")
        
        # Test synonyms
        synonyms = extractor.get_synonyms("premium")
        print(f"\n✓ Synonyms for 'premium': {synonyms}")
        
        print("\n✅ Linguistic features working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Linguistic features test failed: {e}")
        traceback.print_exc()
        return False


def test_svm_classifier():
    """Test SVM classifier"""
    print("\n" + "=" * 80)
    print("TEST 5: SVM Classifier")
    print("=" * 80)
    
    try:
        from src.svm_classifier import SVMClassifier
        
        print("Loading SVM classifier...")
        svm = SVMClassifier()
        
        print(f"✓ SVM model loaded")
        print(f"✓ Scaler loaded")
        
        print("\n✅ SVM classifier loaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ SVM classifier test failed: {e}")
        traceback.print_exc()
        return False


def test_hybrid_classifier():
    """Test hybrid classifier (end-to-end)"""
    print("\n" + "=" * 80)
    print("TEST 6: Hybrid Classifier (End-to-End)")
    print("=" * 80)
    
    try:
        from src.svm_classifier import HybridSimilarityClassifier
        
        print("Initializing hybrid classifier...")
        classifier = HybridSimilarityClassifier()
        
        # Test prediction
        mark1 = "SuperCoffee"
        mark2 = "Super Coffee"
        print(f"\nTesting: '{mark1}' vs '{mark2}'")
        
        label, prob, details = classifier.predict(mark1, mark2, return_details=True)
        
        print(f"\n✓ Prediction: {'Similar' if label == 1 else 'Dissimilar'}")
        print(f"✓ Probability: {prob:.3f}")
        print(f"✓ Details: {len(details)} components")
        
        # Test batch
        print("\nTesting batch prediction...")
        pairs = [("BrandA", "BrandB"), ("TechX", "TechY")]
        results = classifier.batch_predict(pairs)
        print(f"✓ Batch results: {len(results)} predictions")
        
        print("\n✅ Hybrid classifier working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Hybrid classifier test failed: {e}")
        traceback.print_exc()
        return False


def test_cache():
    """Test caching system"""
    print("\n" + "=" * 80)
    print("TEST 7: Caching System")
    print("=" * 80)
    
    try:
        from src.cache_manager import CacheManager
        from pathlib import Path
        
        cache_dir = Path("test_cache")
        print(f"Creating test cache at: {cache_dir}")
        
        cache = CacheManager(cache_dir, ttl_seconds=60)
        
        # Test set/get
        cache.set("test_key", {"data": "test_value"})
        result = cache.get("test_key")
        assert result == {"data": "test_value"}
        print("✓ Cache set/get working")
        
        # Test miss
        result = cache.get("non_existent_key")
        assert result is None
        print("✓ Cache miss handled")
        
        # Cleanup
        cache.clear()
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
        
        print("\n✅ Caching system working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Cache test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("TRADEMARK SIMILARITY ENGINE - VALIDATION TESTS")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("CNN Encoder", test_cnn_encoder),
        ("Linguistic Features", test_linguistic_features),
        ("SVM Classifier", test_svm_classifier),
        ("Hybrid Classifier", test_hybrid_classifier),
        ("Caching System", test_cache),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 80 + "\n")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
