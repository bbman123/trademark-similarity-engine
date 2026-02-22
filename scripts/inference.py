"""
Main inference script for standalone trademark similarity checking
"""

import argparse
import json
from pathlib import Path
import sys

# Add project root to path (script lives in scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.svm_classifier import HybridSimilarityClassifier
from src.config import config


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(
        description="Trademark Similarity Engine - Check similarity between trademarks"
    )
    
    parser.add_argument(
        '--mark1',
        type=str,
        required=True,
        help='First trademark text'
    )
    
    parser.add_argument(
        '--mark2',
        type=str,
        required=True,
        help='Second trademark text'
    )
    
    parser.add_argument(
        '--details',
        action='store_true',
        help='Show detailed feature breakdown and explanation'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    # Load classifier
    print("Loading models...")
    classifier = HybridSimilarityClassifier()
    print("✓ Models loaded\n")
    
    # Predict
    label, probability, details = classifier.predict(
        args.mark1,
        args.mark2,
        return_details=args.details
    )
    
    # Determine risk level
    if probability >= 0.7:
        risk_level = "HIGH"
        risk_emoji = "🔴"
    elif probability >= 0.5:
        risk_level = "MEDIUM"
        risk_emoji = "🟡"
    else:
        risk_level = "LOW"
        risk_emoji = "🟢"
    
    # Output results
    if args.json:
        output = {
            'mark1': args.mark1,
            'mark2': args.mark2,
            'prediction': {
                'label': int(label),
                'label_text': 'Similar' if label == 1 else 'Dissimilar',
                'probability': float(probability),
                'risk_level': risk_level
            }
        }
        
        if args.details and details:
            output['details'] = details
        
        print(json.dumps(output, indent=2))
    
    else:
        print("=" * 80)
        print("TRADEMARK SIMILARITY ANALYSIS")
        print("=" * 80)
        print(f"\nMark 1: {args.mark1}")
        print(f"Mark 2: {args.mark2}")
        print(f"\n{risk_emoji} Risk Level: {risk_level}")
        print(f"   Similarity Probability: {probability:.2%}")
        print(f"   Prediction: {'⚠️  SIMILAR - Potential confusion' if label == 1 else '✓ DISSIMILAR - No confusion expected'}")
        
        if args.details and details:
            print("\n" + "=" * 80)
            print("DETAILED ANALYSIS")
            print("=" * 80)
            
            # CNN features
            print(f"\n🔍 Visual Embedding Similarity: {details['cnn_features']['embedding_similarity']:.3f}")
            
            # Key linguistic features
            print("\n📊 Key Linguistic Features:")
            ling = details['linguistic_features']
            
            print(f"   • Spelling Similarity (Jaro-Winkler): {ling.get('jaro_winkler', 0):.3f}")
            print(f"   • Semantic Similarity (English): {ling.get('semantic_similarity_en', 0):.3f}")
            print(f"   • Phonetic Match (Soundex): {'Yes' if ling.get('soundex_match', 0) else 'No'}")
            print(f"   • Phonetic Match (Metaphone): {'Yes' if ling.get('metaphone_match', 0) else 'No'}")
            print(f"   • Synonym Overlap: {ling.get('synonym_overlap_score', 0):.3f}")
            print(f"   • Antonym Flag: {'⚠️  Yes' if ling.get('antonym_flag', 0) else 'No'}")
            
            # Multilingual
            if 'semantic_similarity_ha' in ling:
                print(f"\n🌍 Multilingual Analysis:")
                print(f"   • Hausa Semantic Similarity: {ling.get('semantic_similarity_ha', 0):.3f}")
                print(f"   • Yoruba Semantic Similarity: {ling.get('semantic_similarity_yo', 0):.3f}")
            
            # Key factors
            if 'key_factors' in details and details['key_factors']:
                print(f"\n💡 Key Factors Contributing to Decision:")
                for factor in details['key_factors']:
                    print(f"   • {factor}")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
