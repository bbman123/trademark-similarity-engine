"""
SVM Classifier for trademark similarity prediction
"""

import pickle
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging

from .config import config
from .cnn_encoder import CNNEncoder
from .linguistic_features import LinguisticFeatureExtractor

logger = logging.getLogger(__name__)


class SVMClassifier:
    """SVM classifier using hybrid CNN+linguistic features"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize SVM classifier
        
        Args:
            model_path: Path to trained SVM model pickle file
        """
        self.model_path = model_path or config.SVM_MODEL_PATH
        self.svm = None
        self.scaler = None
        
        # Load model
        self._load()
    
    def _load(self):
        """Load trained SVM and scaler"""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.svm = data['svm']
                self.scaler = data['scaler']
            logger.info(f"✓ Loaded SVM model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load SVM model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict similarity labels
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Binary predictions (0=dissimilar, 1=similar)
        """
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.svm.predict(features_scaled)
        return predictions
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict similarity probabilities
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Probability of similarity (0-1)
        """
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict probabilities
        probas = self.svm.predict_proba(features_scaled)[:, 1]
        return probas
    
    def get_decision_function(self, features: np.ndarray) -> np.ndarray:
        """
        Get decision function values (distance from hyperplane)
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Decision function values
        """
        features_scaled = self.scaler.transform(features)
        return self.svm.decision_function(features_scaled)


class HybridSimilarityClassifier:
    """
    Complete hybrid classifier combining CNN + Linguistic Features + SVM
    """
    
    def __init__(
        self,
        cnn_encoder: Optional[CNNEncoder] = None,
        linguistic_extractor: Optional[LinguisticFeatureExtractor] = None,
        svm_classifier: Optional[SVMClassifier] = None
    ):
        """
        Initialize hybrid classifier
        
        Args:
            cnn_encoder: CNN encoder instance (loads default if None)
            linguistic_extractor: Linguistic feature extractor (creates new if None)
            svm_classifier: SVM classifier (loads default if None)
        """
        self.cnn_encoder = cnn_encoder or CNNEncoder()
        self.linguistic_extractor = linguistic_extractor or LinguisticFeatureExtractor()
        self.svm_classifier = svm_classifier or SVMClassifier()
        
        logger.info("✓ Hybrid classifier initialized")
    
    def extract_hybrid_features(
        self, 
        text1: str, 
        text2: str,
        include_multilingual: bool = True
    ) -> np.ndarray:
        """
        Extract combined CNN + linguistic features for a pair
        
        Args:
            text1: First trademark text
            text2: Second trademark text
            include_multilingual: Whether to include HA/YO features
            
        Returns:
            Combined feature vector
        """
        # Get CNN embeddings
        emb1 = self.cnn_encoder.get_single_embedding(text1)
        emb2 = self.cnn_encoder.get_single_embedding(text2)
        
        # Get linguistic features
        ling_features = self.linguistic_extractor.extract_features_as_vector(
            text1, text2, include_multilingual=include_multilingual
        )
        
        # Concatenate: [emb1, emb2, linguistic_features]
        hybrid_features = np.concatenate([emb1, emb2, ling_features])
        
        return hybrid_features
    
    def predict(
        self, 
        text1: str, 
        text2: str,
        return_details: bool = False
    ) -> Tuple[int, float, Optional[Dict]]:
        """
        Predict if two trademarks are similar
        
        Args:
            text1: First trademark text
            text2: Second trademark text
            return_details: Whether to return detailed feature breakdown
            
        Returns:
            Tuple of:
            - label: 0 (dissimilar) or 1 (similar)
            - probability: Similarity probability (0-1)
            - details: Optional dictionary with feature values and explanations
        """
        # Extract features
        features = self.extract_hybrid_features(text1, text2).reshape(1, -1)
        
        # Predict
        label = int(self.svm_classifier.predict(features)[0])
        probability = float(self.svm_classifier.predict_proba(features)[0])
        
        # Get details if requested
        details = None
        if return_details:
            details = self._get_prediction_details(text1, text2, features, label, probability)
        
        return label, probability, details
    
    def _get_prediction_details(
        self,
        text1: str,
        text2: str,
        features: np.ndarray,
        label: int,
        probability: float
    ) -> Dict:
        """
        Generate detailed explanation of prediction
        
        Returns:
            Dictionary with feature breakdown and explanations
        """
        # Get individual feature components
        ling_features_dict = self.linguistic_extractor.extract_all_features(text1, text2)
        
        # Get CNN similarity
        cnn_similarity = self.cnn_encoder.compute_similarity(text1, text2)
        
        # Build explanation
        details = {
            'prediction': {
                'label': label,
                'label_text': 'Similar' if label == 1 else 'Dissimilar',
                'probability': probability,
                'confidence': abs(probability - 0.5) * 2  # 0-1 scale
            },
            'cnn_features': {
                'embedding_similarity': cnn_similarity
            },
            'linguistic_features': ling_features_dict,
            'key_factors': self._identify_key_factors(ling_features_dict, cnn_similarity)
        }
        
        return details
    
    def _identify_key_factors(
        self,
        ling_features: Dict[str, float],
        cnn_similarity: float
    ) -> list:
        """
        Identify key factors contributing to similarity/dissimilarity
        
        Returns:
            List of explanation strings
        """
        factors = []
        
        # CNN similarity
        if cnn_similarity > 0.8:
            factors.append(f"High visual embedding similarity ({cnn_similarity:.2f})")
        elif cnn_similarity < 0.3:
            factors.append(f"Low visual embedding similarity ({cnn_similarity:.2f})")
        
        # Phonetic match
        if ling_features.get('soundex_match', 0) == 1:
            factors.append("Same phonetic encoding (sounds alike)")
        if ling_features.get('metaphone_match', 0) == 1:
            factors.append("Same metaphone code (pronunciation match)")
        
        # Spelling similarity
        jaro = ling_features.get('jaro_winkler', 0)
        if jaro > 0.85:
            factors.append(f"Very similar spelling (Jaro-Winkler: {jaro:.2f})")
        elif jaro < 0.4:
            factors.append(f"Very different spelling (Jaro-Winkler: {jaro:.2f})")
        
        # Synonym overlap
        syn_overlap = ling_features.get('synonym_overlap_score', 0)
        if syn_overlap > 0.5:
            factors.append(f"High synonym overlap ({syn_overlap:.2f})")
        
        # Antonym presence
        if ling_features.get('antonym_flag', 0) == 1:
            count = ling_features.get('antonym_count', 0)
            factors.append(f"Contains opposite meanings ({count} antonym pairs)")
        
        # Semantic similarity
        sem_en = ling_features.get('semantic_similarity_en', 0)
        if sem_en > 0.7:
            factors.append(f"High semantic similarity ({sem_en:.2f})")
        elif sem_en < 0.3:
            factors.append(f"Low semantic similarity ({sem_en:.2f})")
        
        # Multilingual
        sem_ha = ling_features.get('semantic_similarity_ha', 0)
        sem_yo = ling_features.get('semantic_similarity_yo', 0)
        if sem_ha > 0.7 or sem_yo > 0.7:
            factors.append(f"Similar in local languages (HA: {sem_ha:.2f}, YO: {sem_yo:.2f})")
        
        return factors
    
    def batch_predict(
        self,
        text_pairs: list,
        return_details: bool = False
    ) -> list:
        """
        Predict similarity for multiple pairs
        
        Args:
            text_pairs: List of (text1, text2) tuples
            return_details: Whether to return detailed explanations
            
        Returns:
            List of prediction results
        """
        results = []
        
        for text1, text2 in text_pairs:
            result = self.predict(text1, text2, return_details=return_details)
            results.append(result)
        
        return results
