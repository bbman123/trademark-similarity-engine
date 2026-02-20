"""
Enhanced Linguistic Features with Synonyms, Antonyms, Phonetics, and Multilingual Support
"""

import re
import jellyfish
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .config import config
from .cache_manager import CacheManager, cached

logger = logging.getLogger(__name__)


# ============================================================================
# ENHANCED LEXICON WITH SYNONYMS & ANTONYMS
# ============================================================================

TRADEMARK_LEXICON_ENHANCED = {
    # Business & Commerce
    "company": {
        "ha": "kamfani", "yo": "ilé-iṣẹ́",
        "synonyms": ["business", "corporation", "firm", "enterprise", "organization"],
        "antonyms": ["individual", "person", "consumer"]
    },
    "business": {
        "ha": "kasuwanci", "yo": "iṣẹ́ owó",
        "synonyms": ["company", "enterprise", "trade", "commerce", "venture"],
        "antonyms": ["leisure", "hobby", "personal"]
    },
    "market": {
        "ha": "kasuwa", "yo": "ọjà",
        "synonyms": ["marketplace", "bazaar", "store", "shop", "outlet"],
        "antonyms": ["storage", "warehouse"]
    },
    "store": {
        "ha": "shago", "yo": "ṣọ́ọ̀bù",
        "synonyms": ["shop", "outlet", "mart", "market", "retailer"],
        "antonyms": []
    },
    "brand": {
        "ha": "alama", "yo": "àmì",
        "synonyms": ["trademark", "label", "logo", "mark", "name"],
        "antonyms": ["generic", "unbranded"]
    },
    
    # Quality
    "premium": {
        "ha": "mai kyau", "yo": "iyebíye",
        "synonyms": ["quality", "superior", "elite", "deluxe", "luxury"],
        "antonyms": ["basic", "standard", "economy", "cheap"]
    },
    "super": {
        "ha": "babba", "yo": "púpọ̀",
        "synonyms": ["mega", "ultra", "great", "supreme", "excellent"],
        "antonyms": ["mini", "small", "inferior"]
    },
    "best": {
        "ha": "mafi kyau", "yo": "dára jùlọ",
        "synonyms": ["finest", "top", "premier", "supreme", "optimal"],
        "antonyms": ["worst", "poorest", "inferior"]
    },
    "gold": {
        "ha": "zinariya", "yo": "wúrà",
        "synonyms": ["golden", "premium", "deluxe"],
        "antonyms": ["bronze", "silver"]
    },
    
    # Food & Beverage
    "coffee": {
        "ha": "kofi", "yo": "kọ́fí",
        "synonyms": ["cafe", "caffeine", "java"],
        "antonyms": ["tea"]
    },
    "tea": {
        "ha": "shayi", "yo": "tii",
        "synonyms": ["beverage", "drink"],
        "antonyms": ["coffee"]
    },
    "food": {
        "ha": "abinci", "yo": "oúnjẹ",
        "synonyms": ["meal", "cuisine", "dish", "nourishment"],
        "antonyms": []
    },
    "restaurant": {
        "ha": "gidan cin abinci", "yo": "ilé oúnjẹ",
        "synonyms": ["cafe", "diner", "eatery", "bistro"],
        "antonyms": []
    },
    
    # Technology
    "tech": {
        "ha": "fasaha", "yo": "ìmọ̀-ẹrọ",
        "synonyms": ["technology", "digital", "electronic", "smart"],
        "antonyms": ["analog", "manual", "traditional"]
    },
    "digital": {
        "ha": "na dijital", "yo": "oníjìtù",
        "synonyms": ["electronic", "tech", "cyber", "online"],
        "antonyms": ["analog", "physical", "manual"]
    },
    "smart": {
        "ha": "mai hankali", "yo": "ológbọ́n",
        "synonyms": ["intelligent", "clever", "advanced", "tech"],
        "antonyms": ["basic", "simple", "dumb"]
    },
    
    # Common terms
    "new": {
        "ha": "sabon", "yo": "títun",
        "synonyms": ["fresh", "modern", "novel", "recent", "latest"],
        "antonyms": ["old", "vintage", "classic", "ancient"]
    },
    "fresh": {
        "ha": "sabo", "yo": "tuntun",
        "synonyms": ["new", "crisp", "pure", "natural"],
        "antonyms": ["stale", "old", "spoiled"]
    },
    "natural": {
        "ha": "na halitta", "yo": "àdáyébá",
        "synonyms": ["organic", "pure", "authentic", "real"],
        "antonyms": ["artificial", "synthetic", "fake"]
    },
}


class LinguisticFeatureExtractor:
    """Extract comprehensive linguistic features including synonyms/antonyms"""
    
    def __init__(self):
        self.lexicon = TRADEMARK_LEXICON_ENHANCED
        self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.cache_manager = CacheManager(config.CACHE_DIR / "linguistic")
        
        # Build reverse lookup for faster synonym/antonym checking
        self._build_lookup_indices()
    
    def _build_lookup_indices(self):
        """Build indices for fast synonym/antonym lookup"""
        self.word_to_synonyms = {}
        self.word_to_antonyms = {}
        
        for word, data in self.lexicon.items():
            synonyms = set(data.get('synonyms', []))
            synonyms.add(word)  # Include the word itself
            self.word_to_synonyms[word] = synonyms
            
            # Add reverse mappings
            for syn in synonyms:
                if syn not in self.word_to_synonyms:
                    self.word_to_synonyms[syn] = set()
                self.word_to_synonyms[syn].update(synonyms)
            
            # Antonyms
            antonyms = set(data.get('antonyms', []))
            self.word_to_antonyms[word] = antonyms
            
            for ant in antonyms:
                if ant not in self.word_to_antonyms:
                    self.word_to_antonyms[ant] = set()
                self.word_to_antonyms[ant].add(word)
    
    def get_synonyms(self, word: str) -> Set[str]:
        """Get all synonyms for a word"""
        word_lower = word.lower().strip()
        return self.word_to_synonyms.get(word_lower, set())
    
    def get_antonyms(self, word: str) -> Set[str]:
        """Get all antonyms for a word"""
        word_lower = word.lower().strip()
        return self.word_to_antonyms.get(word_lower, set())
    
    def extract_words(self, text: str) -> List[str]:
        """Extract words from text"""
        if not text or not str(text).strip():
            return []
        return [w.lower() for w in re.findall(r'\b\w+\b', str(text))]
    
    # ========================================================================
    # SYNONYM & ANTONYM FEATURES
    # ========================================================================
    
    def compute_synonym_overlap(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Compute synonym overlap between two texts
        
        Returns:
            Dictionary with:
            - synonym_overlap_score: Jaccard similarity of expanded synonym sets
            - synonym_exact_matches: Count of exact synonym matches
        """
        words1 = self.extract_words(text1)
        words2 = self.extract_words(text2)
        
        # Expand with synonyms
        expanded1 = set()
        for word in words1:
            expanded1.update(self.get_synonyms(word))
        
        expanded2 = set()
        for word in words2:
            expanded2.update(self.get_synonyms(word))
        
        # Jaccard similarity
        if not expanded1 or not expanded2:
            overlap_score = 0.0
        else:
            intersection = len(expanded1 & expanded2)
            union = len(expanded1 | expanded2)
            overlap_score = intersection / union if union > 0 else 0.0
        
        # Count exact matches in original words
        exact_matches = len(set(words1) & set(words2))
        
        return {
            'synonym_overlap_score': overlap_score,
            'synonym_exact_matches': exact_matches,
            'synonym_expanded_intersection': len(expanded1 & expanded2)
        }
    
    def compute_antonym_presence(self, text1: str, text2: str) -> Dict[str, int]:
        """
        Check if texts contain antonyms (indicates dissimilarity)
        
        Returns:
            Dictionary with:
            - antonym_flag: 1 if antonyms present, 0 otherwise
            - antonym_count: Number of antonym pairs found
        """
        words1 = set(self.extract_words(text1))
        words2 = set(self.extract_words(text2))
        
        antonym_count = 0
        for word1 in words1:
            antonyms = self.get_antonyms(word1)
            if antonyms & words2:
                antonym_count += 1
        
        return {
            'antonym_flag': 1 if antonym_count > 0 else 0,
            'antonym_count': antonym_count
        }
    
    # ========================================================================
    # PHONETIC FEATURES
    # ========================================================================
    
    def compute_phonetic_features(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Compute phonetic similarity features
        
        Returns:
            Dictionary with phonetic comparison scores
        """
        def safe_soundex(text):
            try:
                if not text or not str(text).strip():
                    return ""
                return jellyfish.soundex(str(text))
            except:
                return ""
        
        def safe_metaphone(text):
            try:
                if not text or not str(text).strip():
                    return ""
                return jellyfish.metaphone(str(text))
            except:
                return ""
        
        # Soundex comparison
        soundex1 = safe_soundex(text1)
        soundex2 = safe_soundex(text2)
        soundex_match = int(soundex1 == soundex2 and soundex1 != "")
        
        # Metaphone comparison
        metaphone1 = safe_metaphone(text1)
        metaphone2 = safe_metaphone(text2)
        metaphone_match = int(metaphone1 == metaphone2 and metaphone1 != "")
        
        return {
            'soundex_match': soundex_match,
            'metaphone_match': metaphone_match
        }
    
    # ========================================================================
    # VISUAL/SPELLING FEATURES
    # ========================================================================
    
    def compute_visual_features(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Compute visual/spelling similarity features
        
        Returns:
            Dictionary with visual comparison scores
        """
        text1_str = str(text1) if text1 else ""
        text2_str = str(text2) if text2 else ""
        
        # Levenshtein distance (normalized)
        lev_dist = jellyfish.levenshtein_distance(text1_str, text2_str)
        max_len = max(len(text1_str), len(text2_str))
        lev_normalized = 1.0 - (lev_dist / max_len) if max_len > 0 else 0.0
        
        # Jaro-Winkler similarity
        jaro_winkler = jellyfish.jaro_winkler_similarity(text1_str, text2_str)
        
        # Hamming distance (if same length)
        if len(text1_str) == len(text2_str) and len(text1_str) > 0:
            hamming_dist = jellyfish.hamming_distance(text1_str, text2_str)
            hamming_normalized = 1.0 - (hamming_dist / len(text1_str))
        else:
            hamming_normalized = 0.0
        
        return {
            'levenshtein_normalized': lev_normalized,
            'jaro_winkler': jaro_winkler,
            'hamming_normalized': hamming_normalized,
            'length_diff': abs(len(text1_str) - len(text2_str))
        }
    
    # ========================================================================
    # SEMANTIC FEATURES (MULTILINGUAL)
    # ========================================================================
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity using multilingual embeddings
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        cache_key = self.cache_manager._get_cache_key(text1, text2)
        cached_sim = self.cache_manager.get(cache_key)
        if cached_sim is not None:
            return cached_sim
        
        # Handle empty texts
        if not text1 or not text2:
            return 0.0
        
        # Encode
        emb1 = self.semantic_model.encode([str(text1)], show_progress_bar=False)
        emb2 = self.semantic_model.encode([str(text2)], show_progress_bar=False)
        
        # Cosine similarity
        sim = float(cosine_similarity(emb1, emb2)[0][0])
        
        # Cache
        self.cache_manager.set(cache_key, sim)
        
        return sim
    
    # ========================================================================
    # MULTILINGUAL TRANSLATION FEATURES
    # ========================================================================
    
    def get_local_equivalents(self, text: str, lang: str) -> str:
        """
        Get Hausa or Yoruba equivalent for English text
        
        Args:
            text: English text
            lang: 'ha' for Hausa, 'yo' for Yoruba
            
        Returns:
            Translated/equivalent text
        """
        words = self.extract_words(text)
        translated_words = []
        
        for word in words:
            if word in self.lexicon and lang in self.lexicon[word]:
                translated_words.append(self.lexicon[word][lang])
            else:
                translated_words.append(word)  # Keep original
        
        return " ".join(translated_words)
    
    # ========================================================================
    # COMBINED FEATURE EXTRACTION
    # ========================================================================
    
    def extract_all_features(
        self, 
        text1: str, 
        text2: str,
        include_multilingual: bool = True
    ) -> Dict[str, float]:
        """
        Extract all linguistic features for a pair of texts
        
        Args:
            text1: First trademark text
            text2: Second trademark text
            include_multilingual: Whether to compute multilingual semantic features
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Synonym/Antonym features
        syn_features = self.compute_synonym_overlap(text1, text2)
        features.update(syn_features)
        
        ant_features = self.compute_antonym_presence(text1, text2)
        features.update(ant_features)
        
        # Phonetic features
        phon_features = self.compute_phonetic_features(text1, text2)
        features.update(phon_features)
        
        # Visual features
        vis_features = self.compute_visual_features(text1, text2)
        features.update(vis_features)
        
        # Semantic similarity (English)
        features['semantic_similarity_en'] = self.compute_semantic_similarity(text1, text2)
        
        # Multilingual semantic features
        if include_multilingual:
            # Hausa equivalents
            text1_ha = self.get_local_equivalents(text1, 'ha')
            text2_ha = self.get_local_equivalents(text2, 'ha')
            features['semantic_similarity_ha'] = self.compute_semantic_similarity(text1_ha, text2_ha)
            
            # Yoruba equivalents
            text1_yo = self.get_local_equivalents(text1, 'yo')
            text2_yo = self.get_local_equivalents(text2, 'yo')
            features['semantic_similarity_yo'] = self.compute_semantic_similarity(text1_yo, text2_yo)
        
        return features
    
    def extract_features_as_vector(
        self, 
        text1: str, 
        text2: str,
        include_multilingual: bool = True
    ) -> np.ndarray:
        """
        Extract features as numpy vector for ML models
        
        Returns:
            Feature vector as numpy array
        """
        features_dict = self.extract_all_features(text1, text2, include_multilingual)
        
        # Define feature order
        feature_names = [
            'synonym_overlap_score', 'synonym_exact_matches', 'synonym_expanded_intersection',
            'antonym_flag', 'antonym_count',
            'soundex_match', 'metaphone_match',
            'levenshtein_normalized', 'jaro_winkler', 'hamming_normalized', 'length_diff',
            'semantic_similarity_en'
        ]
        
        if include_multilingual:
            feature_names.extend(['semantic_similarity_ha', 'semantic_similarity_yo'])
        
        # Build vector
        feature_vector = np.array([features_dict.get(name, 0.0) for name in feature_names])
        
        return feature_vector
