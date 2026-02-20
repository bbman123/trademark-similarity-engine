"""
Retrieval system for efficient candidate trademark selection
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import pickle
import logging

from .config import config
from .cnn_encoder import CNNEncoder
from .linguistic_features import LinguisticFeatureExtractor

logger = logging.getLogger(__name__)


class TrademarkRetriever:
    """
    Efficient retrieval of candidate trademarks using approximate nearest neighbors
    and lexical filtering
    """
    
    def __init__(
        self,
        cnn_encoder: Optional[CNNEncoder] = None,
        linguistic_extractor: Optional[LinguisticFeatureExtractor] = None
    ):
        """
        Initialize retriever
        
        Args:
            cnn_encoder: CNN encoder for embeddings
            linguistic_extractor: Linguistic feature extractor
        """
        self.cnn_encoder = cnn_encoder or CNNEncoder()
        self.linguistic_extractor = linguistic_extractor or LinguisticFeatureExtractor()
        
        # Database of indexed trademarks
        self.trademark_db = []  # List of {text, metadata}
        self.embeddings = None  # Embedding matrix
        self.phonetic_index = {}  # Soundex/metaphone buckets
        
        logger.info("✓ Trademark retriever initialized")
    
    def index_trademarks(self, trademarks: List[Dict]):
        """
        Index a list of trademarks for fast retrieval
        
        Args:
            trademarks: List of dictionaries with at least {'text': str, ...}
        """
        logger.info(f"Indexing {len(trademarks)} trademarks...")
        
        self.trademark_db = trademarks
        
        # Extract embeddings for all trademarks
        texts = [tm['text'] for tm in trademarks]
        self.embeddings = self.cnn_encoder.get_embeddings(texts, use_cache=False)
        
        # Build phonetic index
        self._build_phonetic_index(texts)
        
        logger.info("✓ Indexing complete")
    
    def _build_phonetic_index(self, texts: List[str]):
        """Build phonetic buckets for fast phonetic matching"""
        self.phonetic_index = {
            'soundex': {},
            'metaphone': {}
        }
        
        for idx, text in enumerate(texts):
            # Soundex
            try:
                import jellyfish
                soundex = jellyfish.soundex(text)
                if soundex not in self.phonetic_index['soundex']:
                    self.phonetic_index['soundex'][soundex] = []
                self.phonetic_index['soundex'][soundex].append(idx)
            except:
                pass
            
            # Metaphone
            try:
                metaphone = jellyfish.metaphone(text)
                if metaphone not in self.phonetic_index['metaphone']:
                    self.phonetic_index['metaphone'][metaphone] = []
                self.phonetic_index['metaphone'][metaphone].append(idx)
            except:
                pass
    
    def retrieve_by_embedding(
        self,
        query_text: str,
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve candidates using embedding-based nearest neighbor search
        
        Args:
            query_text: Query trademark text
            top_k: Number of candidates to retrieve
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        if top_k is None:
            top_k = config.TOP_K_CANDIDATES
        
        # Get query embedding
        query_emb = self.cnn_encoder.get_single_embedding(query_text)
        
        # Compute cosine similarities with all indexed embeddings
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return with scores
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def retrieve_by_phonetic(
        self,
        query_text: str,
        method: str = 'soundex'
    ) -> List[int]:
        """
        Retrieve candidates using phonetic matching
        
        Args:
            query_text: Query trademark text
            method: 'soundex' or 'metaphone'
            
        Returns:
            List of matching trademark indices
        """
        import jellyfish
        
        try:
            if method == 'soundex':
                code = jellyfish.soundex(query_text)
                return self.phonetic_index['soundex'].get(code, [])
            elif method == 'metaphone':
                code = jellyfish.metaphone(query_text)
                return self.phonetic_index['metaphone'].get(code, [])
        except:
            return []
    
    def retrieve_by_ngrams(
        self,
        query_text: str,
        n: int = 3,
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve candidates using character n-gram overlap
        
        Args:
            query_text: Query trademark text
            n: N-gram size
            top_k: Number of candidates to retrieve
            
        Returns:
            List of (index, overlap_score) tuples
        """
        if top_k is None:
            top_k = config.TOP_K_CANDIDATES
        
        # Generate query n-grams
        query_ngrams = set(self._get_ngrams(query_text.lower(), n))
        
        # Compute overlap with all trademarks
        scores = []
        for idx, tm in enumerate(self.trademark_db):
            tm_ngrams = set(self._get_ngrams(tm['text'].lower(), n))
            
            if not query_ngrams or not tm_ngrams:
                overlap = 0.0
            else:
                intersection = len(query_ngrams & tm_ngrams)
                union = len(query_ngrams | tm_ngrams)
                overlap = intersection / union if union > 0 else 0.0
            
            scores.append((idx, overlap))
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _get_ngrams(self, text: str, n: int) -> List[str]:
        """Generate character n-grams"""
        if len(text) < n:
            return [text]
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    def retrieve_hybrid(
        self,
        query_text: str,
        top_k: int = None,
        use_phonetic: bool = True,
        use_ngrams: bool = True,
        threshold: float = None
    ) -> List[Dict]:
        """
        Hybrid retrieval using multiple strategies
        
        Args:
            query_text: Query trademark text
            top_k: Number of final candidates
            use_phonetic: Whether to include phonetic matches
            use_ngrams: Whether to include n-gram matches
            threshold: Minimum similarity threshold
            
        Returns:
            List of candidate dictionaries with scores
        """
        if top_k is None:
            top_k = config.TOP_K_CANDIDATES
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD
        
        # Collect candidates from different strategies
        candidate_scores = {}  # index -> {'embedding': score, 'phonetic': bool, 'ngram': score}
        
        # 1. Embedding-based retrieval (primary)
        emb_results = self.retrieve_by_embedding(query_text, top_k=top_k * 2)
        for idx, score in emb_results:
            if score >= threshold:
                candidate_scores[idx] = {'embedding': score}
        
        # 2. Phonetic matches
        if use_phonetic:
            phonetic_soundex = set(self.retrieve_by_phonetic(query_text, 'soundex'))
            phonetic_metaphone = set(self.retrieve_by_phonetic(query_text, 'metaphone'))
            phonetic_matches = phonetic_soundex | phonetic_metaphone
            
            for idx in phonetic_matches:
                if idx not in candidate_scores:
                    candidate_scores[idx] = {'embedding': 0.0}
                candidate_scores[idx]['phonetic'] = True
        
        # 3. N-gram matches
        if use_ngrams:
            ngram_results = self.retrieve_by_ngrams(query_text, n=3, top_k=top_k)
            for idx, score in ngram_results:
                if score >= 0.3:  # Minimum n-gram overlap
                    if idx not in candidate_scores:
                        candidate_scores[idx] = {'embedding': 0.0}
                    candidate_scores[idx]['ngram'] = score
        
        # Compute combined scores
        candidates = []
        for idx, scores in candidate_scores.items():
            # Weighted combination
            combined_score = (
                scores.get('embedding', 0.0) * 0.6 +
                (1.0 if scores.get('phonetic', False) else 0.0) * 0.2 +
                scores.get('ngram', 0.0) * 0.2
            )
            
            candidate = {
                'index': idx,
                'text': self.trademark_db[idx]['text'],
                'metadata': self.trademark_db[idx],
                'scores': {
                    'combined': combined_score,
                    'embedding': scores.get('embedding', 0.0),
                    'phonetic_match': scores.get('phonetic', False),
                    'ngram': scores.get('ngram', 0.0)
                }
            }
            candidates.append(candidate)
        
        # Sort by combined score and return top-k
        candidates.sort(key=lambda x: x['scores']['combined'], reverse=True)
        return candidates[:top_k]
    
    def save_index(self, path: str):
        """Save indexed database and embeddings"""
        data = {
            'trademark_db': self.trademark_db,
            'embeddings': self.embeddings,
            'phonetic_index': self.phonetic_index
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"✓ Saved index to {path}")
    
    def load_index(self, path: str):
        """Load indexed database and embeddings"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.trademark_db = data['trademark_db']
        self.embeddings = data['embeddings']
        self.phonetic_index = data['phonetic_index']
        
        logger.info(f"✓ Loaded index from {path} ({len(self.trademark_db)} trademarks)")
