"""
CNN Encoder for trademark text embedding extraction
"""

import pickle
import numpy as np
from typing import List, Optional
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

from .config import config
from .cache_manager import CacheManager, cached

logger = logging.getLogger(__name__)


class CNNEncoder:
    """Character-level CNN for extracting trademark embeddings"""
    
    def __init__(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None):
        """
        Initialize CNN encoder
        
        Args:
            model_path: Path to trained CNN model (.keras file)
            tokenizer_path: Path to tokenizer pickle file
        """
        self.model_path = model_path or config.CNN_MODEL_PATH
        self.tokenizer_path = tokenizer_path or config.CNN_TOKENIZER_PATH
        self.model = None
        self.encoder = None
        self.tokenizer = None
        self.cache_manager = CacheManager(config.CACHE_DIR / "embeddings")
        
        # Load model and tokenizer
        self._load()
    
    def _load(self):
        """Load trained model and tokenizer"""
        try:
            # Load full model
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"✓ Loaded CNN model from {self.model_path}")
            
            # Extract encoder (cnn_encoder layer)
            try:
                encoder_layer = self.model.get_layer('cnn_encoder')
                self.encoder = keras.models.Model(
                    inputs=encoder_layer.input,
                    outputs=encoder_layer.output
                )
            except ValueError:
                # If model doesn't have nested encoder, use the full model's intermediate output
                logger.warning("Using full model for embeddings")
                self.encoder = self.model
            
            # Load tokenizer
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            logger.info(f"✓ Loaded tokenizer (vocab size: {len(self.tokenizer.word_index)})")
            
        except Exception as e:
            logger.error(f"Failed to load CNN model: {e}")
            raise
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to padded sequences
        
        Args:
            texts: List of trademark texts
            
        Returns:
            Padded sequences array
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=config.MAX_SEQUENCE_LENGTH,
            padding='post',
            truncating='post'
        )
        return padded
    
    def get_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Extract CNN embeddings for texts
        
        Args:
            texts: List of trademark texts
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embeddings array of shape (len(texts), embedding_dim)
        """
        if use_cache and config.CACHE_ENABLED:
            # Try to get from cache
            cache_key = self.cache_manager._get_cache_key(*texts)
            cached_emb = self.cache_manager.get(cache_key)
            if cached_emb is not None:
                return cached_emb
        
        # Encode texts
        sequences = self.encode_texts(texts)
        
        # Get embeddings
        embeddings = self.encoder.predict(sequences, verbose=0)
        
        # Cache results
        if use_cache and config.CACHE_ENABLED:
            self.cache_manager.set(cache_key, embeddings)
        
        return embeddings
    
    def get_single_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Extract embedding for a single text
        
        Args:
            text: Single trademark text
            use_cache: Whether to use cached embedding
            
        Returns:
            Embedding vector
        """
        embeddings = self.get_embeddings([text], use_cache=use_cache)
        return embeddings[0]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts using CNN embeddings
        
        Args:
            text1: First trademark text
            text2: Second trademark text
            
        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = self.get_single_embedding(text1)
        emb2 = self.get_single_embedding(text2)
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache_manager.clear()
        logger.info("Cleared embedding cache")
