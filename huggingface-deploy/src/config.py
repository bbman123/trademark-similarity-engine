"""
Configuration management for Trademark Similarity Engine
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class Config:
    """Central configuration for the similarity engine"""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODEL_DIR: Path = BASE_DIR / "models"
    CACHE_DIR: Path = BASE_DIR / "cache"
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Model files
    CNN_MODEL_PATH: str = "models/cnn_encoder.keras"
    CNN_TOKENIZER_PATH: str = "models/cnn_encoder_tokenizer.pkl"
    SVM_MODEL_PATH: str = "models/hybrid_svm.pkl"
    
    # CNN parameters
    MAX_SEQUENCE_LENGTH: int = 50
    MAX_VOCAB_SIZE: int = 10000
    EMBEDDING_DIM: int = 128
    
    # Retrieval parameters
    TOP_K_CANDIDATES: int = 100
    SIMILARITY_THRESHOLD: float = 0.5
    
    # Caching
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Linguistic features
    LANGUAGES: list = None
    
    def __post_init__(self):
        """Initialize default values"""
        if self.LANGUAGES is None:
            self.LANGUAGES = ['en', 'ha', 'yo']
        
        # Create directories if needed
        self.MODEL_DIR.mkdir(exist_ok=True)
        self.CACHE_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def from_json(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str):
        """Save configuration to JSON file"""
        data = {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# Global config instance
config = Config()
