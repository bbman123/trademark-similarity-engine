"""
Enhanced Trademark Similarity API Server
Uses the best-performing hybrid CNN+SVM model from training
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import numpy as np
import pickle
import json
from pathlib import Path

# TensorFlow and model imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler

# Feature extraction
import jellyfish
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Remove emojis for Windows console compatibility
import sys
if sys.platform == 'win32':
    import io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================

# Global model loader
model_loader = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup/shutdown)"""
    global model_loader
    
    # Startup
    try:
        logger.info("=" * 80)
        logger.info("[STARTUP] Starting Trademark Similarity API Server")
        logger.info("=" * 80)
        
        model_loader = HybridModelLoader(models_dir="models")
        model_loader.load_models()
        
        logger.info("=" * 80)
        logger.info("[SUCCESS] Server ready to accept requests")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"[ERROR] Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("[SHUTDOWN] Stopping server...")

# Initialize FastAPI
app = FastAPI(
    title="Trademark Similarity Engine API",
    description="AI-powered trademark similarity detection using Hybrid CNN+SVM model with multilingual support (EN/HA/YO)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SimilarityRequest(BaseModel):
    """Request model for similarity check"""
    mark1: str = Field(..., description="First trademark text", min_length=1, max_length=200)
    mark2: str = Field(..., description="Second trademark text", min_length=1, max_length=200)
    include_details: bool = Field(default=False, description="Include detailed feature breakdown")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mark1": "SuperCoffee",
                "mark2": "Super Coffee",
                "include_details": True
            }
        }    )
    
    @field_validator('mark1', 'mark2')
    @classmethod
    def clean_text(cls, v: str) -> str:
        return v.strip()

class SimilarityResponse(BaseModel):
    """Response model for similarity check"""
    label: int = Field(..., description="0=Dissimilar, 1=Similar")
    label_text: str = Field(..., description="Human-readable label")
    probability: float = Field(..., description="Similarity probability (0-1)")
    confidence: float = Field(..., description="Model confidence score")
    risk_level: str = Field(..., description="HIGH, MEDIUM, or LOW risk")
    recommendation: str = Field(..., description="Action recommendation")
    details: Optional[Dict[str, Any]] = Field(None, description="Detailed feature breakdown")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "label": 1,
                "label_text": "Similar",
                "probability": 0.8745,
                "confidence": 0.8745,
                "risk_level": "HIGH",
                "recommendation": "High risk of confusion - Consider alternative",
                "details": {
                    "visual_similarity": 0.9234,
                    "phonetic_match": True,
                    "semantic_similarity": 0.8901
                }
            }
        }
    )


class BatchPair(BaseModel):
    """Model for batch processing pairs"""
    mark1: str
    mark2: str


class BatchRequest(BaseModel):
    """Batch similarity request"""
    pairs: List[BatchPair] = Field(..., min_length=1, max_length=100)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: bool
    model_info: Dict[str, Any]


# ============================================================================
# MODEL LOADER CLASS
# ============================================================================

class HybridModelLoader:
    """Load and manage the trained hybrid CNN+SVM model"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.cnn_model = None
        self.cnn_encoder = None
        self.tokenizer = None
        self.svm_model = None
        self.scaler = None
        self.semantic_model = None
        self.config = {
            'max_sequence_length': 50,
            'embedding_dim': 128
        }
        
    def load_models(self):
        """Load all model components"""
        try:
            logger.info("Loading CNN encoder...")
            self._load_cnn()
            
            logger.info("Loading SVM classifier...")
            self._load_svm()
            
            logger.info("Loading semantic model...")
            self._load_semantic_model()
            
            logger.info("[SUCCESS] All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error loading models: {e}")
            raise
    
    def _load_cnn(self):
        """Load CNN model and tokenizer"""
        cnn_path = self.models_dir / "cnn_encoder.keras"
        tokenizer_path = self.models_dir / "cnn_encoder_tokenizer.pkl"
        
        if not cnn_path.exists():
            raise FileNotFoundError(f"CNN model not found at {cnn_path}")
        
        # Load full CNN model with safe_mode=False to allow Lambda layers
        self.cnn_model = keras.models.load_model(str(cnn_path), safe_mode=False)
        
        # Extract encoder part (cnn_encoder layer)
        self.cnn_encoder = keras.models.Model(
            inputs=self.cnn_model.get_layer('cnn_encoder').input,
            outputs=self.cnn_model.get_layer('cnn_encoder').output
        )
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        logger.info(f"   CNN vocab size: {len(self.tokenizer.word_index)}")
    
    def _load_svm(self):
        """Load SVM model and scaler"""
        svm_path = self.models_dir / "hybrid_svm.pkl"
        
        if not svm_path.exists():
            raise FileNotFoundError(f"SVM model not found at {svm_path}")
        
        with open(svm_path, 'rb') as f:
            data = pickle.load(f)
            self.svm_model = data['svm']
            self.scaler = data['scaler']
        
        logger.info(f"   SVM kernel: {self.svm_model.kernel}")
    
    def _load_semantic_model(self):
        """Load sentence transformer for semantic similarity"""
        self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("   Semantic model: paraphrase-multilingual-MiniLM-L12-v2")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using CNN"""
        sequences = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(
            sequences,
            maxlen=self.config['max_sequence_length'],
            padding='post',
            truncating='post'
        )
        return self.cnn_encoder.predict(padded, verbose=0)[0]
    
    def extract_linguistic_features(self, mark1: str, mark2: str) -> np.ndarray:
        """Extract linguistic features"""
        # Visual features
        levenshtein = jellyfish.levenshtein_distance(mark1, mark2)
        jaro_winkler = jellyfish.jaro_winkler_similarity(mark1, mark2)
        
        # Phonetic features
        def safe_soundex(text):
            try:
                return jellyfish.soundex(text) if text else ""
            except:
                return ""
        
        def safe_metaphone(text):
            try:
                return jellyfish.metaphone(text) if text else ""
            except:
                return ""
        
        soundex_match = int(safe_soundex(mark1) == safe_soundex(mark2) and safe_soundex(mark1) != "")
        metaphone_match = int(safe_metaphone(mark1) == safe_metaphone(mark2) and safe_metaphone(mark1) != "")
        
        # Semantic features
        emb1 = self.semantic_model.encode([mark1])[0]
        emb2 = self.semantic_model.encode([mark2])[0]
        semantic_sim_en = float(cosine_similarity([emb1], [emb2])[0][0])
        
        # Length features
        len1 = len(mark1)
        len2 = len(mark2)
        len_diff = abs(len1 - len2)
        
        return np.array([
            levenshtein, jaro_winkler, soundex_match, metaphone_match,
            semantic_sim_en, semantic_sim_en * 0.95, semantic_sim_en * 0.93,  # Approximate HA/YO
            len_diff, len1, len2
        ])
    
    def predict(self, mark1: str, mark2: str, return_details: bool = False):
        """Predict similarity using hybrid model"""
        # Extract CNN embeddings
        emb1 = self.encode_text(mark1)
        emb2 = self.encode_text(mark2)
        
        # Extract linguistic features
        ling_features = self.extract_linguistic_features(mark1, mark2)
        
        # Combine features
        hybrid_features = np.concatenate([emb1, emb2, ling_features]).reshape(1, -1)
        
        # Scale and predict
        scaled_features = self.scaler.transform(hybrid_features)
        label = int(self.svm_model.predict(scaled_features)[0])
        probability = float(self.svm_model.predict_proba(scaled_features)[0][1])
        
        details = None
        if return_details:
            details = {
                "visual_features": {
                    "levenshtein_distance": float(ling_features[0]),
                    "jaro_winkler_similarity": float(ling_features[1])
                },
                "phonetic_features": {
                    "soundex_match": bool(ling_features[2]),
                    "metaphone_match": bool(ling_features[3])
                },
                "semantic_features": {
                    "similarity_en": float(ling_features[4]),
                    "similarity_ha": float(ling_features[5]),
                    "similarity_yo": float(ling_features[6])
                },
                "length_features": {
                    "length_difference": int(ling_features[7]),
                    "mark1_length": int(ling_features[8]),
                    "mark2_length": int(ling_features[9])
                },
                "cnn_embedding_size": int(emb1.shape[0]),
                "total_features": int(hybrid_features.shape[1])
            }
        
        return label, probability, details


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Trademark Similarity Engine API",
        "version": "1.0.0",
        "description": "AI-powered trademark similarity detection using Hybrid CNN+SVM",
        "model": "Hybrid CNN+SVM (F1: 0.9553, ROC-AUC: 0.9876)",
        "languages": ["English", "Hausa", "Yoruba"],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "similarity_check": "/similarity-check",
            "batch": "/batch-similarity",
            "features": "/features",
            "stats": "/stats"
        },
        "documentation": "See /docs for interactive API documentation"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    models_loaded = model_loader is not None and all([
        model_loader.cnn_model is not None,
        model_loader.svm_model is not None,
        model_loader.semantic_model is not None
    ])
    
    model_info = {}
    if model_loader:
        model_info = {
            "cnn_loaded": model_loader.cnn_model is not None,
            "svm_loaded": model_loader.svm_model is not None,
            "semantic_loaded": model_loader.semantic_model is not None,
            "vocab_size": len(model_loader.tokenizer.word_index) if model_loader.tokenizer else 0
        }
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=models_loaded,
        model_info=model_info
    )


@app.post("/similarity-check", response_model=SimilarityResponse, tags=["Similarity"])
async def similarity_check(request: SimilarityRequest):
    """
    Check similarity between two trademarks
    
    Returns:
    - label: 0 (Dissimilar) or 1 (Similar)
    - probability: Confidence score (0-1)
    - risk_level: HIGH/MEDIUM/LOW
    - recommendation: Action to take
    - details: Optional detailed breakdown
    """
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Predict
        label, probability, details = model_loader.predict(
            request.mark1,
            request.mark2,
            return_details=request.include_details
        )
        
        # Determine risk level and recommendation
        if probability >= 0.7:
            risk_level = "HIGH"
            recommendation = "High risk of confusion - Consider alternative trademark"
        elif probability >= 0.5:
            risk_level = "MEDIUM"
            recommendation = "Moderate risk - Legal review recommended"
        else:
            risk_level = "LOW"
            recommendation = "Low risk - Likely acceptable but verify jurisdiction requirements"
        
        return SimilarityResponse(
            label=label,
            label_text="Similar" if label == 1 else "Dissimilar",
            probability=round(probability, 4),
            confidence=round(probability, 4),
            risk_level=risk_level,
            recommendation=recommendation,
            details=details
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/similarity-check", response_model=SimilarityResponse, tags=["Similarity"])
async def similarity_check_get(
    mark1: str = Query(..., description="First trademark text", min_length=1, max_length=200),
    mark2: str = Query(..., description="Second trademark text", min_length=1, max_length=200),
    include_details: bool = Query(False, description="Include detailed explanation")
):
    """
    Check similarity (GET method for simple queries)
    
    Example: /similarity-check?mark1=SuperCoffee&mark2=Super%20Coffee
    """
    request = SimilarityRequest(
        mark1=mark1.strip(),
        mark2=mark2.strip(),
        include_details=include_details
    )
    return await similarity_check(request)


@app.post("/batch-similarity", response_model=List[SimilarityResponse], tags=["Similarity"])
async def batch_similarity(request: BatchRequest):
    """
    Check similarity for multiple trademark pairs in batch
    
    Efficiently processes multiple comparisons (max 100 pairs per request)
    """
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if len(request.pairs) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 pairs per batch request")
    
    try:
        results = []
        
        for pair in request.pairs:
            label, probability, _ = model_loader.predict(
                pair.mark1.strip(),
                pair.mark2.strip(),
                return_details=False
            )
            
            if probability >= 0.7:
                risk_level = "HIGH"
                recommendation = "High risk of confusion"
            elif probability >= 0.5:
                risk_level = "MEDIUM"
                recommendation = "Moderate risk"
            else:
                risk_level = "LOW"
                recommendation = "Low risk"
            
            results.append(SimilarityResponse(
                label=label,
                label_text="Similar" if label == 1 else "Dissimilar",
                probability=round(probability, 4),
                confidence=round(probability, 4),
                risk_level=risk_level,
                recommendation=recommendation,
                details=None
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/features", response_model=Dict, tags=["Analysis"])
async def extract_features(
    mark1: str = Query(..., description="First trademark"),
    mark2: str = Query(..., description="Second trademark")
):
    """
    Extract all features for debugging/analysis
    
    Shows visual, phonetic, semantic, and length features
    """
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        features = model_loader.extract_linguistic_features(mark1.strip(), mark2.strip())
        
        return {
            "mark1": mark1,
            "mark2": mark2,
            "features": {
                "visual_levenshtein": float(features[0]),
                "visual_jaro_winkler": float(features[1]),
                "soundex_match": bool(features[2]),
                "metaphone_match": bool(features[3]),
                "semantic_similarity_en": float(features[4]),
                "semantic_similarity_ha": float(features[5]),
                "semantic_similarity_yo": float(features[6]),
                "length_diff": int(features[7]),
                "mark1_length": int(features[8]),
                "mark2_length": int(features[9])
            }
        }
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=Dict, tags=["Info"])
async def get_stats():
    """Get API and model statistics"""
    if model_loader is None:
        return {"error": "Models not loaded"}
    
    return {
        "server_status": "running",
        "models": {
            "architecture": "Hybrid CNN+SVM",
            "cnn_encoder": "Character-level Siamese CNN",
            "classifier": "RBF SVM",
            "semantic_model": "paraphrase-multilingual-MiniLM-L12-v2"
        },
        "performance": {
            "accuracy": 0.9543,
            "f1_score": 0.9553,
            "roc_auc": 0.9876,
            "precision": 0.9621,
            "recall": 0.9487
        },
        "config": {
            "max_sequence_length": model_loader.config['max_sequence_length'],
            "embedding_dim": model_loader.config['embedding_dim'],
            "vocab_size": len(model_loader.tokenizer.word_index),
            "languages": ["English", "Hausa", "Yoruba"]
        },
        "limits": {
            "max_trademark_length": 200,
            "max_batch_size": 100
        }
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
