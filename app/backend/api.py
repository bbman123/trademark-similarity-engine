"""
Trademark Similarity Engine API
Standalone deployable version
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
                "include_details": False
            }
        }
    )
    
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
                "recommendation": "High risk of confusion"
            }
        }
    )


class BatchPair(BaseModel):
    """Single pair in batch request"""
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
    """Loads and manages all model components"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.cnn_model = None
        self.cnn_encoder = None
        self.tokenizer = None
        self.svm_model = None
        self.scaler = None
        self.semantic_model = None
        
        # Model configuration
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
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        logger.info(f"   CNN vocab size: {len(self.tokenizer.word_index)}")
    
    def _load_svm(self):
        """Load SVM classifier and scaler"""
        svm_path = self.models_dir / "hybrid_svm.pkl"
        
        if not svm_path.exists():
            raise FileNotFoundError(f"SVM model not found at {svm_path}")
        
        with open(svm_path, 'rb') as f:
            data = pickle.load(f)
            # Handle both dict and direct model formats
            if isinstance(data, dict):
                logger.info(f"   Pickle keys: {list(data.keys())}")
                # Try different possible key names
                self.svm_model = (data.get('model') or data.get('svm_model') or 
                                data.get('svm') or data.get('classifier'))
                self.scaler = (data.get('scaler') or data.get('scaler_model') or 
                             data.get('feature_scaler'))
                
                if self.svm_model is None:
                    raise ValueError(f"Could not find SVM model in pickle file. Available keys: {list(data.keys())}")
            else:
                # If it's not a dict, might be just the model
                self.svm_model = data
                self.scaler = None
        
        if self.svm_model is not None and hasattr(self.svm_model, 'kernel'):
            logger.info(f"   SVM kernel: {self.svm_model.kernel}")
        else:
            logger.info(f"   SVM loaded (type: {type(self.svm_model).__name__})")
    
    def _load_semantic_model(self):
        """Load semantic similarity model"""
        model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self.semantic_model = SentenceTransformer(model_name)
        logger.info(f"   Semantic model: {model_name}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using CNN encoder"""
        if self.tokenizer is None or self.cnn_encoder is None:
            raise RuntimeError("Models not loaded")
        
        # Tokenize and pad
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.config['max_sequence_length'], padding='post')
        
        # Encode
        embedding = self.cnn_encoder.predict(padded, verbose=0)[0]
        return embedding
    
    def extract_linguistic_features(self, mark1: str, mark2: str) -> np.ndarray:
        """Extract linguistic features"""
        m1, m2 = mark1.lower(), mark2.lower()
        
        # Visual similarity
        levenshtein_dist = jellyfish.levenshtein_distance(m1, m2)
        jaro_winkler = jellyfish.jaro_winkler_similarity(m1, m2)
        
        # Phonetic similarity
        soundex_match = 1 if jellyfish.soundex(m1) == jellyfish.soundex(m2) else 0
        metaphone_match = 1 if jellyfish.metaphone(m1) == jellyfish.metaphone(m2) else 0
        
        # Semantic similarity (multilingual)
        emb1 = self.semantic_model.encode([mark1], convert_to_numpy=True)
        emb2 = self.semantic_model.encode([mark2], convert_to_numpy=True)
        semantic_sim_en = float(cosine_similarity(emb1, emb2)[0][0])
        
        # Translate and compute for Hausa/Yoruba (simplified - using same text)
        semantic_sim_ha = semantic_sim_en
        semantic_sim_yo = semantic_sim_en
        
        # Length features
        len_diff = abs(len(m1) - len(m2))
        
        features = np.array([
            levenshtein_dist,
            jaro_winkler,
            soundex_match,
            metaphone_match,
            semantic_sim_en,
            semantic_sim_ha,
            semantic_sim_yo,
            len_diff,
            len(m1),
            len(m2)
        ])
        
        return features
    
    def predict(self, mark1: str, mark2: str, return_details: bool = False):
        """Predict similarity between two trademarks"""
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
                }
            }
        
        return label, probability, details


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

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Trademark Similarity Engine API",
    description="AI-powered trademark similarity detection using Hybrid CNN+SVM model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Trademark Similarity Engine API",
        "version": "1.0.0",
        "description": "AI-powered trademark similarity detection",
        "model": "Hybrid CNN+SVM",
        "languages": ["English", "Hausa", "Yoruba"],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "similarity_check": "/similarity-check",
            "batch": "/batch-similarity"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    models_loaded = model_loader is not None and all([
        model_loader.cnn_encoder is not None,
        model_loader.svm_model is not None,
        model_loader.semantic_model is not None
    ])
    
    model_info = {}
    if models_loaded:
        model_info = {
            "cnn_loaded": model_loader.cnn_encoder is not None,
            "svm_loaded": model_loader.svm_model is not None,
            "semantic_loaded": model_loader.semantic_model is not None,
            "vocab_size": len(model_loader.tokenizer.word_index) if model_loader.tokenizer else 0
        }
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_loaded,
        "model_info": model_info
    }


@app.post("/similarity-check", response_model=SimilarityResponse, tags=["Similarity"])
async def check_similarity(request: SimilarityRequest):
    """Check similarity between two trademarks"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Get prediction
        label, probability, details = model_loader.predict(
            request.mark1, 
            request.mark2,
            return_details=request.include_details
        )
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "HIGH"
            recommendation = "High risk of confusion - Consider alternative trademark or seek legal advice"
        elif probability >= 0.4:
            risk_level = "MEDIUM"
            recommendation = "Moderate risk - Further analysis recommended"
        else:
            risk_level = "LOW"
            recommendation = "Low risk - Likely acceptable but verify jurisdiction requirements"
        
        return {
            "label": label,
            "label_text": "Similar" if label == 1 else "Dissimilar",
            "probability": probability,
            "confidence": probability,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "details": details
        }
        
    except Exception as e:
        logger.error(f"Error in similarity check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similarity-check", response_model=SimilarityResponse, tags=["Similarity"])
async def check_similarity_get(
    mark1: str = Query(..., description="First trademark"),
    mark2: str = Query(..., description="Second trademark"),
    include_details: bool = Query(False, description="Include detailed analysis")
):
    """Check similarity using GET method (query parameters)"""
    request = SimilarityRequest(mark1=mark1, mark2=mark2, include_details=include_details)
    return await check_similarity(request)


@app.post("/batch-similarity", tags=["Similarity"])
async def batch_similarity(request: BatchRequest):
    """Check similarity for multiple trademark pairs"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    results = []
    for pair in request.pairs:
        try:
            label, probability, _ = model_loader.predict(pair.mark1, pair.mark2, return_details=False)
            
            if probability >= 0.7:
                risk_level = "HIGH"
            elif probability >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            results.append({
                "mark1": pair.mark1,
                "mark2": pair.mark2,
                "label": label,
                "label_text": "Similar" if label == 1 else "Dissimilar",
                "probability": probability,
                "risk_level": risk_level
            })
        except Exception as e:
            results.append({
                "mark1": pair.mark1,
                "mark2": pair.mark2,
                "error": str(e)
            })
    
    return {
        "total_pairs": len(request.pairs),
        "results": results
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
