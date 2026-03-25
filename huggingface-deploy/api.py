"""
Trademark Similarity Detection and Registration Decision System
Using Hybrid SVM-CNN Model

Backend API - Compares submitted trademarks against existing database
and provides approve/reject registration decisions.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# TensorFlow and model imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler

# Feature extraction
import jellyfish
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

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

class RegistrationRequest(BaseModel):
    """Request to check trademark registration eligibility"""
    trademarks: List[str] = Field(
        ...,
        description="List of 1-3 trademark names to check for registration"
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for rejection (default 0.7)"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of most similar existing trademarks to return"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trademarks": ["SuperCoffee", "TechSmart", "GoldBrand"],
                "threshold": 0.7,
                "top_k": 5
            }
        }
    )

    @field_validator('trademarks')
    @classmethod
    def clean_trademarks(cls, v: List[str]) -> List[str]:
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError('At least one non-empty trademark name is required')
        if len(cleaned) > 3:
            raise ValueError('Maximum 3 trademarks allowed per request')
        return cleaned


class TrademarkMatch(BaseModel):
    """A matching existing trademark"""
    trademark: str
    similarity_score: float
    details: Optional[Dict[str, Any]] = None


class TrademarkDecision(BaseModel):
    """Registration decision for a single trademark"""
    trademark: str
    decision: str  # "APPROVED" or "REJECTED"
    max_similarity: float
    closest_match: Optional[TrademarkMatch] = None
    top_matches: List[TrademarkMatch] = []
    reason: str


class RegistrationResponse(BaseModel):
    """Response for trademark registration check"""
    threshold: float
    database_size: int
    results: List[TrademarkDecision]
    summary: Dict[str, int]


class SimilarityRequest(BaseModel):
    """Request model for direct similarity check"""
    mark1: str = Field(..., min_length=1, max_length=200)
    mark2: str = Field(..., min_length=1, max_length=200)
    include_details: bool = Field(default=False)

    @field_validator('mark1', 'mark2')
    @classmethod
    def clean_text(cls, v: str) -> str:
        return v.strip()


class SimilarityResponse(BaseModel):
    """Response model for similarity check"""
    label: int
    label_text: str
    probability: float
    confidence: float
    risk_level: str
    recommendation: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(protected_namespaces=())

    status: str
    timestamp: str
    models_loaded: bool
    database_loaded: bool
    database_size: int
    model_info: Dict[str, Any]


# ============================================================================
# MODEL LOADER
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
        self.config = {'max_sequence_length': 50, 'embedding_dim': 128}

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
        tokenizer_path = self.models_dir / "cnn_encoder_tokenizer.pkl"
        model_paths = [
            self.models_dir / "best_cnn_model.h5",
            self.models_dir / "cnn_encoder.keras"
        ]

        loaded = False
        for cnn_path in model_paths:
            if not cnn_path.exists():
                continue
            try:
                self.cnn_model = keras.models.load_model(
                    str(cnn_path), safe_mode=False, compile=False
                )
                try:
                    self.cnn_encoder = keras.models.Model(
                        inputs=self.cnn_model.get_layer('cnn_encoder').input,
                        outputs=self.cnn_model.get_layer('cnn_encoder').output
                    )
                except Exception:
                    self.cnn_encoder = self.cnn_model
                loaded = True
                logger.info(f"   Loaded CNN model from: {cnn_path.name}")
                break
            except Exception as e:
                logger.warning(f"   Failed to load {cnn_path.name}: {e}")

        if not loaded:
            raise FileNotFoundError("Could not load any CNN model file")

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
            if isinstance(data, dict):
                self.svm_model = (data.get('model') or data.get('svm_model') or
                                  data.get('svm') or data.get('classifier'))
                self.scaler = (data.get('scaler') or data.get('scaler_model') or
                               data.get('feature_scaler'))
                if self.svm_model is None:
                    raise ValueError(f"Could not find SVM model. Keys: {list(data.keys())}")
            else:
                self.svm_model = data
                self.scaler = None

    def _load_semantic_model(self):
        """Load semantic similarity model"""
        self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def encode_text(self, text: str) -> np.ndarray:
        """Encode single text using CNN encoder"""
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.config['max_sequence_length'], padding='post')
        return self.cnn_encoder.predict(padded, verbose=0)[0]

    def encode_texts_batch(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Encode multiple texts using CNN encoder in batches"""
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=self.config['max_sequence_length'], padding='post')
        return self.cnn_encoder.predict(padded, verbose=0, batch_size=batch_size)

    def extract_linguistic_features(self, mark1: str, mark2: str) -> np.ndarray:
        """Extract 10 linguistic features for a trademark pair"""
        m1, m2 = mark1.lower(), mark2.lower()

        levenshtein_dist = jellyfish.levenshtein_distance(m1, m2)
        jaro_winkler = jellyfish.jaro_winkler_similarity(m1, m2)

        try:
            soundex_match = 1 if jellyfish.soundex(m1) == jellyfish.soundex(m2) else 0
        except Exception:
            soundex_match = 0
        try:
            metaphone_match = 1 if jellyfish.metaphone(m1) == jellyfish.metaphone(m2) else 0
        except Exception:
            metaphone_match = 0

        emb1 = self.semantic_model.encode([mark1], convert_to_numpy=True)
        emb2 = self.semantic_model.encode([mark2], convert_to_numpy=True)
        semantic_sim = float(sklearn_cosine_similarity(emb1, emb2)[0][0])

        len_diff = abs(len(m1) - len(m2))

        return np.array([
            levenshtein_dist, jaro_winkler,
            soundex_match, metaphone_match,
            semantic_sim, semantic_sim, semantic_sim,
            len_diff, len(m1), len(m2)
        ])

    def extract_linguistic_features_detail(self, mark1: str, mark2: str) -> Dict[str, Any]:
        """Extract linguistic features as a detailed dict"""
        m1, m2 = mark1.lower(), mark2.lower()

        levenshtein_dist = jellyfish.levenshtein_distance(m1, m2)
        jaro_winkler = jellyfish.jaro_winkler_similarity(m1, m2)
        try:
            soundex_match = jellyfish.soundex(m1) == jellyfish.soundex(m2)
        except Exception:
            soundex_match = False
        try:
            metaphone_match = jellyfish.metaphone(m1) == jellyfish.metaphone(m2)
        except Exception:
            metaphone_match = False

        emb1 = self.semantic_model.encode([mark1], convert_to_numpy=True)
        emb2 = self.semantic_model.encode([mark2], convert_to_numpy=True)
        semantic_sim = float(sklearn_cosine_similarity(emb1, emb2)[0][0])

        return {
            "visual_features": {
                "levenshtein_distance": int(levenshtein_dist),
                "jaro_winkler_similarity": round(jaro_winkler, 4)
            },
            "phonetic_features": {
                "soundex_match": bool(soundex_match),
                "metaphone_match": bool(metaphone_match)
            },
            "semantic_features": {
                "similarity_en": round(semantic_sim, 4),
                "similarity_ha": round(semantic_sim, 4),
                "similarity_yo": round(semantic_sim, 4)
            },
            "length_features": {
                "length_difference": abs(len(m1) - len(m2)),
                "mark1_length": len(m1),
                "mark2_length": len(m2)
            }
        }

    def predict(self, mark1: str, mark2: str, return_details: bool = False):
        """Predict similarity between two trademarks"""
        emb1 = self.encode_text(mark1)
        emb2 = self.encode_text(mark2)
        ling_features = self.extract_linguistic_features(mark1, mark2)

        hybrid_features = np.concatenate([emb1, emb2, ling_features]).reshape(1, -1)
        scaled_features = self.scaler.transform(hybrid_features)
        label = int(self.svm_model.predict(scaled_features)[0])
        probability = float(self.svm_model.predict_proba(scaled_features)[0][1])

        details = None
        if return_details:
            details = self.extract_linguistic_features_detail(mark1, mark2)

        return label, probability, details


# ============================================================================
# TRADEMARK DATABASE
# ============================================================================

class TrademarkDatabase:
    """Manages the existing trademark database for registration checks"""

    def __init__(self):
        self.trademarks: List[Dict[str, str]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.loaded = False

    def load(self, csv_path: str, model_loader: HybridModelLoader):
        """Load trademarks from CSV and pre-compute CNN embeddings"""
        logger.info(f"Loading trademark database from {csv_path}...")

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['wordmark'])
        df = df[df['wordmark'].str.strip() != '']

        self.trademarks = df.to_dict('records')
        logger.info(f"   Loaded {len(self.trademarks)} registered trademarks")

        logger.info("   Pre-computing CNN embeddings for database...")
        texts = [tm['wordmark'] for tm in self.trademarks]
        self.embeddings = model_loader.encode_texts_batch(texts)
        logger.info(f"   Embeddings shape: {self.embeddings.shape}")

        self.loaded = True
        logger.info("[SUCCESS] Trademark database ready")

    def find_candidates(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Dict]:
        """Find top-K most similar trademarks using cosine similarity on CNN embeddings"""
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        db_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(db_norms, query_norm)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        candidates = []
        for idx in top_indices:
            candidates.append({
                'index': int(idx),
                'trademark': self.trademarks[idx]['wordmark'],
                'embedding_similarity': float(similarities[idx]),
            })
        return candidates


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

model_loader: Optional[HybridModelLoader] = None
trademark_db: Optional[TrademarkDatabase] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global model_loader, trademark_db

    try:
        logger.info("=" * 80)
        logger.info("[STARTUP] Trademark Registration Decision System")
        logger.info("=" * 80)

        model_loader = HybridModelLoader(models_dir="models")
        model_loader.load_models()

        trademark_db = TrademarkDatabase()
        db_path = Path("data/trademark_database.csv")
        if db_path.exists():
            trademark_db.load(str(db_path), model_loader)
        else:
            logger.warning(f"[WARNING] Database not found at {db_path}")

        logger.info("=" * 80)
        logger.info("[SUCCESS] Server ready to accept requests")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"[ERROR] Startup failed: {e}")
        raise

    yield
    logger.info("[SHUTDOWN] Stopping server...")


# Initialize FastAPI
app = FastAPI(
    title="Trademark Registration Decision System",
    description=(
        "AI-powered trademark similarity detection and registration decision system "
        "using a Hybrid SVM-CNN model. Compares submitted trademarks against a database "
        "of existing registered trademarks and provides approve/reject decisions."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "name": "Trademark Registration Decision System",
        "version": "2.0.0",
        "description": "AI-powered trademark registration decision system using Hybrid SVM-CNN",
        "model": "Hybrid CNN+SVM",
        "languages": ["English", "Hausa", "Yoruba"],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "check_registration": "/check-registration",
            "similarity_check": "/similarity-check"
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
    db_loaded = trademark_db is not None and trademark_db.loaded
    db_size = len(trademark_db.trademarks) if db_loaded else 0

    model_info = {}
    if models_loaded:
        model_info = {
            "cnn_loaded": True,
            "svm_loaded": True,
            "semantic_loaded": True,
            "vocab_size": len(model_loader.tokenizer.word_index) if model_loader.tokenizer else 0
        }

    return {
        "status": "healthy" if (models_loaded and db_loaded) else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_loaded,
        "database_loaded": db_loaded,
        "database_size": db_size,
        "model_info": model_info
    }


@app.post("/check-registration", response_model=RegistrationResponse, tags=["Registration"])
async def check_registration(request: RegistrationRequest):
    """
    Check if submitted trademarks can be registered.

    Compares each submitted trademark against the database of existing
    registered trademarks. Returns APPROVED or REJECTED for each based
    on the similarity threshold.

    - Submit 1-3 trademark names
    - System finds the most similar existing trademarks for each
    - If max similarity >= threshold: REJECTED
    - If max similarity < threshold: APPROVED
    """
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if trademark_db is None or not trademark_db.loaded:
        raise HTTPException(status_code=503, detail="Trademark database not loaded")

    results = []
    approved_count = 0
    rejected_count = 0

    for tm_name in request.trademarks:
        try:
            decision = _evaluate_trademark(tm_name, request.threshold, request.top_k)
            results.append(decision)
            if decision.decision == "APPROVED":
                approved_count += 1
            else:
                rejected_count += 1
        except Exception as e:
            logger.error(f"Error evaluating '{tm_name}': {e}")
            results.append(TrademarkDecision(
                trademark=tm_name,
                decision="ERROR",
                max_similarity=0.0,
                closest_match=None,
                top_matches=[],
                reason=f"Error during evaluation: {str(e)}"
            ))

    return RegistrationResponse(
        threshold=request.threshold,
        database_size=len(trademark_db.trademarks),
        results=results,
        summary={
            "total_submitted": len(request.trademarks),
            "approved": approved_count,
            "rejected": rejected_count
        }
    )


def _evaluate_trademark(tm_name: str, threshold: float, top_k: int) -> TrademarkDecision:
    """Evaluate a single trademark against the database"""
    # Stage 1: Fast retrieval using CNN embeddings
    query_embedding = model_loader.encode_text(tm_name)
    candidates = trademark_db.find_candidates(query_embedding, top_k=max(top_k, 20))

    # Stage 2: Full hybrid scoring with CNN+SVM on top candidates
    scored_matches = []
    for candidate in candidates:
        existing_mark = candidate['trademark']
        try:
            _, probability, details = model_loader.predict(
                tm_name, existing_mark, return_details=True
            )
            scored_matches.append({
                'trademark': existing_mark,
                'similarity_score': round(probability, 4),
                'details': details
            })
        except Exception as e:
            logger.warning(f"   Failed to score '{tm_name}' vs '{existing_mark}': {e}")

    # Sort by similarity score descending
    scored_matches.sort(key=lambda x: x['similarity_score'], reverse=True)

    top_matches = [
        TrademarkMatch(
            trademark=m['trademark'],
            similarity_score=m['similarity_score'],
            details=None
        )
        for m in scored_matches[:top_k]
    ]

    max_similarity = scored_matches[0]['similarity_score'] if scored_matches else 0.0
    closest = scored_matches[0] if scored_matches else None

    if max_similarity >= threshold:
        decision = "REJECTED"
        reason = (
            f"Too similar to existing trademark '{closest['trademark']}' "
            f"({max_similarity * 100:.1f}% similarity exceeds {threshold * 100:.0f}% threshold)"
        )
    else:
        decision = "APPROVED"
        if closest:
            reason = (
                f"Sufficiently unique. Closest match is '{closest['trademark']}' "
                f"at {max_similarity * 100:.1f}% similarity (below {threshold * 100:.0f}% threshold)"
            )
        else:
            reason = "No similar trademarks found in the database"

    closest_match = None
    if closest:
        closest_match = TrademarkMatch(
            trademark=closest['trademark'],
            similarity_score=closest['similarity_score'],
            details=closest['details']
        )

    return TrademarkDecision(
        trademark=tm_name,
        decision=decision,
        max_similarity=max_similarity,
        closest_match=closest_match,
        top_matches=top_matches,
        reason=reason
    )


# ============================================================================
# LEGACY ENDPOINTS (backward compatibility)
# ============================================================================

@app.post("/similarity-check", response_model=SimilarityResponse, tags=["Similarity"])
async def check_similarity(request: SimilarityRequest):
    """Check similarity between two trademarks (direct comparison)"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        label, probability, details = model_loader.predict(
            request.mark1, request.mark2, return_details=request.include_details
        )

        if probability >= 0.7:
            risk_level = "HIGH"
            recommendation = "High risk of confusion - Consider alternative trademark"
        elif probability >= 0.4:
            risk_level = "MEDIUM"
            recommendation = "Moderate risk - Further analysis recommended"
        else:
            risk_level = "LOW"
            recommendation = "Low risk - Likely acceptable"

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
    """Check similarity using GET method"""
    request = SimilarityRequest(mark1=mark1, mark2=mark2, include_details=include_details)
    return await check_similarity(request)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=False, log_level="info")
