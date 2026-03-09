"""
FastAPI service for Trademark Similarity Engine
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
from datetime import datetime

from .cnn_encoder import CNNEncoder
from .linguistic_features import LinguisticFeatureExtractor
from .svm_classifier import HybridSimilarityClassifier
from .retrieval import TrademarkRetriever
from .config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Trademark Similarity Engine API",
    description="AI-powered trademark similarity detection with multilingual support (EN/HA/YO)",
    version="1.0.0"
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
    mark1: str = Field(..., description="First trademark text", min_length=1)
    mark2: str = Field(..., description="Second trademark text", min_length=1)
    include_details: bool = Field(False, description="Include detailed explanation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "mark1": "SuperCoffee",
                "mark2": "Super Coffee",
                "include_details": True
            }
        }


class SimilarityResponse(BaseModel):
    """Response model for similarity check"""
    label: int = Field(..., description="0=Dissimilar, 1=Similar")
    label_text: str = Field(..., description="Human-readable label")
    probability: float = Field(..., description="Similarity probability (0-1)")
    risk_level: str = Field(..., description="HIGH, MEDIUM, or LOW risk")
    details: Optional[Dict] = Field(None, description="Detailed feature breakdown")


class RetrievalRequest(BaseModel):
    """Request model for candidate retrieval"""
    query: str = Field(..., description="Query trademark text", min_length=1)
    top_k: int = Field(10, description="Number of candidates to return", ge=1, le=100)
    threshold: float = Field(0.5, description="Minimum similarity threshold", ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "TechSmart",
                "top_k": 5,
                "threshold": 0.6
            }
        }


class RetrievalResponse(BaseModel):
    """Response model for candidate retrieval"""
    query: str
    candidates: List[Dict]
    count: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: bool


# ============================================================================
# GLOBAL MODEL INSTANCES (loaded on startup)
# ============================================================================

cnn_encoder = None
linguistic_extractor = None
hybrid_classifier = None
retriever = None


@app.on_event("startup")
async def load_models():
    """Load all models on startup"""
    global cnn_encoder, linguistic_extractor, hybrid_classifier, retriever
    
    try:
        logger.info("Loading models...")
        
        # Load components
        cnn_encoder = CNNEncoder()
        linguistic_extractor = LinguisticFeatureExtractor()
        hybrid_classifier = HybridSimilarityClassifier(
            cnn_encoder=cnn_encoder,
            linguistic_extractor=linguistic_extractor
        )
        retriever = TrademarkRetriever(
            cnn_encoder=cnn_encoder,
            linguistic_extractor=linguistic_extractor
        )
        
        logger.info("✅ All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Trademark Similarity Engine API",
        "version": "1.0.0",
        "description": "AI-powered trademark similarity detection with multilingual support",
        "endpoints": {
            "similarity_check": "/similarity-check",
            "retrieve_candidates": "/retrieve-candidates",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = all([
        cnn_encoder is not None,
        linguistic_extractor is not None,
        hybrid_classifier is not None,
        retriever is not None
    ])
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=models_loaded
    )


@app.post("/similarity-check", response_model=SimilarityResponse)
async def similarity_check(request: SimilarityRequest):
    """
    Check similarity between two trademarks
    
    Returns prediction with risk level and optional detailed explanation
    """
    if hybrid_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Predict
        label, probability, details = hybrid_classifier.predict(
            request.mark1,
            request.mark2,
            return_details=request.include_details
        )
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "HIGH"
        elif probability >= 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return SimilarityResponse(
            label=label,
            label_text="Similar" if label == 1 else "Dissimilar",
            probability=round(probability, 4),
            risk_level=risk_level,
            details=details if request.include_details else None
        )
        
    except Exception as e:
        logger.error(f"Similarity check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve-candidates", response_model=RetrievalResponse)
async def retrieve_candidates(request: RetrievalRequest):
    """
    Retrieve top-K most similar trademark candidates from indexed database
    
    Note: Requires pre-indexed trademark database. Use retriever.index_trademarks()
    or retriever.load_index() to set up the database.
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not loaded")
    
    if not retriever.trademark_db:
        raise HTTPException(
            status_code=400,
            detail="No trademarks indexed. Please index trademarks first."
        )
    
    try:
        # Retrieve candidates
        candidates = retriever.retrieve_hybrid(
            query_text=request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        return RetrievalResponse(
            query=request.query,
            candidates=candidates,
            count=len(candidates)
        )
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similarity-check", response_model=SimilarityResponse)
async def similarity_check_get(
    mark1: str = Query(..., description="First trademark text"),
    mark2: str = Query(..., description="Second trademark text"),
    include_details: bool = Query(False, description="Include detailed explanation")
):
    """
    Check similarity between two trademarks (GET method for simple queries)
    
    Example: /similarity-check?mark1=SuperCoffee&mark2=Super%20Coffee&include_details=true
    """
    request = SimilarityRequest(
        mark1=mark1,
        mark2=mark2,
        include_details=include_details
    )
    return await similarity_check(request)


@app.get("/features", response_model=Dict)
async def extract_features(
    mark1: str = Query(..., description="First trademark text"),
    mark2: str = Query(..., description="Second trademark text")
):
    """
    Extract all linguistic features for a pair (for debugging/analysis)
    """
    if linguistic_extractor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        features = linguistic_extractor.extract_all_features(mark1, mark2)
        
        return {
            "mark1": mark1,
            "mark2": mark2,
            "features": features
        }
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-similarity", response_model=List[SimilarityResponse])
async def batch_similarity_check(
    pairs: List[Dict[str, str]] = Field(..., description="List of {mark1, mark2} pairs")
):
    """
    Check similarity for multiple trademark pairs in batch
    
    Example: [{"mark1": "A", "mark2": "B"}, {"mark1": "C", "mark2": "D"}]
    """
    if hybrid_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        results = []
        
        for pair in pairs:
            mark1 = pair.get('mark1')
            mark2 = pair.get('mark2')
            
            if not mark1 or not mark2:
                raise HTTPException(status_code=400, detail="Each pair must have mark1 and mark2")
            
            label, probability, _ = hybrid_classifier.predict(mark1, mark2, return_details=False)
            
            if probability >= 0.7:
                risk_level = "HIGH"
            elif probability >= 0.5:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            results.append(SimilarityResponse(
                label=label,
                label_text="Similar" if label == 1 else "Dissimilar",
                probability=round(probability, 4),
                risk_level=risk_level,
                details=None
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Batch similarity error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches (embeddings, translations, etc.)"""
    try:
        if cnn_encoder:
            cnn_encoder.clear_cache()
        if linguistic_extractor:
            linguistic_extractor.cache_manager.clear()
        
        return {"status": "success", "message": "All caches cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=Dict)
async def get_stats():
    """Get API statistics"""
    stats = {
        "models_status": {
            "cnn_encoder": cnn_encoder is not None,
            "linguistic_extractor": linguistic_extractor is not None,
            "hybrid_classifier": hybrid_classifier is not None,
            "retriever": retriever is not None
        },
        "indexed_trademarks": len(retriever.trademark_db) if retriever else 0,
        "config": {
            "max_sequence_length": config.MAX_SEQUENCE_LENGTH,
            "top_k_candidates": config.TOP_K_CANDIDATES,
            "similarity_threshold": config.SIMILARITY_THRESHOLD
        }
    }
    
    return stats


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api_service:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )
