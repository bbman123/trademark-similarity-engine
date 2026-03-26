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

import sys
import time as _time
import threading
import re
import unicodedata

# Setup logging - use StreamHandler with flush for HuggingFace
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CROSS-LANGUAGE LEXICON & HELPERS
# ============================================================================

CROSS_LANGUAGE_LEXICON = {
    "company": {"ha": "kamfani", "yo": "ilé-iṣẹ́", "synonyms": ["business", "corporation", "firm", "enterprise"]},
    "business": {"ha": "kasuwanci", "yo": "iṣẹ́ owó", "synonyms": ["company", "enterprise", "trade", "commerce"]},
    "market": {"ha": "kasuwa", "yo": "ọjà", "synonyms": ["marketplace", "bazaar", "store", "shop"]},
    "store": {"ha": "shago", "yo": "ṣọ́ọ̀bù", "synonyms": ["shop", "outlet", "mart"]},
    "brand": {"ha": "alama", "yo": "àmì", "synonyms": ["trademark", "label", "logo", "mark"]},
    "premium": {"ha": "mai kyau", "yo": "iyebíye", "synonyms": ["quality", "superior", "elite", "deluxe", "luxury"]},
    "super": {"ha": "babba", "yo": "púpọ̀", "synonyms": ["mega", "ultra", "great", "supreme"]},
    "best": {"ha": "mafi kyau", "yo": "dára jùlọ", "synonyms": ["finest", "top", "premier"]},
    "gold": {"ha": "zinariya", "yo": "wúrà", "synonyms": ["golden", "premium"]},
    "coffee": {"ha": "kofi", "yo": "kọ́fí", "synonyms": ["cafe", "caffeine", "java"]},
    "tea": {"ha": "shayi", "yo": "tii", "synonyms": ["beverage"]},
    "food": {"ha": "abinci", "yo": "oúnjẹ", "synonyms": ["meal", "cuisine", "dish"]},
    "restaurant": {"ha": "gidan cin abinci", "yo": "ilé oúnjẹ", "synonyms": ["cafe", "diner", "eatery"]},
    "tech": {"ha": "fasaha", "yo": "ìmọ̀-ẹrọ", "synonyms": ["technology", "digital", "electronic", "smart"]},
    "digital": {"ha": "na dijital", "yo": "oníjìtù", "synonyms": ["electronic", "tech", "cyber", "online"]},
    "smart": {"ha": "mai hankali", "yo": "ológbọ́n", "synonyms": ["intelligent", "clever", "advanced"]},
    "new": {"ha": "sabon", "yo": "títun", "synonyms": ["fresh", "modern", "novel", "recent"]},
    "fresh": {"ha": "sabo", "yo": "tuntun", "synonyms": ["new", "crisp", "pure", "natural"]},
    "natural": {"ha": "na halitta", "yo": "àdáyébá", "synonyms": ["organic", "pure", "authentic"]},
    "water": {"ha": "ruwa", "yo": "omi", "synonyms": ["aqua", "hydro"]},
    "king": {"ha": "sarki", "yo": "ọba", "synonyms": ["royal", "crown", "monarch"]},
    "star": {"ha": "tauraro", "yo": "ìràwọ̀", "synonyms": ["stellar", "astro"]},
    "lion": {"ha": "zaki", "yo": "kìnnìún", "synonyms": ["leo", "pride"]},
    "eagle": {"ha": "gaggafa", "yo": "àṣá", "synonyms": ["hawk", "falcon"]},
    "power": {"ha": "iko", "yo": "agbára", "synonyms": ["energy", "force", "strength"]},
    "speed": {"ha": "sauri", "yo": "iyara", "synonyms": ["fast", "quick", "rapid", "swift"]},
    "life": {"ha": "rayuwa", "yo": "ìgbésí ayé", "synonyms": ["living", "vital"]},
    "health": {"ha": "lafiya", "yo": "ìlera", "synonyms": ["wellness", "medical", "healthy"]},
    "beauty": {"ha": "kyau", "yo": "ẹwà", "synonyms": ["beautiful", "pretty", "cosmetic"]},
    "home": {"ha": "gida", "yo": "ilé", "synonyms": ["house", "residence", "domestic"]},
    "light": {"ha": "haske", "yo": "ìmọ́lẹ̀", "synonyms": ["bright", "glow", "shine"]},
    "sun": {"ha": "rana", "yo": "oòrùn", "synonyms": ["solar", "sunny"]},
    "moon": {"ha": "wata", "yo": "oṣùpá", "synonyms": ["lunar"]},
    "earth": {"ha": "duniya", "yo": "ayé", "synonyms": ["world", "globe", "terra"]},
    "green": {"ha": "kore", "yo": "àwọ̀ ewé", "synonyms": ["eco", "organic", "natural"]},
}

_LANG_NAMES = {"ha": "Hausa", "yo": "Yoruba", "en": "English", "primary": "English"}
_REVERSE_LEXICON: Dict[str, tuple] = {}


def _strip_diacritics(text: str) -> str:
    """Remove diacritical marks for fuzzy matching"""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _build_reverse_lexicon():
    for en_word, data in CROSS_LANGUAGE_LEXICON.items():
        for lang in ["ha", "yo"]:
            translation = data.get(lang, "")
            if not translation:
                continue
            # Store full phrase
            clean_full = _strip_diacritics(translation)
            if len(clean_full) >= 3:
                _REVERSE_LEXICON[clean_full] = (en_word, _LANG_NAMES[lang])
            # Store individual words of multi-word translations
            for part in clean_full.split():
                if len(part) >= 3:
                    _REVERSE_LEXICON[part] = (en_word, _LANG_NAMES[lang])


_build_reverse_lexicon()


def _detect_cross_language_words(text: str) -> List[Dict[str, str]]:
    """Detect if any words in the text are known Hausa/Yoruba words"""
    words = re.findall(r'\b\w+\b', _strip_diacritics(text))
    detections = []
    seen = set()
    for word in words:
        if word in _REVERSE_LEXICON and word not in seen:
            en_word, lang = _REVERSE_LEXICON[word]
            detections.append({"word": word, "means": en_word, "language": lang})
            seen.add(word)
    return detections


def _detect_translation_equivalence(text1: str, text2: str) -> List[str]:
    """Detect if words in text1 are translations/synonyms of words in text2"""
    words1 = set(re.findall(r'\b\w+\b', _strip_diacritics(text1)))
    words2 = set(re.findall(r'\b\w+\b', _strip_diacritics(text2)))
    notes = []

    for w1 in words1:
        # Check if w1 is a known HA/YO word whose English equivalent is in text2
        if w1 in _REVERSE_LEXICON:
            en_word, lang = _REVERSE_LEXICON[w1]
            if _strip_diacritics(en_word) in words2:
                notes.append(f"'{w1}' ({lang}) = '{en_word}' (English)")
                continue
            # Check synonyms of the English equivalent
            for syn in CROSS_LANGUAGE_LEXICON.get(en_word, {}).get("synonyms", []):
                if _strip_diacritics(syn) in words2:
                    notes.append(f"'{w1}' ({lang}) ≈ '{syn}' (synonym of '{en_word}')")
                    break

        # Check if w1 is a known English word whose HA/YO translation is in text2
        if w1 in CROSS_LANGUAGE_LEXICON:
            data = CROSS_LANGUAGE_LEXICON[w1]
            for lang_code in ["ha", "yo"]:
                trans = data.get(lang_code, "")
                if trans:
                    trans_parts = set(_strip_diacritics(trans).split())
                    if trans_parts & words2:
                        notes.append(
                            f"'{w1}' (English) = '{trans}' ({_LANG_NAMES[lang_code]})"
                        )
                        break

    return notes

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
    matched_language: Optional[str] = None
    matched_variant: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class TrademarkDecision(BaseModel):
    """Registration decision for a single trademark"""
    trademark: str
    decision: str  # "APPROVED" or "REJECTED"
    max_similarity: float
    closest_match: Optional[TrademarkMatch] = None
    top_matches: List[TrademarkMatch] = []
    reason: str
    cross_language_note: Optional[str] = None


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

    def extract_linguistic_features_detail(
        self, mark1: str, mark2: str,
        mark2_ha: str = "", mark2_yo: str = ""
    ) -> Dict[str, Any]:
        """Extract linguistic features with actual per-language semantic similarities"""
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

        # Compute actual per-language semantic similarities using multilingual model
        emb1 = self.semantic_model.encode([mark1], convert_to_numpy=True)
        emb2_en = self.semantic_model.encode([mark2], convert_to_numpy=True)
        sim_en = float(sklearn_cosine_similarity(emb1, emb2_en)[0][0])

        # Hausa: compare against HA variant if available and different
        if mark2_ha and mark2_ha.lower() != mark2.lower():
            emb2_ha = self.semantic_model.encode([mark2_ha], convert_to_numpy=True)
            sim_ha = float(sklearn_cosine_similarity(emb1, emb2_ha)[0][0])
        else:
            sim_ha = sim_en

        # Yoruba: compare against YO variant if available and different
        if mark2_yo and mark2_yo.lower() != mark2.lower():
            emb2_yo = self.semantic_model.encode([mark2_yo], convert_to_numpy=True)
            sim_yo = float(sklearn_cosine_similarity(emb1, emb2_yo)[0][0])
        else:
            sim_yo = sim_en

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
                "similarity_en": round(sim_en, 4),
                "similarity_ha": round(sim_ha, 4),
                "similarity_yo": round(sim_yo, 4)
            },
            "length_features": {
                "length_difference": abs(len(m1) - len(m2)),
                "mark1_length": len(m1),
                "mark2_length": len(m2)
            }
        }

    def predict(
        self, mark1: str, mark2: str,
        mark2_ha: str = "", mark2_yo: str = "",
        return_details: bool = False
    ):
        """Predict similarity between two trademarks, checking all language variants"""
        emb1 = self.encode_text(mark1)

        # Score against primary wordmark
        emb2 = self.encode_text(mark2)
        ling_features = self.extract_linguistic_features(mark1, mark2)
        hybrid_features = np.concatenate([emb1, emb2, ling_features]).reshape(1, -1)
        scaled_features = self.scaler.transform(hybrid_features)
        label = int(self.svm_model.predict(scaled_features)[0])
        probability = float(self.svm_model.predict_proba(scaled_features)[0][1])

        best_lang = "en"
        best_variant = mark2
        variant_scores = {"en": round(probability, 4)}

        # Also score against HA/YO variants — take the MAX probability
        for variant, lang_code in [(mark2_ha, "ha"), (mark2_yo, "yo")]:
            if variant and variant.strip() and variant.lower() != mark2.lower():
                try:
                    emb_var = self.encode_text(variant)
                    ling_var = self.extract_linguistic_features(mark1, variant)
                    hybrid_var = np.concatenate([emb1, emb_var, ling_var]).reshape(1, -1)
                    scaled_var = self.scaler.transform(hybrid_var)
                    prob_var = float(self.svm_model.predict_proba(scaled_var)[0][1])
                    variant_scores[lang_code] = round(prob_var, 4)
                    if prob_var > probability:
                        probability = prob_var
                        label = int(self.svm_model.predict(scaled_var)[0])
                        best_lang = lang_code
                        best_variant = variant
                except Exception:
                    pass

        details = None
        if return_details:
            details = self.extract_linguistic_features_detail(
                mark1, mark2, mark2_ha=mark2_ha, mark2_yo=mark2_yo
            )

        match_info = {
            "best_lang": best_lang,
            "best_variant": best_variant,
            "variant_scores": variant_scores,
        }
        return label, probability, details, match_info


# ============================================================================
# TRADEMARK DATABASE
# ============================================================================

class TrademarkDatabase:
    """Manages the existing trademark database for registration checks"""

    def __init__(self):
        self.trademarks: List[Dict[str, str]] = []
        self.embeddings: Optional[np.ndarray] = None
        # Variant embeddings: CNN embeddings for HA/YO columns where they differ
        self.variant_embeddings: Optional[np.ndarray] = None
        self.variant_to_trademark_idx: List[int] = []  # maps variant row -> original trademark index
        self.variant_lang: List[str] = []  # maps variant row -> language code
        self.loaded = False

    def load(self, csv_path: str, model_loader: HybridModelLoader):
        """Load trademarks from CSV and pre-compute CNN embeddings (primary only for fast startup)"""
        logger.info(f"Loading trademark database from {csv_path}...")

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['wordmark'])
        df = df[df['wordmark'].str.strip() != '']

        self.trademarks = df.to_dict('records')
        logger.info(f"   Loaded {len(self.trademarks)} registered trademarks")

        # Primary embeddings (wordmark column) — required for basic search
        logger.info("   Pre-computing CNN embeddings for primary wordmarks...")
        texts = [tm['wordmark'] for tm in self.trademarks]
        self.embeddings = model_loader.encode_texts_batch(texts)
        logger.info(f"   Primary embeddings shape: {self.embeddings.shape}")

        self.loaded = True
        logger.info("[SUCCESS] Trademark database ready (primary)")

    def load_variants_background(self, model_loader: HybridModelLoader):
        """Build variant embeddings in background thread (non-blocking)"""
        try:
            logger.info("[BACKGROUND] Starting variant embedding computation...")
            sys.stdout.flush()
            t0 = _time.time()
            self._build_variant_index(model_loader)
            logger.info(f"[BACKGROUND] Variant index ready in {_time.time() - t0:.1f}s")
            sys.stdout.flush()
        except Exception as e:
            logger.warning(f"[BACKGROUND] Variant index failed: {e}")
            logger.warning("[BACKGROUND] Cross-language search not available")
            self.variant_embeddings = None
            sys.stdout.flush()

    def _build_variant_index(self, model_loader: HybridModelLoader):
        """Build deduplicated variant embedding index for cross-language search"""
        # Collect unique variant texts and map them back
        unique_text_to_idx: Dict[str, int] = {}  # lowered text -> index in unique list
        unique_texts: List[str] = []
        variant_entries: List[tuple] = []  # (trademark_idx, lang_code, unique_text_idx)

        for i, tm in enumerate(self.trademarks):
            primary = tm['wordmark'].strip().lower()
            for lang_col, lang_code in [('wordmark_ha', 'ha'), ('wordmark_yo', 'yo')]:
                variant = str(tm.get(lang_col, '')).strip()
                if not variant or variant.lower() == primary:
                    continue
                key = variant.lower()
                if key not in unique_text_to_idx:
                    unique_text_to_idx[key] = len(unique_texts)
                    unique_texts.append(variant)
                variant_entries.append((i, lang_code, unique_text_to_idx[key]))

        if not unique_texts:
            logger.info("   No language variants found")
            self.variant_embeddings = None
            return

        logger.info(f"   Found {len(variant_entries)} variant entries ({len(unique_texts)} unique texts)")
        logger.info(f"   Computing CNN embeddings for {len(unique_texts)} unique variants...")
        unique_embeddings = model_loader.encode_texts_batch(unique_texts)
        logger.info(f"   Unique variant embeddings computed: {unique_embeddings.shape}")

        # Expand back: each variant entry gets the embedding of its unique text
        self.variant_to_trademark_idx = []
        self.variant_lang = []
        variant_emb_list = []
        for tm_idx, lang_code, unique_idx in variant_entries:
            self.variant_to_trademark_idx.append(tm_idx)
            self.variant_lang.append(lang_code)
            variant_emb_list.append(unique_embeddings[unique_idx])

        self.variant_embeddings = np.array(variant_emb_list)
        logger.info(f"   Variant index built: {self.variant_embeddings.shape}")

    def find_candidates(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Dict]:
        """Find top-K most similar trademarks by searching across ALL language variants"""
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # Search primary embeddings
        db_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
        primary_sims = np.dot(db_norms, query_norm)

        # Track best similarity per trademark index
        best_sim = dict(enumerate(primary_sims.tolist()))
        match_lang = {i: 'primary' for i in range(len(self.trademarks))}

        # Search variant embeddings (HA/YO) — may still be loading in background
        try:
            if self.variant_embeddings is not None and len(self.variant_embeddings) > 0:
                var_norms = self.variant_embeddings / (
                    np.linalg.norm(self.variant_embeddings, axis=1, keepdims=True) + 1e-10
                )
                var_sims = np.dot(var_norms, query_norm)
                for vi, sim_val in enumerate(var_sims):
                    tm_idx = self.variant_to_trademark_idx[vi]
                    if sim_val > best_sim.get(tm_idx, -1):
                        best_sim[tm_idx] = float(sim_val)
                        match_lang[tm_idx] = self.variant_lang[vi]
        except Exception:
            pass  # Variant index not ready yet, use primary only

        # Sort by best similarity and take top_k
        sorted_indices = sorted(best_sim.keys(), key=lambda i: best_sim[i], reverse=True)[:top_k]

        candidates = []
        for idx in sorted_indices:
            tm = self.trademarks[idx]
            candidates.append({
                'index': idx,
                'trademark': tm['wordmark'],
                'trademark_ha': str(tm.get('wordmark_ha', '')).strip(),
                'trademark_yo': str(tm.get('wordmark_yo', '')).strip(),
                'embedding_similarity': best_sim[idx],
                'matched_lang': match_lang[idx],
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
        t_start = _time.time()
        logger.info("=" * 80)
        logger.info("[STARTUP] Trademark Registration Decision System")
        logger.info("=" * 80)
        sys.stdout.flush()

        t0 = _time.time()
        model_loader = HybridModelLoader(models_dir="models")
        model_loader.load_models()
        logger.info(f"   Models loaded in {_time.time() - t0:.1f}s")
        sys.stdout.flush()

        trademark_db = TrademarkDatabase()
        db_path = Path("data/trademark_database.csv")
        if db_path.exists():
            t0 = _time.time()
            trademark_db.load(str(db_path), model_loader)
            logger.info(f"   Database loaded in {_time.time() - t0:.1f}s")

            # Start variant embedding computation in background thread
            # Server is usable immediately; cross-language search activates when done
            variant_thread = threading.Thread(
                target=trademark_db.load_variants_background,
                args=(model_loader,),
                daemon=True
            )
            variant_thread.start()
        else:
            logger.warning(f"[WARNING] Database not found at {db_path}")

        logger.info("=" * 80)
        logger.info(f"[SUCCESS] Server ready in {_time.time() - t_start:.1f}s total")
        logger.info("   (Cross-language variant index loading in background)")
        logger.info("=" * 80)
        sys.stdout.flush()

    except Exception as e:
        logger.error(f"[ERROR] Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
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
            "vocab_size": len(model_loader.tokenizer.word_index) if model_loader.tokenizer else 0,
            "cross_language_ready": (
                trademark_db is not None
                and trademark_db.variant_embeddings is not None
            )
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
    """Evaluate a single trademark against the database with cross-language intelligence"""
    # Step 0: Detect cross-language words in the submitted trademark
    lexicon_detections = _detect_cross_language_words(tm_name)

    # Stage 1: Fast retrieval using CNN embeddings (searches all language variants)
    query_embedding = model_loader.encode_text(tm_name)
    candidates = trademark_db.find_candidates(query_embedding, top_k=max(top_k, 20))
    seen_indices = {c['index'] for c in candidates}

    # Stage 1b: If lexicon detects HA/YO words, also search for English equivalents
    # e.g. "Kasuwa Fresh" detected as having "kasuwa" (Hausa for "market")
    #      → also search for "market Fresh" to catch English-named trademarks
    if lexicon_detections:
        for det in lexicon_detections:
            expanded = _strip_diacritics(tm_name).replace(det['word'], det['means'])
            if expanded != _strip_diacritics(tm_name):
                try:
                    exp_emb = model_loader.encode_text(expanded)
                    exp_candidates = trademark_db.find_candidates(exp_emb, top_k=10)
                    for ec in exp_candidates:
                        if ec['index'] not in seen_indices:
                            candidates.append(ec)
                            seen_indices.add(ec['index'])
                except Exception:
                    pass

    # Stage 2: Full hybrid scoring with CNN+SVM on top candidates
    scored_matches = []
    for candidate in candidates:
        existing_mark = candidate['trademark']
        existing_ha = candidate.get('trademark_ha', '')
        existing_yo = candidate.get('trademark_yo', '')
        try:
            _, probability, details, match_info = model_loader.predict(
                tm_name, existing_mark,
                mark2_ha=existing_ha, mark2_yo=existing_yo,
                return_details=True
            )
            best_lang = match_info['best_lang']
            best_variant = match_info['best_variant']
            variant_scores = match_info.get('variant_scores', {})

            # Detect translation equivalences between submitted and existing marks
            equiv_notes = _detect_translation_equivalence(tm_name, existing_mark)
            if existing_ha:
                equiv_notes += _detect_translation_equivalence(tm_name, existing_ha)
            if existing_yo:
                equiv_notes += _detect_translation_equivalence(tm_name, existing_yo)
            # Deduplicate
            equiv_notes = list(dict.fromkeys(equiv_notes))

            # Enrich details with cross-language info
            if details is None:
                details = {}

            if best_lang != 'en' and best_variant:
                details['cross_language'] = {
                    'matched_via': _LANG_NAMES.get(best_lang, best_lang),
                    'matched_variant': best_variant,
                    'variant_scores': variant_scores,
                }
            elif len(variant_scores) > 1:
                details['cross_language'] = {
                    'matched_via': 'English (primary)',
                    'variant_scores': variant_scores,
                }

            if equiv_notes:
                details['translation_notes'] = equiv_notes

            scored_matches.append({
                'trademark': existing_mark,
                'trademark_ha': existing_ha,
                'trademark_yo': existing_yo,
                'similarity_score': round(probability, 4),
                'details': details,
                'matched_language': _LANG_NAMES.get(best_lang, 'English'),
                'matched_variant': best_variant if best_lang != 'en' else None,
            })
        except Exception as e:
            logger.warning(f"   Failed to score '{tm_name}' vs '{existing_mark}': {e}")

    # Sort by similarity score descending
    scored_matches.sort(key=lambda x: x['similarity_score'], reverse=True)

    # Build top matches with language info
    top_matches = [
        TrademarkMatch(
            trademark=m['trademark'],
            similarity_score=m['similarity_score'],
            matched_language=m.get('matched_language'),
            matched_variant=m.get('matched_variant'),
            details=m.get('details'),
        )
        for m in scored_matches[:top_k]
    ]

    max_similarity = scored_matches[0]['similarity_score'] if scored_matches else 0.0
    closest = scored_matches[0] if scored_matches else None

    # Build cross-language note
    cross_notes = []
    if lexicon_detections:
        for d in lexicon_detections:
            cross_notes.append(f"'{d['word']}' is {d['language']} for '{d['means']}'")
    if closest:
        cl_lang = closest.get('matched_language', 'English')
        cl_variant = closest.get('matched_variant')
        if cl_lang != 'English' and cl_variant:
            cross_notes.append(
                f"Strongest match via {cl_lang} variant: \"{cl_variant}\""
            )
        cl_details = closest.get('details', {})
        if cl_details and cl_details.get('translation_notes'):
            cross_notes.extend(cl_details['translation_notes'])
    # Deduplicate while preserving order
    cross_notes = list(dict.fromkeys(cross_notes))
    cross_language_note = ". ".join(cross_notes) if cross_notes else None

    if max_similarity >= threshold:
        decision = "REJECTED"
        base_reason = (
            f"Too similar to existing trademark '{closest['trademark']}' "
            f"({max_similarity * 100:.1f}% similarity exceeds {threshold * 100:.0f}% threshold)"
        )
        if cross_language_note:
            reason = f"{base_reason}. Cross-language alert: {cross_language_note}"
        else:
            reason = base_reason
    else:
        decision = "APPROVED"
        if closest:
            reason = (
                f"Sufficiently unique. Closest match is '{closest['trademark']}' "
                f"at {max_similarity * 100:.1f}% similarity (below {threshold * 100:.0f}% threshold)"
            )
            if cross_language_note:
                reason += f". Note: {cross_language_note}"
        else:
            reason = "No similar trademarks found in the database"

    closest_match = None
    if closest:
        closest_match = TrademarkMatch(
            trademark=closest['trademark'],
            similarity_score=closest['similarity_score'],
            matched_language=closest.get('matched_language'),
            matched_variant=closest.get('matched_variant'),
            details=closest.get('details'),
        )

    return TrademarkDecision(
        trademark=tm_name,
        decision=decision,
        max_similarity=max_similarity,
        closest_match=closest_match,
        top_matches=top_matches,
        reason=reason,
        cross_language_note=cross_language_note,
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
        label, probability, details, _ = model_loader.predict(
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
