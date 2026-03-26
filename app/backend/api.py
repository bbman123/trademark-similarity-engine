"""
Trademark Registration Decision System - App Backend
(Mirror of huggingface-deploy/api.py for local development)
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler

import jellyfish
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

import re
import unicodedata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _build_reverse_lexicon():
    for en_word, data in CROSS_LANGUAGE_LEXICON.items():
        for lang in ["ha", "yo"]:
            translation = data.get(lang, "")
            if not translation:
                continue
            clean_full = _strip_diacritics(translation)
            if len(clean_full) >= 3:
                _REVERSE_LEXICON[clean_full] = (en_word, _LANG_NAMES[lang])
            for part in clean_full.split():
                if len(part) >= 3:
                    _REVERSE_LEXICON[part] = (en_word, _LANG_NAMES[lang])


_build_reverse_lexicon()


def _detect_cross_language_words(text: str) -> List[Dict[str, str]]:
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
    words1 = set(re.findall(r'\b\w+\b', _strip_diacritics(text1)))
    words2 = set(re.findall(r'\b\w+\b', _strip_diacritics(text2)))
    notes = []
    for w1 in words1:
        if w1 in _REVERSE_LEXICON:
            en_word, lang = _REVERSE_LEXICON[w1]
            if _strip_diacritics(en_word) in words2:
                notes.append(f"'{w1}' ({lang}) = '{en_word}' (English)")
                continue
            for syn in CROSS_LANGUAGE_LEXICON.get(en_word, {}).get("synonyms", []):
                if _strip_diacritics(syn) in words2:
                    notes.append(f"'{w1}' ({lang}) ≈ '{syn}' (synonym of '{en_word}')")
                    break
        if w1 in CROSS_LANGUAGE_LEXICON:
            data = CROSS_LANGUAGE_LEXICON[w1]
            for lang_code in ["ha", "yo"]:
                trans = data.get(lang_code, "")
                if trans:
                    trans_parts = set(_strip_diacritics(trans).split())
                    if trans_parts & words2:
                        notes.append(f"'{w1}' (English) = '{trans}' ({_LANG_NAMES[lang_code]})")
                        break
    return notes


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class RegistrationRequest(BaseModel):
    trademarks: List[str] = Field(..., description="List of 1-3 trademark names to check for registration")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator('trademarks')
    @classmethod
    def clean_trademarks(cls, v):
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError('At least one non-empty trademark name is required')
        if len(cleaned) > 3:
            raise ValueError('Maximum 3 trademarks allowed per request')
        return cleaned


class TrademarkMatch(BaseModel):
    trademark: str
    similarity_score: float
    matched_language: Optional[str] = None
    matched_variant: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class TrademarkDecision(BaseModel):
    trademark: str
    decision: str
    max_similarity: float
    closest_match: Optional[TrademarkMatch] = None
    top_matches: List[TrademarkMatch] = []
    reason: str
    cross_language_note: Optional[str] = None


class RegistrationResponse(BaseModel):
    threshold: float
    database_size: int
    results: List[TrademarkDecision]
    summary: Dict[str, int]


class SimilarityRequest(BaseModel):
    mark1: str = Field(..., min_length=1, max_length=200)
    mark2: str = Field(..., min_length=1, max_length=200)
    include_details: bool = Field(default=False)

    @field_validator('mark1', 'mark2')
    @classmethod
    def clean_text(cls, v):
        return v.strip()


class SimilarityResponse(BaseModel):
    label: int
    label_text: str
    probability: float
    confidence: float
    risk_level: str
    recommendation: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
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
        self._load_cnn()
        self._load_svm()
        self._load_semantic_model()
        logger.info("[SUCCESS] All models loaded")

    def _load_cnn(self):
        tokenizer_path = self.models_dir / "cnn_encoder_tokenizer.pkl"
        for cnn_path in [self.models_dir / "best_cnn_model.h5", self.models_dir / "cnn_encoder.keras"]:
            if not cnn_path.exists():
                continue
            try:
                self.cnn_model = keras.models.load_model(str(cnn_path), safe_mode=False, compile=False)
                try:
                    self.cnn_encoder = keras.models.Model(
                        inputs=self.cnn_model.get_layer('cnn_encoder').input,
                        outputs=self.cnn_model.get_layer('cnn_encoder').output
                    )
                except Exception:
                    self.cnn_encoder = self.cnn_model
                break
            except Exception:
                continue
        if self.cnn_encoder is None:
            raise FileNotFoundError("Could not load CNN model")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

    def _load_svm(self):
        with open(self.models_dir / "hybrid_svm.pkl", 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                self.svm_model = data.get('model') or data.get('svm_model') or data.get('svm') or data.get('classifier')
                self.scaler = data.get('scaler') or data.get('scaler_model') or data.get('feature_scaler')
            else:
                self.svm_model = data
                self.scaler = None

    def _load_semantic_model(self):
        self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def encode_text(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.config['max_sequence_length'], padding='post')
        return self.cnn_encoder.predict(padded, verbose=0)[0]

    def encode_texts_batch(self, texts, batch_size=256):
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=self.config['max_sequence_length'], padding='post')
        return self.cnn_encoder.predict(padded, verbose=0, batch_size=batch_size)

    def extract_linguistic_features(self, mark1, mark2):
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
        return np.array([levenshtein_dist, jaro_winkler, soundex_match, metaphone_match,
                         semantic_sim, semantic_sim, semantic_sim, len_diff, len(m1), len(m2)])

    def predict(self, mark1, mark2, mark2_ha="", mark2_yo="", return_details=False):
        emb1 = self.encode_text(mark1)

        # Score against primary wordmark
        emb2 = self.encode_text(mark2)
        ling = self.extract_linguistic_features(mark1, mark2)
        hybrid = np.concatenate([emb1, emb2, ling]).reshape(1, -1)
        scaled = self.scaler.transform(hybrid)
        label = int(self.svm_model.predict(scaled)[0])
        prob = float(self.svm_model.predict_proba(scaled)[0][1])

        best_lang = "en"
        best_variant = mark2
        variant_scores = {"en": round(prob, 4)}

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
                    if prob_var > prob:
                        prob = prob_var
                        label = int(self.svm_model.predict(scaled_var)[0])
                        best_lang = lang_code
                        best_variant = variant
                except Exception:
                    pass

        details = None
        if return_details:
            m1, m2 = mark1.lower(), mark2.lower()
            e1 = self.semantic_model.encode([mark1], convert_to_numpy=True)
            e2_en = self.semantic_model.encode([mark2], convert_to_numpy=True)
            sim_en = float(sklearn_cosine_similarity(e1, e2_en)[0][0])

            if mark2_ha and mark2_ha.lower() != mark2.lower():
                e2_ha = self.semantic_model.encode([mark2_ha], convert_to_numpy=True)
                sim_ha = float(sklearn_cosine_similarity(e1, e2_ha)[0][0])
            else:
                sim_ha = sim_en
            if mark2_yo and mark2_yo.lower() != mark2.lower():
                e2_yo = self.semantic_model.encode([mark2_yo], convert_to_numpy=True)
                sim_yo = float(sklearn_cosine_similarity(e1, e2_yo)[0][0])
            else:
                sim_yo = sim_en

            details = {
                "visual_features": {"levenshtein_distance": int(ling[0]), "jaro_winkler_similarity": round(float(ling[1]), 4)},
                "phonetic_features": {"soundex_match": bool(ling[2]), "metaphone_match": bool(ling[3])},
                "semantic_features": {"similarity_en": round(sim_en, 4), "similarity_ha": round(sim_ha, 4), "similarity_yo": round(sim_yo, 4)},
                "length_features": {"length_difference": int(ling[7]), "mark1_length": int(ling[8]), "mark2_length": int(ling[9])}
            }

        match_info = {"best_lang": best_lang, "best_variant": best_variant, "variant_scores": variant_scores}
        return label, prob, details, match_info


class TrademarkDatabase:
    def __init__(self):
        self.trademarks = []
        self.embeddings = None
        self.variant_embeddings = None
        self.variant_to_trademark_idx = []
        self.variant_lang = []
        self.loaded = False

    def load(self, csv_path, model_loader):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['wordmark'])
        df = df[df['wordmark'].str.strip() != '']
        self.trademarks = df.to_dict('records')

        # Primary embeddings
        texts = [tm['wordmark'] for tm in self.trademarks]
        self.embeddings = model_loader.encode_texts_batch(texts)

        # Variant embeddings with deduplication
        try:
            self._build_variant_index(model_loader)
        except Exception as e:
            logger.warning(f"Could not build variant index: {e}")
            self.variant_embeddings = None

        self.loaded = True
        logger.info(f"Database loaded: {len(self.trademarks)} trademarks")

    def _build_variant_index(self, model_loader):
        unique_text_to_idx = {}
        unique_texts = []
        variant_entries = []
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
            self.variant_embeddings = None
            return
        unique_embeddings = model_loader.encode_texts_batch(unique_texts)
        self.variant_to_trademark_idx = []
        self.variant_lang = []
        variant_emb_list = []
        for tm_idx, lang_code, unique_idx in variant_entries:
            self.variant_to_trademark_idx.append(tm_idx)
            self.variant_lang.append(lang_code)
            variant_emb_list.append(unique_embeddings[unique_idx])
        self.variant_embeddings = np.array(variant_emb_list)
        logger.info(f"   Variant index: {len(variant_entries)} entries ({len(unique_texts)} unique)")

    def find_candidates(self, query_embedding, top_k=20):
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # Search primary embeddings
        db_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
        primary_sims = np.dot(db_norms, query_norm)

        best_sim = dict(enumerate(primary_sims.tolist()))
        match_lang = {i: 'primary' for i in range(len(self.trademarks))}

        # Search variant embeddings
        if self.variant_embeddings is not None and len(self.variant_embeddings) > 0:
            var_norms = self.variant_embeddings / (np.linalg.norm(self.variant_embeddings, axis=1, keepdims=True) + 1e-10)
            var_sims = np.dot(var_norms, query_norm)
            for vi, sim_val in enumerate(var_sims):
                tm_idx = self.variant_to_trademark_idx[vi]
                if sim_val > best_sim.get(tm_idx, -1):
                    best_sim[tm_idx] = float(sim_val)
                    match_lang[tm_idx] = self.variant_lang[vi]

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


model_loader = None
trademark_db = None


@asynccontextmanager
async def lifespan(app):
    global model_loader, trademark_db
    model_loader = HybridModelLoader(models_dir="models")
    model_loader.load_models()
    trademark_db = TrademarkDatabase()
    db_path = Path("data/trademark_database.csv")
    if db_path.exists():
        trademark_db.load(str(db_path), model_loader)
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Trademark Registration Decision System", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {"name": "Trademark Registration Decision System", "version": "2.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    ml = model_loader is not None and model_loader.cnn_encoder is not None
    db = trademark_db is not None and trademark_db.loaded
    return {"status": "healthy" if ml and db else "unhealthy", "timestamp": datetime.now().isoformat(),
            "models_loaded": ml, "database_loaded": db, "database_size": len(trademark_db.trademarks) if db else 0,
            "model_info": {"cnn_loaded": True, "svm_loaded": True, "semantic_loaded": True} if ml else {}}


@app.post("/check-registration", response_model=RegistrationResponse)
async def check_registration(request: RegistrationRequest):
    if not model_loader or not trademark_db or not trademark_db.loaded:
        raise HTTPException(status_code=503, detail="System not ready")
    results, approved, rejected = [], 0, 0
    for name in request.trademarks:
        try:
            d = _evaluate_trademark(name, request.threshold, request.top_k)
            results.append(d)
            if d.decision == "APPROVED":
                approved += 1
            else:
                rejected += 1
        except Exception as e:
            results.append(TrademarkDecision(trademark=name, decision="ERROR", max_similarity=0.0, reason=str(e)))
    return RegistrationResponse(threshold=request.threshold, database_size=len(trademark_db.trademarks),
                                results=results, summary={"total_submitted": len(request.trademarks), "approved": approved, "rejected": rejected})


def _evaluate_trademark(name, threshold, top_k):
    lexicon_detections = _detect_cross_language_words(name)

    qe = model_loader.encode_text(name)
    candidates = trademark_db.find_candidates(qe, top_k=max(top_k, 20))
    seen_indices = {c['index'] for c in candidates}

    # Expanded search: if HA/YO words detected, also search for English equivalents
    if lexicon_detections:
        for det in lexicon_detections:
            expanded = _strip_diacritics(name).replace(det['word'], det['means'])
            if expanded != _strip_diacritics(name):
                try:
                    exp_emb = model_loader.encode_text(expanded)
                    for ec in trademark_db.find_candidates(exp_emb, top_k=10):
                        if ec['index'] not in seen_indices:
                            candidates.append(ec)
                            seen_indices.add(ec['index'])
                except Exception:
                    pass

    scored = []
    for c in candidates:
        try:
            _, prob, det, match_info = model_loader.predict(
                name, c['trademark'],
                mark2_ha=c.get('trademark_ha', ''),
                mark2_yo=c.get('trademark_yo', ''),
                return_details=True
            )
            best_lang = match_info['best_lang']
            best_variant = match_info['best_variant']
            variant_scores = match_info.get('variant_scores', {})

            equiv_notes = _detect_translation_equivalence(name, c['trademark'])
            if c.get('trademark_ha'):
                equiv_notes += _detect_translation_equivalence(name, c['trademark_ha'])
            if c.get('trademark_yo'):
                equiv_notes += _detect_translation_equivalence(name, c['trademark_yo'])
            equiv_notes = list(dict.fromkeys(equiv_notes))

            if det is None:
                det = {}
            if best_lang != 'en' and best_variant:
                det['cross_language'] = {'matched_via': _LANG_NAMES.get(best_lang, best_lang),
                                          'matched_variant': best_variant, 'variant_scores': variant_scores}
            elif len(variant_scores) > 1:
                det['cross_language'] = {'matched_via': 'English (primary)', 'variant_scores': variant_scores}
            if equiv_notes:
                det['translation_notes'] = equiv_notes

            scored.append({
                'trademark': c['trademark'], 'similarity_score': round(prob, 4), 'details': det,
                'matched_language': _LANG_NAMES.get(best_lang, 'English'),
                'matched_variant': best_variant if best_lang != 'en' else None,
            })
        except Exception:
            pass
    scored.sort(key=lambda x: x['similarity_score'], reverse=True)

    top_m = [TrademarkMatch(trademark=m['trademark'], similarity_score=m['similarity_score'],
                            matched_language=m.get('matched_language'), matched_variant=m.get('matched_variant'),
                            details=m.get('details')) for m in scored[:top_k]]
    mx = scored[0]['similarity_score'] if scored else 0.0
    cl = scored[0] if scored else None

    cross_notes = []
    if lexicon_detections:
        for d in lexicon_detections:
            cross_notes.append(f"'{d['word']}' is {d['language']} for '{d['means']}'")
    if cl:
        if cl.get('matched_language', 'English') != 'English' and cl.get('matched_variant'):
            cross_notes.append(f"Strongest match via {cl['matched_language']} variant: \"{cl['matched_variant']}\"")
        if cl.get('details', {}).get('translation_notes'):
            cross_notes.extend(cl['details']['translation_notes'])
    cross_notes = list(dict.fromkeys(cross_notes))
    cross_language_note = ". ".join(cross_notes) if cross_notes else None

    if mx >= threshold:
        dec = "REJECTED"
        reason = f"Too similar to '{cl['trademark']}' ({mx*100:.1f}% >= {threshold*100:.0f}%)"
        if cross_language_note:
            reason += f". Cross-language alert: {cross_language_note}"
    elif cl:
        dec = "APPROVED"
        reason = f"Sufficiently unique. Closest: '{cl['trademark']}' at {mx*100:.1f}%"
        if cross_language_note:
            reason += f". Note: {cross_language_note}"
    else:
        dec, reason = "APPROVED", "No similar trademarks found"

    cm = TrademarkMatch(trademark=cl['trademark'], similarity_score=cl['similarity_score'],
                        matched_language=cl.get('matched_language'), matched_variant=cl.get('matched_variant'),
                        details=cl.get('details')) if cl else None
    return TrademarkDecision(trademark=name, decision=dec, max_similarity=mx, closest_match=cm,
                             top_matches=top_m, reason=reason, cross_language_note=cross_language_note)


@app.post("/similarity-check", response_model=SimilarityResponse)
async def check_similarity(request: SimilarityRequest):
    if not model_loader:
        raise HTTPException(status_code=503, detail="Models not loaded")
    label, prob, details, _ = model_loader.predict(request.mark1, request.mark2, return_details=request.include_details)
    rl = "HIGH" if prob >= 0.7 else ("MEDIUM" if prob >= 0.4 else "LOW")
    rec = {"HIGH": "High risk of confusion", "MEDIUM": "Moderate risk", "LOW": "Low risk"}[rl]
    return {"label": label, "label_text": "Similar" if label == 1 else "Dissimilar",
            "probability": prob, "confidence": prob, "risk_level": rl, "recommendation": rec, "details": details}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
