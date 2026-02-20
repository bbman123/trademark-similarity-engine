# Trademark Similarity Engine
## Hybrid CNN+SVM with Multilingual Linguistic AI (EN/HA/YO)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-ready AI-powered trademark similarity detection system combining:
- **Character-level CNN** for visual embedding extraction
- **SVM** with hybrid features for classification
- **Enhanced Linguistic Features**: Synonyms, antonyms, phonetics, spelling similarity
- **Multilingual Support**: English, Hausa, Yoruba semantic analysis
- **Efficient Retrieval**: Fast candidate selection using approximate nearest neighbors
- **Explainable Predictions**: Detailed feature breakdown and reasoning

## 📋 Features

- ✅ **Hybrid Architecture**: CNN embeddings + 14+ linguistic features
- ✅ **Multilingual AI**: Semantic analysis across EN/HA/YO
- ✅ **Synonym/Antonym Detection**: Domain-specific lexicon for trademark terms
- ✅ **Phonetic Matching**: Soundex & Metaphone for sound-alike detection
- ✅ **Explainable AI**: Detailed breakdown of similarity factors
- ✅ **RESTful API**: FastAPI with async support
- ✅ **Caching**: Intelligent caching of embeddings and translations
- ✅ **Production Ready**: Modular, tested, documented

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd trademark-similarity-engine

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Models

Ensure trained models are in the `models/` directory:
- `cnn_encoder.keras` (CNN model)
- `cnn_encoder_tokenizer.pkl` (Tokenizer)
- `hybrid_svm.pkl` (SVM + scaler)

### 3. Command-Line Inference

```bash
# Basic similarity check
python inference.py --mark1 "SuperCoffee" --mark2 "Super Coffee"

# With detailed explanation
python inference.py --mark1 "TechSmart" --mark2 "SmartTech" --details

# JSON output
python inference.py --mark1 "BrandA" --mark2 "BrandB" --json --details
```

### 4. Start API Server

```bash
# Start FastAPI server
python -m uvicorn src.api_service:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for interactive API documentation.

## 📚 API Usage

### Check Similarity (POST)

```bash
curl -X POST "http://localhost:8000/similarity-check" \
  -H "Content-Type: application/json" \
  -d '{
    "mark1": "SuperCoffee",
    "mark2": "Super Coffee",
    "include_details": true
  }'
```

### Check Similarity (GET)

```bash
curl "http://localhost:8000/similarity-check?mark1=TechSmart&mark2=SmartTech&include_details=true"
```

### Batch Similarity Check

```bash
curl -X POST "http://localhost:8000/batch-similarity" \
  -H "Content-Type: application/json" \
  -d '[
    {"mark1": "BrandA", "mark2": "BrandB"},
    {"mark1": "TechX", "mark2": "TechY"}
  ]'
```

### Extract Features (for analysis)

```bash
curl "http://localhost:8000/features?mark1=Coffee&mark2=Cafe"
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADEMARK SIMILARITY ENGINE              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │     Query: "SuperCoffee" vs          │
        │            "Super Coffee"             │
        └───────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
    ┌───────────────┐              ┌────────────────┐
    │  CNN Encoder  │              │   Linguistic   │
    │   (Char-CNN)  │              │    Features    │
    │               │              │                │
    │  Embeddings:  │              │ • Synonyms     │
    │  [64 dims]    │              │ • Antonyms     │
    │  for each     │              │ • Phonetics    │
    │  trademark    │              │ • Spelling     │
    │               │              │ • Semantic     │
    │               │              │   (EN/HA/YO)   │
    └───────────────┘              └────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Hybrid Feature Vector │
                │  [128 + 14 features]   │
                └───────────────────────┘
                            │
                            ▼
                    ┌───────────┐
                    │    SVM    │
                    │ Classifier │
                    └───────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │  Prediction + Explainability  │
            │                               │
            │  • Label: Similar/Dissimilar  │
            │  • Probability: 0.87          │
            │  • Risk: HIGH                 │
            │  • Key Factors:               │
            │    - Phonetic match           │
            │    - High synonym overlap     │
            │    - Visual similarity: 0.85  │
            └───────────────────────────────┘
```

## 📊 Model Performance

Based on test set evaluation:
- **Accuracy**: 73.3%
- **Precision**: 66.7%
- **Recall**: 85.7% (prioritizes catching similar marks)
- **F1 Score**: 75.0%
- **ROC-AUC**: 89.3%

## 🔧 Module Overview

```
src/
├── __init__.py                    # Package initialization
├── config.py                       # Configuration management
├── cache_manager.py                # Caching system with TTL
├── cnn_encoder.py                  # CNN embedding extraction
├── linguistic_features.py          # 14+ linguistic features + synonyms/antonyms
├── svm_classifier.py               # Hybrid CNN+SVM classifier
├── retrieval.py                    # Candidate retrieval system
└── api_service.py                  # FastAPI REST API

inference.py                        # CLI inference script
requirements.txt                    # Dependencies
```

## 🎯 Features Extracted

### Visual/Spelling Features
- Levenshtein distance (normalized)
- Jaro-Winkler similarity
- Hamming distance
- Character n-gram overlap

### Phonetic Features
- Soundex matching
- Metaphone matching

### Semantic Features
- Multilingual embeddings (EN/HA/YO)
- Cosine similarity across languages

### Synonym/Antonym Features (NEW!)
- Synonym overlap score (Jaccard)
- Exact match count
- Antonym detection flag
- Domain-specific trademark lexicon

### CNN Features
- 64-dimensional character-level embeddings
- Learned from training data

## 🌍 Multilingual Support

The system includes a curated lexicon for trademark-relevant terms with:
- **English** base terms
- **Hausa** translations
- **Yoruba** translations
- **Synonyms** for semantic expansion
- **Antonyms** for dissimilarity detection

Example lexicon entry:
```python
"premium": {
    "ha": "mai kyau",
    "yo": "iyebíye",
    "synonyms": ["quality", "superior", "elite", "deluxe", "luxury"],
    "antonyms": ["basic", "standard", "economy", "cheap"]
}
```

## 📈 Risk Levels

- **HIGH** (≥70% probability): Strong similarity, high confusion risk
- **MEDIUM** (50-69% probability): Moderate similarity, potential issues
- **LOW** (<50% probability): Dissimilar, low confusion risk

## 🧪 Example Usage (Python)

```python
from src.svm_classifier import HybridSimilarityClassifier

# Initialize classifier
classifier = HybridSimilarityClassifier()

# Check similarity
label, prob, details = classifier.predict(
    "SuperCoffee",
    "Super Coffee",
    return_details=True
)

print(f"Prediction: {'Similar' if label == 1 else 'Dissimilar'}")
print(f"Probability: {prob:.2%}")
print(f"Key Factors: {details['key_factors']}")
```

## 🛠️ Advanced Usage

### Indexing Trademarks for Retrieval

```python
from src.retrieval import TrademarkRetriever

retriever = TrademarkRetriever()

# Index existing trademarks
trademarks = [
    {"text": "BrandA", "id": "TM001", "class": "25"},
    {"text": "BrandB", "id": "TM002", "class": "25"},
    # ... more trademarks
]

retriever.index_trademarks(trademarks)

# Find similar candidates
candidates = retriever.retrieve_hybrid(
    query_text="BrandX",
    top_k=10,
    threshold=0.5
)

# Save index for later
retriever.save_index("trademark_index.pkl")
```

### Custom Configuration

```python
from src.config import Config

config = Config(
    TOP_K_CANDIDATES=50,
    SIMILARITY_THRESHOLD=0.6,
    CACHE_ENABLED=True,
    CACHE_TTL_SECONDS=7200
)
```

## 📦 Deployment

### Docker (Coming Soon)

```bash
docker build -t trademark-similarity-engine .
docker run -p 8000:8000 trademark-similarity-engine
```

### Production Considerations

1. **Caching**: Enable Redis for distributed caching
2. **Scaling**: Use multiple uvicorn workers
3. **Monitoring**: Add logging and metrics (Prometheus/Grafana)
4. **Security**: Add authentication, rate limiting
5. **Indexing**: Pre-compute and cache embeddings for large databases

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📝 License

MIT License - see LICENSE file

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for trademark protection and business integrity**
