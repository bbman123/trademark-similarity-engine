# Trademark Similarity Engine
## Hybrid CNN+SVM with Multilingual Linguistic AI (EN/HA/YO)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.1-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An AI-powered trademark similarity detection system combining a **character-level Siamese CNN** for visual embedding extraction with an **SVM classifier** that fuses those embeddings with handcrafted multilingual linguistic features.

- **Character-level CNN** — Siamese architecture, learns visual/structural similarity from raw characters
- **Hybrid SVM** — 138-dimensional feature vector (CNN embeddings + 10 linguistic features)
- **Multilingual Semantic Analysis** — sentence embeddings in English, Hausa, and Yoruba
- **Phonetic Matching** — Soundex & Metaphone for sound-alike detection
- **RESTful API** — FastAPI with async support

## 📊 Model Performance (last full run, 29,606 pairs)

| Metric | Value |
|--------|-------|
| Test Accuracy | **86.5%** |
| Precision | **89.7%** |
| Recall | **82.4%** |
| F1 Score | **85.9%** |
| ROC-AUC | **93.5%** |

See [STEP1_2_GUIDE.md](STEP1_2_GUIDE.md) for full training details and output file descriptions.

## 📁 Project Structure

```
trademark-similarity-engine/
├── step1_2.ipynb          # Main training notebook (Steps 1–3)
├── requirements.txt       # Pinned dependencies
├── STEP1_2_GUIDE.md       # Notebook walkthrough (inputs, outputs, metrics)
├── ARCHITECTURE.md        # System architecture deep-dive
│
├── data/                  # Input & processed CSV files (git-ignored)
│   └── trademark_file.csv # Source trademark opposition pairs
├── models/                # Trained model weights (git-ignored)
│   ├── cnn_encoder.keras
│   ├── cnn_encoder_tokenizer.pkl
│   └── hybrid_svm.pkl
├── results/               # Evaluation metrics & plots
│   ├── evaluation_results.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── notebooks/             # Exploratory / legacy notebooks
├── scripts/               # CLI tools
│   ├── inference.py
│   ├── examples.py
│   └── test_system.py
└── src/                   # Library modules
    ├── api_service.py
    ├── cnn_encoder.py
    ├── svm_classifier.py
    ├── linguistic_features.py
    ├── retrieval.py
    ├── cache_manager.py
    └── config.py
```

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

# GPU support (CUDA 12.1, NVIDIA driver >= 555)
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

### 2. Train the Model

Open `step1_2.ipynb` in VS Code or Jupyter and run the cells in order:
- **Cell 1** — environment setup
- **Cell 3** — preprocesses `data/trademark_file.csv`, generates features, saves processed CSV
- **Cell 4** — trains CNN encoder + SVM, saves models to `models/`

See [STEP1_2_GUIDE.md](STEP1_2_GUIDE.md) for a full walkthrough.

### 3. Command-Line Inference

```bash
# Basic similarity check
python scripts/inference.py --mark1 "SuperCoffee" --mark2 "Super Coffee"

# With detailed explanation
python scripts/inference.py --mark1 "TechSmart" --mark2 "SmartTech" --details

# JSON output
python scripts/inference.py --mark1 "BrandA" --mark2 "BrandB" --json --details
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

## � Module Overview

## 🔧 Module Overview

```
src/
├── config.py              # Configuration management
├── cache_manager.py       # Caching system with TTL
├── cnn_encoder.py         # CNN embedding extraction
├── linguistic_features.py # 14+ linguistic features + synonyms/antonyms
├── svm_classifier.py      # Hybrid CNN+SVM classifier
├── retrieval.py           # Candidate retrieval system
└── api_service.py         # FastAPI REST API

scripts/
├── inference.py           # CLI inference script
├── examples.py            # Usage examples
└── test_system.py         # Module validation tests

step1_2.ipynb              # Main training pipeline notebook
requirements.txt           # Pinned dependencies
STEP1_2_GUIDE.md           # Pipeline walkthrough
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
