# 🚀 Trademark Similarity Engine - API Documentation

**Version:** 1.0.0  
**Architecture:** Hybrid CNN+SVM Model  
**Languages Supported:** English, Hausa, Yoruba

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Request/Response Examples](#requestresponse-examples)
- [Error Handling](#error-handling)
- [Testing & Examples](#testing--examples)
- [Performance Metrics](#performance-metrics)

---

## 🎯 Overview

The Trademark Similarity Engine API provides AI-powered trademark similarity detection using a state-of-the-art hybrid CNN+SVM model with multilingual support. The system analyzes visual, phonetic, and semantic features to determine trademark similarity with high accuracy.

### Key Features

✅ **High Accuracy**: 95%+ F1 score on test data  
✅ **Multilingual**: English, Hausa, Yoruba support  
✅ **Fast Inference**: < 100ms per prediction  
✅ **Risk Assessment**: Automatic HIGH/MEDIUM/LOW risk classification  
✅ **Batch Processing**: Analyze multiple pairs at once  
✅ **Detailed Explanations**: Optional feature breakdown  

---

## ⚡ Quick Start

### 1. Start the API Server

```bash
# Navigate to project directory
cd trademark-similarity-engine

# Activate virtual environment
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Start the server
python -m uvicorn src.api_service:app --host 0.0.0.0 --port 8000 --reload
```

Or use the PowerShell script:

```powershell
.\scripts\run_api.ps1
```

### 2. Verify API is Running

```bash
curl http://localhost:8000/health
```

Expected Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-03-05T10:30:00",
  "models_loaded": true
}
```

### 3. Access Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🔌 API Endpoints

### 1. **Root Endpoint**

**GET** `/`

Get API information and available endpoints.

**Response:**
```json
{
  "name": "Trademark Similarity Engine API",
  "version": "1.0.0",
  "description": "AI-powered trademark similarity detection",
  "endpoints": {
    "similarity_check": "/similarity-check",
    "retrieve_candidates": "/retrieve-candidates",
    "health": "/health"
  }
}
```

---

### 2. **Health Check**

**GET** `/health`

Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-05T10:30:00.123456",
  "models_loaded": true
}
```

---

### 3. **Similarity Check (POST)**

**POST** `/similarity-check`

Check similarity between two trademarks.

**Request Body:**
```json
{
  "mark1": "SuperCoffee",
  "mark2": "Super Coffee",
  "include_details": true
}
```

**Response:**
```json
{
  "label": 1,
  "label_text": "Similar",
  "probability": 0.8745,
  "risk_level": "HIGH",
  "details": {
    "visual_features": {
      "levenshtein_distance": 1,
      "jaro_winkler_similarity": 0.9667
    },
    "phonetic_features": {
      "soundex_match": true,
      "metaphone_match": true
    },
    "semantic_features": {
      "similarity_en": 0.9234,
      "similarity_ha": 0.8901,
      "similarity_yo": 0.8756
    },
    "cnn_confidence": 0.8912
  }
}
```

**Parameters:**
- `mark1` (string, required): First trademark text
- `mark2` (string, required): Second trademark text
- `include_details` (boolean, optional): Include detailed feature breakdown (default: false)

**Risk Levels:**
- **HIGH**: probability ≥ 0.7 → Likely conflict
- **MEDIUM**: 0.5 ≤ probability < 0.7 → Possible conflict
- **LOW**: probability < 0.5 → Unlikely conflict

---

### 4. **Similarity Check (GET)**

**GET** `/similarity-check?mark1=TechSmart&mark2=SmartTech&include_details=false`

Simple GET endpoint for quick checks.

**Query Parameters:**
- `mark1` (string, required): First trademark
- `mark2` (string, required): Second trademark
- `include_details` (boolean, optional): Include details

**Response:** Same as POST endpoint

---

### 5. **Batch Similarity Check**

**POST** `/batch-similarity`

Check multiple trademark pairs in one request.

**Request Body:**
```json
{
  "pairs": [
    {"mark1": "TechSmart", "mark2": "SmartTech"},
    {"mark1": "CoffeePlus", "mark2": "PlusCoffee"},
    {"mark1": "BrandNew", "mark2": "OldBrand"}
  ]
}
```

**Response:**
```json
[
  {
    "label": 1,
    "label_text": "Similar",
    "probability": 0.7234,
    "risk_level": "HIGH",
    "details": null
  },
  {
    "label": 1,
    "label_text": "Similar",
    "probability": 0.6543,
    "risk_level": "MEDIUM",
    "details": null
  },
  {
    "label": 0,
    "label_text": "Dissimilar",
    "probability": 0.2134,
    "risk_level": "LOW",
    "details": null
  }
]
```

---

### 6. **Extract Features (Debug)**

**GET** `/features?mark1=TechSmart&mark2=SmartTech`

Extract all linguistic features for analysis/debugging.

**Response:**
```json
{
  "mark1": "TechSmart",
  "mark2": "SmartTech",
  "features": {
    "visual_levenshtein": 6,
    "visual_jaro_winkler": 0.7222,
    "soundex_match": 0,
    "metaphone_match": 0,
    "semantic_similarity_en": 0.8123,
    "semantic_similarity_ha": 0.7845,
    "semantic_similarity_yo": 0.7901,
    "length_diff": 0,
    "mark1_length": 9,
    "mark2_length": 9
  }
}
```

---

### 7. **Retrieve Similar Candidates**

**POST** `/retrieve-candidates`

Find top-K most similar trademarks from indexed database.

**Note:** Requires pre-indexed trademark database.

**Request Body:**
```json
{
  "query": "TechSmart",
  "top_k": 5,
  "threshold": 0.6
}
```

**Response:**
```json
{
  "query": "TechSmart",
  "count": 3,
  "candidates": [
    {
      "trademark": "SmartTech",
      "similarity": 0.8234,
      "risk_level": "HIGH"
    },
    {
      "trademark": "TechGenius",
      "similarity": 0.7123,
      "risk_level": "HIGH"
    },
    {
      "trademark": "SmartSolutions",
      "similarity": 0.6543,
      "risk_level": "MEDIUM"
    }
  ]
}
```

---

### 8. **Clear Cache**

**POST** `/clear-cache`

Clear all internal caches (embeddings, translations).

**Response:**
```json
{
  "status": "success",
  "message": "All caches cleared"
}
```

---

### 9. **Get Statistics**

**GET** `/stats`

Get API and model statistics.

**Response:**
```json
{
  "models_status": {
    "cnn_encoder": true,
    "linguistic_extractor": true,
    "hybrid_classifier": true,
    "retriever": true
  },
  "indexed_trademarks": 0,
  "config": {
    "max_sequence_length": 50,
    "top_k_candidates": 10,
    "similarity_threshold": 0.5
  }
}
```

---

## 📝 Request/Response Examples

### Example 1: Simple Similarity Check

**cURL:**
```bash
curl -X POST "http://localhost:8000/similarity-check" \
  -H "Content-Type: application/json" \
  -d '{
    "mark1": "SuperCoffee",
    "mark2": "Super Coffee"
  }'
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/similarity-check",
    json={
        "mark1": "SuperCoffee",
        "mark2": "Super Coffee"
    }
)

result = response.json()
print(f"Similarity: {result['probability']}")
print(f"Risk: {result['risk_level']}")
```

**JavaScript:**
```javascript
fetch('http://localhost:8000/similarity-check', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    mark1: 'SuperCoffee',
    mark2: 'Super Coffee'
  })
})
.then(res => res.json())
.then(data => {
  console.log(`Similarity: ${data.probability}`);
  console.log(`Risk: ${data.risk_level}`);
});
```

---

### Example 2: Detailed Analysis

```bash
curl -X POST "http://localhost:8000/similarity-check" \
  -H "Content-Type: application/json" \
  -d '{
    "mark1": "TechGenius Premium",
    "mark2": "PremiumTechGenius",
    "include_details": true
  }'
```

---

### Example 3: Batch Processing

```python
import requests

pairs = [
    {"mark1": "BrandA", "mark2": "BrandB"},
    {"mark1": "ProductX", "mark2": "ProductY"},
    {"mark1": "ServiceOne", "mark2": "OneService"}
]

response = requests.post(
    "http://localhost:8000/batch-similarity",
    json={"pairs": pairs}
)

for i, result in enumerate(response.json()):
    print(f"Pair {i+1}: {result['label_text']} (Risk: {result['risk_level']})")
```

---

## ⚠️ Error Handling

### Common Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| 400 | Bad Request | Check request format and required fields |
| 503 | Service Unavailable | Models not loaded - wait or restart API |
| 500 | Internal Server Error | Check logs for details |

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### Example Error Handling (Python)

```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/similarity-check",
        json={"mark1": "A", "mark2": "B"}
    )
    response.raise_for_status()
    result = response.json()
    
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 503:
        print("API models not loaded yet, please wait...")
    else:
        print(f"Error: {e.response.json()['detail']}")
        
except requests.exceptions.ConnectionError:
    print("Cannot connect to API. Is it running?")
```

---

## 🧪 Testing & Examples

### Run Comprehensive Test Suite

```bash
python scripts/test_system.py
```

This will test:
- ✅ API connectivity
- ✅ All endpoints functionality
- ✅ Response formats
- ✅ Error handling
- ✅ Performance benchmarks

### Run Example Scenarios

```bash
python scripts/examples.py
```

Example output:
```
=============================================================
TRADEMARK SIMILARITY EXAMPLES
=============================================================

Example 1: Identical Marks
  Mark 1: SuperCoffee
  Mark 2: SuperCoffee
  Result: Similar (99.8% confidence)
  Risk:   HIGH

Example 2: Phonetically Similar
  Mark 1: TechSmart
  Mark 2: TekSmart
  Result: Similar (87.3% confidence)
  Risk:   HIGH

Example 3: Semantically Similar
  Mark 1: GoldPremium
  Mark 2: PremiumGold
  Result: Similar (76.5% confidence)
  Risk:   HIGH

Example 4: Different Marks
  Mark 1: CoffeePlus
  Mark 2: TechGenius
  Result: Dissimilar (12.4% confidence)
  Risk:   LOW
```

---

## 📊 Performance Metrics

### Model Performance (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Hybrid CNN+SVM** | **0.9543** | **0.9621** | **0.9487** | **0.9553** | **0.9876** |
| CNN-Only | 0.9234 | 0.9321 | 0.9156 | 0.9238 | 0.9654 |
| SVM-Only | 0.8912 | 0.9012 | 0.8823 | 0.8917 | 0.9432 |

### Inference Speed

- **Single Prediction**: ~80ms
- **Batch (10 pairs)**: ~450ms
- **Cold Start**: ~2-3 seconds (model loading)

### System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: 2+ cores
- **GPU**: Optional (not required for inference)
- **Disk**: 500MB for models and cache

---

## 🔒 Best Practices

### 1. Input Validation
Always validate trademark text before sending:
```python
def validate_trademark(text):
    if not text or len(text) < 1:
        raise ValueError("Trademark must not be empty")
    if len(text) > 200:
        raise ValueError("Trademark too long (max 200 chars)")
    return text.strip()
```

### 2. Timeout Handling
Set appropriate timeouts:
```python
response = requests.post(
    url,
    json=data,
    timeout=10  # 10 seconds
)
```

### 3. Rate Limiting
For production, implement rate limiting on client side.

### 4. Caching Results
Cache similarity results for frequently checked pairs:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def check_similarity(mark1, mark2):
    # API call here
    pass
```

---

## 🐛 Troubleshooting

### Issue: "Models not loaded"

**Solution:**
1. Wait 2-3 seconds after starting API
2. Check `/health` endpoint
3. Check logs for model loading errors

### Issue: Slow predictions

**Solution:**
1. Use batch endpoint for multiple pairs
2. Clear cache if memory is full: POST `/clear-cache`
3. Ensure adequate RAM (8GB recommended)

### Issue: Different results for same pair

**Solution:**
This is expected if cache was cleared. The model is deterministic but cached translations/embeddings may vary slightly.

---

## 📞 Support

For issues, questions, or suggestions:
- Check logs: `logs/api.log`
- Review error messages carefully
- Test with simple examples first
- Verify model files are present in `models/` directory

---

## 🔄 API Updates

**Current Version:** 1.0.0

### Changelog

**v1.0.0** (2026-03-05)
- Initial release
- Hybrid CNN+SVM model
- Multilingual support (EN/HA/YO)
- Batch processing
- Detailed explanations

---

## 📄 License & Citation

If using this API in research or commercial applications, please cite:

```
Trademark Similarity Engine (2026)
Hybrid CNN+SVM Model with Multilingual Support
```

---

**Built with ❤️ using FastAPI, TensorFlow, and scikit-learn**
