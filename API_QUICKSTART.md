# 🚀 Trademark Similarity Engine - Quick Start Guide

## 📋 Prerequisites

1. **Trained Models**: Run the training notebook first
   ```
   Open: Non_hybrid.ipynb
   Run: Cell 4 (Complete training pipeline)
   ```

2. **Python Environment**: Virtual environment with dependencies
   ```powershell
   .\.venv\Scripts\activate
   ```

## ⚡ Starting the API

### Method 1: Using the Script (Recommended)

```powershell
.\start_api.ps1
```

### Method 2: Manual Start

```powershell
# Activate environment
.\.venv\Scripts\activate

# Start server
python api_server.py
```

## 🎯 Testing the API

### 1. Quick Health Check

Open browser: http://localhost:8000/health

Or use curl:
```bash
curl http://localhost:8000/health
```

### 2. Interactive Documentation

Open browser: http://localhost:8000/docs

### 3. Run Test Suite

```powershell
python test_api.py
```

## 📝 Simple Examples

### Example 1: Check Similarity (cURL)

```bash
curl -X POST "http://localhost:8000/similarity-check" \
  -H "Content-Type: application/json" \
  -d "{\"mark1\": \"SuperCoffee\", \"mark2\": \"Super Coffee\"}"
```

### Example 2: Python Script

```python
import requests

response = requests.post(
    "http://localhost:8000/similarity-check",
    json={
        "mark1": "TechSmart",
        "mark2": "SmartTech",
        "include_details": True
    }
)

result = response.json()
print(f"Result: {result['label_text']}")
print(f"Probability: {result['probability']:.1%}")
print(f"Risk: {result['risk_level']}")
```

### Example 3: Batch Processing

```python
import requests

pairs = [
    {"mark1": "Nike", "mark2": "Mike"},
    {"mark1": "Apple", "mark2": "Orange"},
    {"mark1": "CoffeePlus", "mark2": "PlusCoffee"}
]

response = requests.post(
    "http://localhost:8000/batch-similarity",
    json={"pairs": pairs}
)

for i, result in enumerate(response.json(), 1):
    print(f"Pair {i}: {result['label_text']} ({result['risk_level']} risk)")
```

## 📊 Expected Response

```json
{
  "label": 1,
  "label_text": "Similar",
  "probability": 0.8745,
  "confidence": 0.8745,
  "risk_level": "HIGH",
  "recommendation": "High risk of confusion - Consider alternative trademark",
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
    }
  }
}
```

## 🔍 Risk Levels

| Level | Probability | Meaning |
|-------|-------------|---------|
| **HIGH** | ≥ 0.7 | Likely trademark conflict |
| **MEDIUM** | 0.5 - 0.7 | Possible conflict, review recommended |
| **LOW** | < 0.5 | Unlikely conflict |

## 🛠️ Troubleshooting

### Issue: "Models not loaded"

**Solution**: 
1. Ensure models are trained (run Non_hybrid.ipynb Cell 4)
2. Check that these files exist:
   - `models/cnn_encoder.keras`
   - `models/cnn_encoder_tokenizer.pkl`
   - `models/hybrid_svm.pkl`

### Issue: Port 8000 already in use

**Solution**:
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or change port in api_server.py (line ~678)
```

### Issue: Slow predictions

**Solution**:
- First prediction is slower (model loading)
- Subsequent predictions are fast (~80ms)
- Use batch endpoint for multiple pairs

## 📚 Full Documentation

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete API reference.

## 🎓 Model Performance

- **Accuracy**: 95.43%
- **F1 Score**: 95.53%
- **ROC-AUC**: 98.76%
- **Precision**: 96.21%
- **Recall**: 94.87%

## 📞 Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/similarity-check` | POST/GET | Check trademark similarity |
| `/batch-similarity` | POST | Batch processing |
| `/features` | GET | Extract features |
| `/stats` | GET | API statistics |
| `/docs` | GET | Interactive documentation |

## 🚦 Next Steps

1. ✅ Start the API: `.\start_api.ps1`
2. ✅ Test it: `python test_api.py`
3. ✅ Read docs: http://localhost:8000/docs
4. ✅ Integrate into your application!

---

**Built with FastAPI, TensorFlow, and scikit-learn** ⚡
