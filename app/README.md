# Trademark Similarity Engine

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![React](https://img.shields.io/badge/react-18.2-61dafb.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

**AI-Powered Trademark Similarity Detection with Multilingual Support**

[Features](#features) • [Quick Start](#quick-start) • [API Documentation](#api-documentation) • [License](#license)

</div>

---

## 🌟 Features

- **🎯 High Accuracy**: 95.43% accuracy using Hybrid CNN+SVM architecture
- **🌍 Multilingual Support**: Analyzes English, Hausa, and Yoruba trademarks
- **⚡ Real-time Analysis**: Instant similarity detection with detailed breakdowns
- **📊 Batch Processing**: Check up to 100 trademark pairs simultaneously
- **🎨 Modern UI**: Professional React + Tailwind CSS interface
- **🔍 Detailed Insights**: Visual, phonetic, and semantic feature analysis
- **📈 Risk Assessment**: Three-level risk classification (HIGH/MEDIUM/LOW)
- **🚀 Easy Deployment**: Self-contained application with all dependencies

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 95.43% |
| **F1 Score** | 95.53% |
| **ROC-AUC** | 98.76% |
| **Precision** | 96.21% |
| **Recall** | 94.87% |

## 📋 Prerequisites

- **Python**: 3.11 or higher
- **Node.js**: 18.0 or higher
- **npm**: 9.0 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 2GB free space

## 🚀 Quick Start

### 1. Navigate to App Directory

```powershell
cd app
```

### 2. Backend Setup

```powershell
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
python api.py
```

The API server will start at: **http://localhost:8000**

### 3. Frontend Setup (New Terminal)

```powershell
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will start at: **http://localhost:3000**

### 4. Access the Application

Open your browser and navigate to:
- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 📦 Project Structure

```
app/
├── backend/              # FastAPI backend
│   ├── models/           # Trained ML models
│   │   ├── cnn_encoder.keras
│   │   ├── cnn_encoder_tokenizer.pkl
│   │   └── hybrid_svm.pkl
│   ├── api.py            # Main API server
│   ├── requirements.txt  # Python dependencies
│   └── api.log           # Server logs
│
└── frontend/             # React frontend
    ├── public/           # Static assets
    ├── src/
    │   ├── components/   # React components
    │   │   ├── Header.jsx
    │   │   ├── SimilarityChecker.jsx
    │   │   ├── BatchChecker.jsx
    │   │   ├── ResultCard.jsx
    │   │   ├── AnalysisDetails.jsx
    │   │   └── ApiStatus.jsx
    │   ├── App.jsx       # Main application
    │   ├── main.jsx      # Entry point
    │   └── index.css     # Tailwind styles
    ├── package.json      # Node dependencies
    ├── vite.config.js    # Vite configuration
    └── tailwind.config.js # Tailwind configuration
```

## 🔧 Configuration

### Backend Configuration

Edit `backend/api.py` to customize:

```python
# Server settings
host = "0.0.0.0"  # Listen on all interfaces
port = 8000       # API port

# CORS settings (for production, specify exact origins)
allow_origins = ["*"]  # Change to ["https://yourdomain.com"]
```

### Frontend Configuration

Edit `frontend/vite.config.js` to customize:

```javascript
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',  // Backend URL
    },
  },
}
```

## 📖 API Documentation

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-05T14:39:00",
  "models_loaded": true,
  "model_info": {
    "cnn_loaded": true,
    "svm_loaded": true,
    "semantic_loaded": true,
    "vocab_size": 98
  }
}
```

#### 2. Single Similarity Check
```http
POST /similarity-check
Content-Type: application/json

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
  "probability": 0.9876,
  "confidence": 0.9876,
  "risk_level": "HIGH",
  "recommendation": "High risk of confusion - Consider alternative",
  "details": {
    "visual_features": {...},
    "phonetic_features": {...},
    "semantic_features": {...}
  }
}
```

#### 3. Batch Similarity Check
```http
POST /batch-similarity
Content-Type: application/json

{
  "pairs": [
    {"mark1": "Nike", "mark2": "Mike"},
    {"mark1": "Apple", "mark2": "Orange"}
  ]
}
```

**Full API documentation available at**: http://localhost:8000/docs

## 🎨 UI Features

### Single Check Mode
- **Input Validation**: Real-time character count and validation
- **Example Pairs**: Quick-load common test cases
- **Detailed Analysis**: Toggle feature breakdown
- **Visual Indicators**: Color-coded risk levels
- **Progress Tracking**: Confidence meters and progress bars

### Batch Check Mode
- **Dynamic Pair Management**: Add/remove pairs on the fly
- **Bulk Operations**: Process up to 100 pairs
- **Export Results**: Download results as CSV
- **Summary Statistics**: Quick overview of risk distribution
- **Interactive Table**: Sortable results with visual indicators

## 🔬 Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework
- **TensorFlow/Keras**: Deep learning models
- **scikit-learn**: ML algorithms and preprocessing
- **sentence-transformers**: Multilingual semantic embeddings
- **jellyfish**: Phonetic similarity algorithms

### Frontend
- **React 18**: Modern UI library
- **Vite**: Next-generation build tool
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide React**: Beautiful icon library
- **Axios**: HTTP client

## 🚢 Production Deployment

### Backend (API Server)

1. **Use Production WSGI Server**:
```powershell
pip install gunicorn  # Linux/Mac
pip install waitress  # Windows

# Run with Waitress (Windows)
waitress-serve --host=0.0.0.0 --port=8000 api:app

# Run with Gunicorn (Linux/Mac)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app
```

2. **Set Environment Variables**:
```powershell
$env:PYTHONUNBUFFERED="1"
$env:TF_CPP_MIN_LOG_LEVEL="2"  # Reduce TensorFlow logs
```

### Frontend

1. **Build for Production**:
```powershell
cd frontend
npm run build
```

2. **Serve Build**:
```powershell
npm run preview
# Or use a static server like nginx/apache
```

### Docker Deployment (Optional)

Create `Dockerfile` in backend:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "api.py"]
```

## 🐛 Troubleshooting

### Backend Issues

**Issue**: Models not loading
```powershell
# Check model files exist
ls backend/models/
# Should show: cnn_encoder.keras, cnn_encoder_tokenizer.pkl, hybrid_svm.pkl
```

**Issue**: Port already in use
```powershell
# Find process using port 8000
netstat -ano | findstr :8000
# Kill process
taskkill /PID <PID> /F
```

### Frontend Issues

**Issue**: API connection errors
- Ensure backend is running on port 8000
- Check CORS settings in backend
- Verify proxy configuration in `vite.config.js`

**Issue**: Build errors
```powershell
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## 📊 Performance Tips

1. **Backend Optimization**:
   - Use GPU for faster inference (if available)
   - Enable model caching
   - Increase worker processes for production

2. **Frontend Optimization**:
   - Use production build for deployment
   - Enable code splitting
   - Optimize images and assets

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Model Architecture**: Hybrid CNN+SVM for trademark similarity detection
- **Multilingual Support**: paraphrase-multilingual-MiniLM-L12-v2
- **UI Design**: Inspired by modern web applications
- **Icons**: Lucide React icon library

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review the troubleshooting section

---

<div align="center">

**Built with ❤️ using Python, TensorFlow, React, and Tailwind CSS**

[⬆ Back to Top](#trademark-similarity-engine)

</div>
