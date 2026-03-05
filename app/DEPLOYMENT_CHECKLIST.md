# 📦 Deployment Checklist

Use this checklist when sharing the `app` folder with others.

## ✅ Pre-Deployment Verification

### Backend Files
- [ ] `backend/api.py` - Main API server
- [ ] `backend/requirements.txt` - Python dependencies
- [ ] `backend/models/cnn_encoder.keras` - CNN model
- [ ] `backend/models/cnn_encoder_tokenizer.pkl` - Tokenizer
- [ ] `backend/models/hybrid_svm.pkl` - SVM model
- [ ] `backend/start.ps1` - Windows PowerShell startup script
- [ ] `backend/start.bat` - Windows CMD startup script
- [ ] `backend/start.sh` - Linux/Mac startup script
- [ ] `backend/.gitignore` - Git ignore rules

### Frontend Files
- [ ] `frontend/package.json` - Node dependencies
- [ ] `frontend/vite.config.js` - Vite configuration
- [ ] `frontend/tailwind.config.js` - Tailwind configuration
- [ ] `frontend/postcss.config.js` - PostCSS configuration
- [ ] `frontend/index.html` - HTML entry point
- [ ] `frontend/src/main.jsx` - React entry point
- [ ] `frontend/src/App.jsx` - Main application
- [ ] `frontend/src/index.css` - Global styles
- [ ] `frontend/src/components/*.jsx` - All React components
- [ ] `frontend/.gitignore` - Git ignore rules

### Documentation
- [ ] `README.md` - Full documentation
- [ ] `QUICKSTART.md` - Quick start guide
- [ ] `DEPLOYMENT_CHECKLIST.md` - This file

## 🚀 Deployment Steps

### For Recipients

1. **Extract the `app` folder** to a location of your choice

2. **Verify Prerequisites**
   ```powershell
   # Check Python
   python --version  # Should be 3.11+
   
   # Check Node.js
   node --version    # Should be 18+
   ```

3. **Backend Setup** (First Terminal)
   ```powershell
   cd app/backend
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows
   pip install -r requirements.txt
   python api.py
   ```

4. **Frontend Setup** (Second Terminal)
   ```powershell
   cd app/frontend
   npm install
   npm run dev
   ```

5. **Access Application**
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs

## 📋 Pre-Share Checklist

Before sharing the `app` folder:

- [ ] Test backend starts successfully
- [ ] Test frontend builds without errors
- [ ] Verify all model files are present
- [ ] Check file sizes are reasonable
- [ ] Test a similarity check end-to-end
- [ ] Remove any sensitive data/credentials
- [ ] Include this checklist
- [ ] Include README.md and QUICKSTART.md

## 💾 Compression

### Recommended Compression

```powershell
# Create a ZIP file (Windows)
Compress-Archive -Path app -DestinationPath trademark-similarity-engine.zip

# Create a tar.gz file (Linux/Mac)
tar -czf trademark-similarity-engine.tar.gz app/
```

### Expected Size
- **Uncompressed**: ~500MB (models included)
- **Compressed**: ~200MB

## 🔒 Security Notes

### Before Production Deployment

1. **Update CORS Settings**
   ```python
   # In backend/api.py
   allow_origins=["https://yourdomain.com"]  # Not "*"
   ```

2. **Enable HTTPS**
   - Use reverse proxy (nginx/Apache)
   - Obtain SSL certificate

3. **Set Environment Variables**
   - API keys (if any)
   - Database credentials (if added)

4. **Rate Limiting**
   - Add rate limiting middleware
   - Protect against abuse

5. **Authentication** (Optional)
   - Add API key authentication
   - Implement user authentication

## 🐛 Common Issues & Solutions

### Issue: Models Not Loading
**Solution**: Verify all 3 model files exist in `backend/models/`:
- `cnn_encoder.keras` (~40MB)
- `cnn_encoder_tokenizer.pkl` (~10KB)
- `hybrid_svm.pkl` (~1MB)

### Issue: Port Already in Use
**Solution**:
```powershell
# Find and kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue: npm Install Fails
**Solution**:
```powershell
# Clear cache and retry
npm cache clean --force
rm -rf node_modules
npm install
```

### Issue: Python Dependencies Failed
**Solution**:
```powershell
# Upgrade pip first
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 System Requirements

### Minimum
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 2GB
- **OS**: Windows 10+, Ubuntu 20.04+, macOS 12+

### Recommended
- **CPU**: 4+ cores
- **RAM**: 8GB
- **Storage**: 5GB SSD
- **OS**: Windows 11, Ubuntu 22.04, macOS 13+

## 🎯 Testing Checklist

Before marking deployment as complete:

- [ ] Health check returns "healthy"
- [ ] Single similarity check works
- [ ] Detailed analysis displays correctly
- [ ] Batch processing works (5+ pairs)
- [ ] Results can be exported as CSV
- [ ] UI is responsive on mobile
- [ ] API documentation is accessible
- [ ] No console errors in browser
- [ ] No Python errors in terminal

## 📞 Support

If deployment issues occur:

1. Check `backend/api.log` for errors
2. Review browser console (F12)
3. Verify all prerequisites are met
4. Follow QUICKSTART.md step-by-step
5. Check README.md troubleshooting section

## ✅ Deployment Complete

When all checks pass:

✅ **Backend Running**: http://localhost:8000  
✅ **Frontend Running**: http://localhost:3000  
✅ **Health Check**: Passing  
✅ **Test Query**: Successful  

**Status**: READY FOR USE 🎉

---

**Last Updated**: March 2026  
**Version**: 1.0.0
