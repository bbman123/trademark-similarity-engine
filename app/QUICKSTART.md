# 🚀 Quick Start Guide

## Prerequisites Check
- [ ] Python 3.11+ installed
- [ ] Node.js 18+ installed
- [ ] 4GB+ RAM available
- [ ] 2GB+ free disk space

## Setup (5 minutes)

### Step 1: Backend Setup
```powershell
# Navigate to backend
cd app/backend

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start server
python api.py
```

✅ Backend running at http://localhost:8000

### Step 2: Frontend Setup (New Terminal)
```powershell
# Navigate to frontend
cd app/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

✅ Frontend running at http://localhost:3000

### Step 3: Test
Open browser: http://localhost:3000

Try example:
- Mark 1: `SuperCoffee`
- Mark 2: `Super Coffee`
- Click "Check Similarity"

## Using Startup Scripts

### Windows (PowerShell)
```powershell
cd app/backend
.\start.ps1
```

### Windows (CMD)
```cmd
cd app\backend
start.bat
```

### Linux/Mac
```bash
cd app/backend
chmod +x start.sh
./start.sh
```

## Common Issues

**Backend won't start?**
- Verify models exist in `backend/models/`
- Check Python version: `python --version`
- Reinstall: `pip install -r requirements.txt`

**Frontend errors?**
- Delete `node_modules` and reinstall: `npm install`
- Check Node version: `node --version`
- Clear cache: `npm cache clean --force`

**Port already in use?**
```powershell
# Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## Features to Try

1. **Single Check**: Compare two trademarks
2. **Detailed Analysis**: Enable "Include details" for insights
3. **Batch Check**: Load examples and check multiple pairs
4. **Export Results**: Download CSV from batch results

## Next Steps

- Read full [README.md](README.md)
- Explore [API Documentation](http://localhost:8000/docs)
- Check example pairs in the UI
- Try batch processing

## Support

Having issues? Check:
1. Backend logs: `backend/api.log`
2. Browser console (F12)
3. API health: http://localhost:8000/health
4. Full documentation in README.md
