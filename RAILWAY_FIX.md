# Railway Deployment Fix

## 🚨 Problem
Railway is trying to build from the root directory which contains both `dashboard/` and `pipeline/` folders, causing confusion.

## ✅ Solution Options

### Option 1: Deploy from pipeline directory only (Recommended)

```bash
# 1. Navigate to pipeline directory
cd pipeline

# 2. Initialize a new git repository for just the pipeline
git init
git add .
git commit -m "Initial backend deployment"

# 3. Deploy to Railway from this directory
railway login
railway up

# 4. Set environment variables in Railway dashboard:
# - DB_HOST
# - DB_NAME  
# - DB_USER
# - DB_PASS
# - DB_PORT=5432
```

### Option 2: Use Railway's root directory settings

If you want to deploy from the root directory:

1. Go to your Railway project dashboard
2. Go to Settings → Environment
3. Add this environment variable:
   - `RAILWAY_ROOT_DIR` = `pipeline`

### Option 3: GitHub integration with path

1. Connect Railway to your GitHub repo
2. In Railway project settings:
   - Set **Root Directory** to `pipeline`
   - Set **Build Command** to `pip install -r requirements.txt`
   - Set **Start Command** to `uvicorn main:app --host 0.0.0.0 --port $PORT`

## 🧪 Testing the deployment

After deployment:

```bash
# Test your Railway API
curl https://your-railway-url.railway.app/health

# Should return:
# {"status": "healthy", "timestamp": "...", ...}
```

## 🔧 Environment Variables Needed

Set these in Railway dashboard:

```
DB_HOST=db.harespppdrswapdocdmv.supabase.co
DB_NAME=postgres
DB_USER=postgres
DB_PASS=your-supabase-password
DB_PORT=5432
```

## 📋 Quick Fix Commands

```bash
# Quick fix: Deploy just the pipeline
cd pipeline
git init
git add .
git commit -m "Backend for Railway"
railway login
railway up
```

## 🎯 Expected Result

After successful deployment:
- Railway URL: `https://your-app.railway.app`
- Health check: `https://your-app.railway.app/health` ✅
- Refresh endpoint: `https://your-app.railway.app/refresh-data` ✅
