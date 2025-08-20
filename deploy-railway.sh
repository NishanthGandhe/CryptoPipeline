#!/bin/bash

# Railway Backend Deployment Script
# Deploys only the pipeline directory to Railway

echo "🚀 Railway Backend Deployment"
echo "============================"

# Check current directory
if [[ ! -d "pipeline" ]]; then
    echo "❌ Please run this script from the CryptoPipeline root directory"
    exit 1
fi

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    npm install -g @railway/cli
fi

echo "📍 Moving to pipeline directory..."
cd pipeline

# Check if this is already a git repo
if [[ ! -d ".git" ]]; then
    echo "🔧 Initializing git repository for backend..."
    git init
    git add .
    git commit -m "Initial backend deployment for Railway"
fi

# Check if logged into Railway
if ! railway whoami &> /dev/null; then
    echo "🔐 Please log into Railway:"
    railway login
fi

echo "🚀 Deploying backend to Railway..."
railway up

echo ""
echo "✅ Backend deployment initiated!"
echo ""
echo "📋 Next steps:"
echo "1. Check Railway dashboard for deployment status"
echo "2. Set environment variables in Railway:"
echo "   - DB_HOST=db.harespppdrswapdocdmv.supabase.co"
echo "   - DB_NAME=postgres" 
echo "   - DB_USER=postgres"
echo "   - DB_PASS=your-supabase-password"
echo "   - DB_PORT=5432"
echo "3. Test health endpoint: https://your-app.railway.app/health"
echo "4. Update frontend .env.local with your Railway URL"
echo ""
echo "🔗 Railway Dashboard: https://railway.app/dashboard"
