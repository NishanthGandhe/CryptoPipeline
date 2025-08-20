#!/bin/bash

# CryptoPipeline Deployment Setup Script

echo "🚀 Setting up CryptoPipeline for deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📋 Pre-deployment Checklist:${NC}"
echo "1. ✅ Supabase database is already set up"
echo "2. ✅ GitHub repository exists"
echo "3. ✅ Environment variables are configured"
echo ""

echo -e "${YELLOW}🔧 Next Steps for Deployment:${NC}"
echo ""

echo -e "${GREEN}1. Setup Vercel (Frontend):${NC}"
echo "   a. Go to https://vercel.com and sign in with GitHub"
echo "   b. Import your repository: CryptoPipeline"
echo "   c. Set root directory to: dashboard"
echo "   d. Add environment variables:"
echo "      - NEXT_PUBLIC_SUPABASE_URL"
echo "      - NEXT_PUBLIC_SUPABASE_ANON_KEY"
echo "      - NEXT_PUBLIC_API_BASE_URL (Railway URL after step 2)"
echo ""

echo -e "${GREEN}2. Setup Railway (Backend):${NC}"
echo "   a. Go to https://railway.app and sign in with GitHub"
echo "   b. Deploy from GitHub repository"
echo "   c. Select the 'pipeline' folder as root"
echo "   d. Add environment variables:"
echo "      - DB_HOST (from Supabase)"
echo "      - DB_NAME (from Supabase)"
echo "      - DB_USER (from Supabase)"
echo "      - DB_PASS (from Supabase)"
echo "      - DB_PORT (from Supabase)"
echo "   e. Copy the Railway URL and add it to Vercel as NEXT_PUBLIC_API_BASE_URL"
echo ""

echo -e "${GREEN}3. Setup GitHub Secrets:${NC}"
echo "   a. Go to GitHub → Settings → Secrets and variables → Actions"
echo "   b. Add the following secrets:"
echo "      Frontend (Vercel):"
echo "      - VERCEL_TOKEN"
echo "      - VERCEL_PROJECT_ID"
echo "      - VERCEL_ORG_ID"
echo "      - NEXT_PUBLIC_SUPABASE_URL"
echo "      - NEXT_PUBLIC_SUPABASE_ANON_KEY"
echo "      - NEXT_PUBLIC_API_BASE_URL"
echo ""
echo "      Backend (Railway):"
echo "      - RAILWAY_TOKEN"
echo "      - RAILWAY_SERVICE"
echo "      - DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT"
echo ""

echo -e "${GREEN}4. Test Deployment:${NC}"
echo "   a. Push changes to main branch"
echo "   b. Check GitHub Actions for successful deployment"
echo "   c. Visit your Vercel URL to test the frontend"
echo "   d. Test the refresh button to ensure backend connectivity"
echo ""

echo -e "${YELLOW}💡 Useful Commands:${NC}"
echo "   • Test FastAPI locally: cd pipeline && python main.py"
echo "   • Test Next.js locally: cd dashboard && npm run dev"
echo "   • Check logs: Railway dashboard → Service → Logs"
echo "   • Check deployment: Vercel dashboard → Project → Functions"
echo ""

echo -e "${BLUE}📊 Expected Costs (Free Tiers):${NC}"
echo "   • Vercel: Free (100GB bandwidth, unlimited personal projects)"
echo "   • Railway: Free ($5 credit monthly, then pay-as-you-go)"
echo "   • Supabase: Free (500MB database, 2 projects)"
echo "   • GitHub Actions: Free (2000 minutes/month)"
echo ""

echo -e "${GREEN}✅ Deployment files created:${NC}"
echo "   • 📄 vercel.json (Vercel configuration)"
echo "   • 📄 railway.toml (Railway configuration)"  
echo "   • 📄 main.py (FastAPI wrapper)"
echo "   • 📄 requirements.txt (updated with FastAPI)"
echo "   • 📄 .github/workflows/deploy-frontend.yml"
echo "   • 📄 .github/workflows/deploy-backend.yml"
echo "   • 📄 DEPLOYMENT.md (detailed guide)"
echo ""

echo -e "${BLUE}🎯 Why This Architecture?${NC}"
echo "   ✅ Vercel: Best for Next.js, excellent performance"
echo "   ✅ Railway: Great for Python, free PostgreSQL"
echo "   ✅ Supabase: Already configured, reliable"
echo "   ✅ GitHub Actions: Automated deployments"
echo "   ✅ All services have generous free tiers"
echo ""

echo -e "${GREEN}🚀 Ready to deploy! Follow the steps above.${NC}"
