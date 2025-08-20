# Deployment Guide

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Frontend      │    │   Backend API    │    │    Database         │
│   (Vercel)      │────│   (Railway)      │────│   (Supabase)        │
│   Next.js App   │    │   Python Pipeline│    │   PostgreSQL        │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

## Frontend: Vercel Deployment

### 1. Prepare for Vercel
Create `vercel.json` in dashboard directory:

```json
{
  "framework": "nextjs",
  "buildCommand": "npm run build",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "env": {
    "NEXT_PUBLIC_SUPABASE_URL": "@supabase_url",
    "NEXT_PUBLIC_SUPABASE_ANON_KEY": "@supabase_anon_key",
    "NEXT_PUBLIC_API_BASE_URL": "@api_base_url"
  }
}
```

### 2. Update API calls to use external backend
In your React components, update API calls:

```typescript
// Before: /api/refresh-data
// After: process.env.NEXT_PUBLIC_API_BASE_URL + '/refresh-data'
```

## Backend: Railway Deployment

### 1. Create railway.toml
```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "python -m uvicorn main:app --host 0.0.0.0 --port $PORT"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

### 2. Convert to FastAPI
Railway works best with a web server. Convert your scripts to a FastAPI app.

## Database: Supabase (Already Set Up)

Your Supabase setup is perfect for this architecture!

## GitHub Actions Setup

### Frontend Deployment
```yaml
name: Deploy Frontend
on:
  push:
    branches: [main]
    paths: ['dashboard/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: vercel/action@v1
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          working-directory: ./dashboard
```

### Backend Deployment
```yaml
name: Deploy Backend
on:
  push:
    branches: [main]
    paths: ['pipeline/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: bsolutions/railway-deploy-action@v1.0.6
        with:
          railway-token: ${{ secrets.RAILWAY_TOKEN }}
          service: ${{ secrets.RAILWAY_SERVICE }}
```
