# CryptoPipeline

![AI-driven cryptocurrency price prediction dashboard](https://img.shields.io/badge/AI-Crypto%20Forecasting-blue) ![Next.js](https://img.shields.io/badge/Next.js-15.4-black) ![Python](https://img.shields.io/badge/Python-3.13-green) ![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)

An AI-powered cryptocurrency price prediction system that combines machine learning models with real-time data ingestion and an interactive web dashboard. Built with Python for data processing and Next.js for the frontend, deployed with PostgreSQL/Supabase for data storage.

## 🌟 Features

### 🤖 **Advanced Machine Learning Models**
- **XGBoost Enhanced**: Primary model using 14+ technical indicators and price patterns
- **Multivariate Analysis**: For cryptocurrencies with rich data (volume, metrics)
- **Price-Only Models**: Optimized models for cryptocurrencies with limited data availability
- **Fallback Models**: Holt-Winters and naive baseline models for reliability

### 📊 **Interactive Dashboard**
- **Real-time Forecasts**: 1-30 day prediction slider
- **Live Charts**: Price history with 7-day moving averages
- **Model Transparency**: Detailed information about prediction methods and features
- **22+ Cryptocurrencies**: BTC, ETH, XRP, and more
- **Responsive Design**: Works on desktop, tablet, and mobile

### 🔄 **Automated Data Pipeline**
- **Real-time Ingestion**: Automated price data collection from APIs
- **Smart Processing**: ETL pipeline with data validation and transformation
- **ML Training**: Automated model retraining with latest data
- **Database Integration**: PostgreSQL with optimized queries

### 🚀 **Production Ready**
- **One-click Refresh**: Trigger complete data pipeline from the dashboard
- **Error Handling**: Comprehensive error reporting and recovery
- **Scalable Architecture**: Modular design for easy extension
- **Environment Management**: Secure configuration with environment variables

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Data Sources  │    │   Data Pipeline  │    │    Dashboard UI     │
│                 │    │   (Railway)      │    │     (Vercel)        │
│ • Crypto APIs   │───▶│ • ingest.py      │───▶│ • Next.js Frontend  │
│ • Price Feeds   │    │ • transform.sql  │    │ • Interactive Charts│
│ • Volume Data   │    │ • generate_      │    │ • Forecast Slider   │
│                 │    │   insight.py     │    │ • Model Info        │
└─────────────────┘    │ • FastAPI        │    └─────────────────────┘
                       └──────────────────┘               │
                                │                          │
                                ▼                          │
                       ┌──────────────────┐               │
                       │  PostgreSQL DB   │               │
                       │   (Supabase)     │               │
                       │ • price_daily    │◀──────────────┘
                       │ • model_forecasts│
                       │ • insights       │
                       │ • training_runs  │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.13+**
- **Node.js 18+**
- **PostgreSQL** (or Supabase account)
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/NishanthGandhe/CryptoPipeline.git
cd CryptoPipeline
```

### 2. Setup Environment Variables
Create `.env` files in both root and dashboard directories:

**Root `.env`:**
```env
DB_HOST=your-db-host
DB_NAME=your-database
DB_USER=your-username
DB_PASS=your-password
DB_PORT=5432
```

**Dashboard `.env.local`:**
```env
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
```

### 3. Setup Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd pipeline
pip install -r requirements.txt
```

### 4. Setup Database
```bash
# Run database setup (creates tables and views)
python run_etl.py
```

### 5. Initial Data Load
```bash
# Fetch initial cryptocurrency data
python ingest.py

# Generate first forecasts
python generate_insight.py
```

### 6. Start the Dashboard
```bash
cd ../dashboard
npm install
npm run dev
```

### 7. Open the Application
Navigate to [http://localhost:3000](http://localhost:3000) in your browser.

## 📁 Project Structure

```
CryptoPipeline/
├── 📂 pipeline/           # Python data processing backend
│   ├── ingest.py          # Data ingestion from crypto APIs
│   ├── generate_insight.py # ML model training and prediction
│   ├── run_etl.py         # Database setup and ETL operations
│   ├── transform.sql      # SQL transformations and views
│   ├── requirements.txt   # Python dependencies
│   └── ...
├── 📂 dashboard/          # Next.js frontend application
│   ├── src/
│   │   ├── app/
│   │   │   ├── api/       # API routes (refresh endpoint)
│   │   │   ├── components/ # React components
│   │   │   ├── hooks/     # Custom React hooks
│   │   │   └── page.tsx   # Main dashboard page
│   │   └── lib/           # Utilities and types
│   ├── package.json       # Node.js dependencies
│   └── ...
├── .env                   # Environment variables
└── README.md             # This file
```

## 🔧 Configuration

### Machine Learning Models

The system automatically selects the best model for each cryptocurrency:

1. **XGBoost Enhanced** (Primary)
   - Uses 14+ technical indicators
   - Optimized for trend-following
   - 95%+ success rate

2. **Holt-Winters** (Fallback)
   - Statistical time series model
   - Good for stable markets

3. **Naive Baseline** (Last resort)
   - Simple price continuation
   - Conservative approach

### Supported Cryptocurrencies

- **BTC-USD** (Bitcoin) - Full multivariate model
- **ETH-USD** (Ethereum) - Enhanced price-only model
- **XRP-USD, LTC-USD, BCH-USD** - Price-optimized models
- **LINK-USD, ADA-USD, XLM-USD** - Technical indicator models
- **SOL-USD, DOT-USD, DOGE-USD** - And 11 more...

## 🖥️ Usage

### Dashboard Features

1. **Cryptocurrency Selection**: Choose from 22+ supported cryptocurrencies
2. **Forecast Slider**: Adjust prediction timeframe from 1-30 days
3. **Live Charts**: View price history and trends
4. **Model Information**: See which ML model is being used and why
5. **Data Refresh**: One-click pipeline refresh for latest predictions

### Keyboard Shortcuts

- `Ctrl/Cmd + R`: Refresh data pipeline
- `↑↓←→`: Navigate between cryptocurrencies

### API Endpoints

- `GET /api/refresh-data`: Check refresh endpoint status
- `POST /api/refresh-data`: Trigger complete data pipeline refresh

## 🔄 Data Pipeline

### Automated Workflow

1. **Data Ingestion** (`ingest.py`)
   - Fetches latest prices from cryptocurrency APIs
   - Validates and cleans data
   - Stores in PostgreSQL database

2. **Model Training** (`generate_insight.py`)
   - Prepares features and technical indicators
   - Trains XGBoost models with latest data
   - Generates forecasts for 1-30 days
   - Stores predictions with model metadata

3. **Database Operations** (`run_etl.py`)
   - Creates and maintains database schema
   - Runs SQL transformations
   - Creates optimized views for frontend

### Refresh Process

The dashboard refresh button triggers:
1. Fresh data ingestion (5 minutes)
2. ML model retraining (10 minutes)
3. New forecast generation
4. Automatic UI update

## 🛠️ Development

### Adding New Cryptocurrencies

1. Add symbol to the ingestion list in `ingest.py`
2. Ensure data availability in your APIs
3. The system will automatically detect and model the new cryptocurrency

### Extending Models

1. Modify `generate_insight.py` to add new features
2. Update the model selection logic
3. Add model information in `useCryptoData.ts`

### Custom Features

- Add new technical indicators in the feature preparation functions
- Extend the forecast horizon by modifying the prediction loops
- Add new model types in the training pipeline

## 📊 Performance

### Model Accuracy
- **XGBoost Enhanced**: 95%+ selection rate across cryptocurrencies
- **Training Data**: 2+ years of historical data with technical indicators
- **Update Frequency**: Daily at market close
- **Prediction Range**: 1-30 days ahead

### System Performance
- **Data Refresh**: ~15 minutes for complete pipeline
- **Dashboard Load**: <2 seconds for data display
- **Database Queries**: Optimized with proper indexing
- **Model Training**: Parallel processing for multiple cryptocurrencies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Backend development
cd pipeline
pip install -r requirements.txt
python -m pytest  # Run tests

# Frontend development
cd dashboard
npm install
npm run dev
npm run lint
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **XGBoost** for the machine learning framework
- **Next.js** for the React framework
- **Supabase** for the database and real-time features
- **Recharts** for beautiful data visualizations
- **Tailwind CSS** for the styling system

## 📧 Contact

- **GitHub**: [@NishanthGandhe](https://github.com/NishanthGandhe)
- **Project Link**: [https://github.com/NishanthGandhe/CryptoPipeline](https://github.com/NishanthGandhe/CryptoPipeline)

---

⭐ **Star this repository if you found it helpful!**

