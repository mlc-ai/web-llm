# Affiliate Product Analysis Platform

A comprehensive platform for analyzing affiliate products using AI/LLM, tracking trends, and optimizing marketing strategies.

## 🎯 Features

✅ **Image Search** - Upload product images to find similar items
✅ **Review Analysis** - AI-powered sentiment analysis of customer reviews  
✅ **Trend Analytics** - Track daily/weekly/monthly trends
✅ **Product Comparison** - Deep compare features and pricing
✅ **Design Insights** - Learn from successful product designs
✅ **Affiliate Integration** - Easy sync with Shopee, Lazada, TikTok Shop

## 🏗️ Architecture

```
frontend/          - Next.js 14 web application
backend/           - FastAPI Python backend
shared/            - Shared types and utilities
docs/              - Documentation
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- pip and npm

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
# API available at http://localhost:8000/docs
```

## 📚 API Documentation

Auto-generated API docs available at: `http://localhost:8000/docs`

### Key Endpoints

**Products**
- `POST /api/v1/products/search-image` - Search by image
- `GET /api/v1/products` - List products
- `POST /api/v1/products/compare` - Compare products

**Reviews**
- `GET /api/v1/reviews/{product_id}` - Get reviews
- `POST /api/v1/reviews/{product_id}/analyze` - Analyze with LLM

**Trends**
- `GET /api/v1/trends/daily` - Daily trends
- `GET /api/v1/trends/weekly` - Weekly trends
- `POST /api/v1/trends/analyze` - Analyze trends

**Affiliate**
- `POST /api/v1/affiliate/sync` - Sync products
- `POST /api/v1/affiliate/generate-link` - Generate links
- `POST /api/v1/affiliate/track-discount` - Track discounts

## 🔧 Configuration

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Backend (.env)
```
DATABASE_URL=sqlite:///./affiliate.db
OPENAI_API_KEY=your_key
SHOPEE_API_KEY=your_key
LAZADA_API_KEY=your_key
TIKTOK_SHOP_API_KEY=your_key
```

## 📦 Dependencies

**Frontend:**
- Next.js 14
- React 18
- Tailwind CSS
- Axios
- Lucide Icons

**Backend:**
- FastAPI
- SQLAlchemy
- OpenAI
- Pillow (Image processing)
- Uvicorn

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License

## 📞 Support

For support, email support@affiliateai.com or open an issue on GitHub.
