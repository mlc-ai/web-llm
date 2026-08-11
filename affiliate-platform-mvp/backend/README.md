# Affiliate Product Analysis Platform - Backend

Backend API dùng FastAPI cho:
- Image search & recognition
- Product review analysis
- Affiliate link management
- Trend analysis
- LLM integration

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Migrations
alembic upgrade head

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Image Analysis
- POST `/api/v1/products/search-image` - Image reverse search

### Reviews
- GET `/api/v1/reviews/{product_id}` - Get reviews
- POST `/api/v1/reviews/{product_id}/analyze` - Analyze reviews with LLM

### Trends
- GET `/api/v1/trends/daily` - Daily trends
- POST `/api/v1/trends/analyze` - Analyze trend data

### Affiliate
- POST `/api/v1/affiliate/sync` - Sync affiliate products
