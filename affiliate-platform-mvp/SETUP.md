# Affiliate Product Analysis Platform - Setup Guide

## Prerequisites

- Node.js 18+
- Python 3.10+
- Docker & Docker Compose (optional)
- Git

## Quick Start (Local Development)

### 1. Clone & Setup Project Structure

```bash
cd affiliate-platform-mvp
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy .env file
cp .env.example .env
# Edit .env with your API keys

# Run backend
python main.py
# Backend will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Setup environment
cp .env.example .env.local
# Edit .env.local if needed

# Run development server
npm run dev
# Frontend will be available at http://localhost:3000
```

## Docker Setup (Recommended)

```bash
# Build and start all services
docker-compose up -d

# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Database: localhost:5432
# Redis: localhost:6379

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down
```

## Project Structure

```
affiliate-platform-mvp/
├── frontend/      - Next.js web application
├── backend/       - FastAPI Python API
├── shared/        - Shared utilities
├── docs/          - Documentation
└── README.md
```

## Available Scripts

### Frontend
```bash
npm run dev       # Start development server
npm run build     # Build for production
npm run start     # Start production server
npm run lint      # Run ESLint
npm run type-check # TypeScript type checking
```

### Backend
```bash
python main.py                  # Run development server
uvicorn main:app --reload       # With auto-reload
python -m pytest               # Run tests
alembic upgrade head           # Run migrations
```

## API Documentation

After starting the backend, visit: **http://localhost:8000/docs**

This provides an interactive Swagger UI with all available endpoints.

## Environment Variables

### Backend (.env)
```
DATABASE_URL=sqlite:///./affiliate.db
OPENAI_API_KEY=your_key_here
SHOPEE_API_KEY=your_key_here
LAZADA_API_KEY=your_key_here
TIKTOK_SHOP_API_KEY=your_key_here
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
```

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Development Workflow

1. Backend changes: Changes auto-reload with `--reload` flag
2. Frontend changes: Changes auto-reload with Next.js dev server
3. Database changes: Create migration, then run `alembic upgrade head`

## Troubleshooting

### Port already in use
```bash
# Find process using port
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>  # Mac/Linux
taskkill /PID <PID> /F  # Windows
```

### API connection issues
- Check backend is running: `curl http://localhost:8000/health`
- Check CORS settings in backend
- Check frontend .env.local has correct API_URL

### Database errors
- Reset database: Delete `affiliate.db` or drop PostgreSQL database
- Run migrations: `alembic upgrade head`

## Next Steps

1. Install dependencies for both frontend and backend
2. Setup environment variables
3. Start backend: `python main.py`
4. Start frontend: `npm run dev`
5. Open http://localhost:3000 in browser
6. Check backend API docs at http://localhost:8000/docs

For detailed architecture info, see [ARCHITECTURE.md](./ARCHITECTURE.md)
