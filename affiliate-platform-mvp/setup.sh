#!/bin/bash

echo "🚀 Setting up Affiliate Product Analysis Platform..."

# Backend setup
echo "📦 Setting up backend..."
cd backend
python -m venv venv
source venv/bin/activate || source venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
echo "✅ Backend setup complete"

# Frontend setup
echo "📦 Setting up frontend..."
cd ../frontend
npm install
cp .env.example .env.local
echo "✅ Frontend setup complete"

echo ""
echo "✨ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Backend: cd backend && source venv/bin/activate && python main.py"
echo "2. Frontend: cd frontend && npm run dev"
echo ""
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000/docs"
