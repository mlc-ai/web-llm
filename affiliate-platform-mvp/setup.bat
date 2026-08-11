@echo off
echo 🚀 Setting up Affiliate Product Analysis Platform...

REM Backend setup
echo 📦 Setting up backend...
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
echo ✅ Backend setup complete

REM Frontend setup
echo 📦 Setting up frontend...
cd ..\frontend
call npm install
copy .env.example .env.local
echo ✅ Frontend setup complete

echo.
echo ✨ Setup complete!
echo.
echo Next steps:
echo 1. Backend: cd backend ^&^& venv\Scripts\activate ^&^& python main.py
echo 2. Frontend: cd frontend ^&^& npm run dev
echo.
echo Frontend: http://localhost:3000
echo Backend API: http://localhost:8000/docs
