"""
Main FastAPI application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

load_dotenv()

# Import routers
from api.routes import products, reviews, trends
from api.routes import affiliate_api

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Application starting...")
    yield
    # Shutdown
    print("🛑 Application shutting down...")

app = FastAPI(
    title="Affiliate Product Analysis API",
    description="API for analyzing affiliate products, reviews, and trends",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

# Include routers
app.include_router(products.router, prefix="/api/v1/products", tags=["products"])
app.include_router(reviews.router, prefix="/api/v1/reviews", tags=["reviews"])
app.include_router(trends.router, prefix="/api/v1/trends", tags=["trends"])
app.include_router(affiliate_api.router, prefix="/api/v1/affiliate", tags=["affiliate"])

@app.get("/")
async def root():
    return {
        "message": "Welcome to Affiliate Product Analysis API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
