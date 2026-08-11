# Project Structure

```
affiliate-platform-mvp/
├── frontend/                    # Next.js web application
│   ├── src/
│   │   ├── app/                # Next.js app directory
│   │   │   ├── layout.tsx      # Root layout
│   │   │   └── page.tsx        # Home page
│   │   ├── components/         # React components
│   │   │   ├── Navbar.tsx
│   │   │   ├── Hero.tsx
│   │   │   ├── Features.tsx
│   │   │   ├── Dashboard.tsx
│   │   │   └── Footer.tsx
│   │   ├── hooks/              # Custom React hooks
│   │   ├── utils/              # Utility functions
│   │   │   ├── api.config.ts   # API endpoint configuration
│   │   │   └── http.ts         # HTTP client
│   │   ├── types/              # TypeScript types
│   │   ├── styles/             # CSS files
│   │   └── public/             # Static assets
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── next.config.js
│   └── .env.local
│
├── backend/                     # FastAPI Python backend
│   ├── main.py                 # Application entry point
│   ├── config.py               # Configuration
│   ├── api/
│   │   ├── routes/             # API route handlers
│   │   │   ├── products.py     # Product endpoints
│   │   │   ├── reviews.py      # Review endpoints
│   │   │   ├── trends.py       # Trend endpoints
│   │   │   └── affiliate_api.py # Affiliate endpoints
│   │   └── __init__.py
│   ├── models/                 # Database models (to be implemented)
│   ├── services/               # Business logic services (to be implemented)
│   │   ├── image_service.py
│   │   ├── product_service.py
│   │   ├── review_service.py
│   │   ├── trend_service.py
│   │   ├── llm_service.py
│   │   └── affiliate_service.py
│   ├── schemas/                # Pydantic schemas (to be implemented)
│   ├── db/                     # Database utilities (to be implemented)
│   ├── utils/                  # Utility functions
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example
│   └── README.md
│
├── shared/                     # Shared types and utilities
│   └── README.md
│
├── docs/                       # Documentation
│   ├── API.md
│   ├── SETUP.md
│   └── ARCHITECTURE.md
│
└── README.md                   # Project overview
```

## Key Features

### Frontend (Next.js)
- Modern React 18 with TypeScript
- Server-side rendering with App Router
- Responsive Tailwind CSS design
- Real-time API integration
- Image upload and processing
- Dashboard with analytics

### Backend (FastAPI)
- RESTful API architecture
- Image processing and analysis
- LLM integration (OpenAI/Claude)
- Product data aggregation
- Trend analysis engine
- Affiliate platform integrations
- Review sentiment analysis

### Database
- SQLAlchemy ORM
- PostgreSQL/SQLite support
- User management
- Product data persistence
- Analytics data storage

### LLM Features
- Review sentiment analysis
- Product description generation
- Trend prediction
- Design recommendations
- Marketing insights

## API Endpoints Structure

```
/api/v1/
├── /products
│   ├── POST /search-image
│   ├── GET /
│   ├── GET /{product_id}
│   └── POST /compare
├── /reviews
│   ├── GET /{product_id}
│   ├── POST /{product_id}/analyze
│   └── GET /{product_id}/summary
├── /trends
│   ├── GET /daily
│   ├── GET /weekly
│   └── POST /analyze
└── /affiliate
    ├── POST /sync
    ├── GET /products/{platform}
    ├── POST /generate-link
    └── POST /track-discount
```

## Environment Variables

### Frontend
- `NEXT_PUBLIC_API_URL`: Backend API URL

### Backend
- `DATABASE_URL`: Database connection string
- `OPENAI_API_KEY`: OpenAI API key
- `SHOPEE_API_KEY`: Shopee affiliate API key
- `LAZADA_API_KEY`: Lazada affiliate API key
- `TIKTOK_SHOP_API_KEY`: TikTok Shop API key
- `ALLOWED_ORIGINS`: CORS allowed origins

## Next Steps

1. ✅ Project structure initialized
2. ⏳ Setup database models and migrations
3. ⏳ Implement image processing service
4. ⏳ Integrate LLM services
5. ⏳ Setup affiliate platform APIs
6. ⏳ Implement review scraping
7. ⏳ Build trend analysis engine
8. ⏳ User authentication system
9. ⏳ Data dashboard and visualizations
10. ⏳ Deployment setup (Docker, CI/CD)
