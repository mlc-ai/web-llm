// API configuration
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export const API_ENDPOINTS = {
  // Products
  PRODUCTS: `${API_URL}/api/v1/products`,
  PRODUCT_SEARCH_IMAGE: `${API_URL}/api/v1/products/search-image`,
  PRODUCT_COMPARE: `${API_URL}/api/v1/products/compare`,
  
  // Reviews
  REVIEWS: `${API_URL}/api/v1/reviews`,
  REVIEW_ANALYZE: `${API_URL}/api/v1/reviews/{product_id}/analyze`,
  REVIEW_SUMMARY: `${API_URL}/api/v1/reviews/{product_id}/summary`,
  
  // Trends
  TRENDS: `${API_URL}/api/v1/trends`,
  TRENDS_DAILY: `${API_URL}/api/v1/trends/daily`,
  TRENDS_WEEKLY: `${API_URL}/api/v1/trends/weekly`,
  TRENDS_ANALYZE: `${API_URL}/api/v1/trends/analyze`,
  
  // Affiliate
  AFFILIATE_SYNC: `${API_URL}/api/v1/affiliate/sync`,
  AFFILIATE_PRODUCTS: `${API_URL}/api/v1/affiliate/products`,
  AFFILIATE_GENERATE_LINK: `${API_URL}/api/v1/affiliate/generate-link`,
  
  // Health
  HEALTH: `${API_URL}/health`,
}

export default API_ENDPOINTS
