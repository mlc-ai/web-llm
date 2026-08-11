import axios, { AxiosInstance } from 'axios'
import { API_ENDPOINTS } from './api.config'

class APIClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
      headers: {
        'Content-Type': 'application/json',
      },
    })
  }

  // Products
  async getProducts(skip = 0, limit = 20) {
    return this.client.get(`/api/v1/products`, { params: { skip, limit } })
  }

  async searchByImage(file: File) {
    const formData = new FormData()
    formData.append('file', file)
    return this.client.post('/api/v1/products/search-image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  }

  async compareProducts(productIds: string[]) {
    return this.client.post('/api/v1/products/compare', { product_ids: productIds })
  }

  // Reviews
  async getReviews(productId: string, limit = 50) {
    return this.client.get(`/api/v1/reviews/${productId}`, { params: { limit } })
  }

  async analyzeReviews(productId: string) {
    return this.client.post(`/api/v1/reviews/${productId}/analyze`)
  }

  // Trends
  async getDailyTrends(limit = 10) {
    return this.client.get('/api/v1/trends/daily', { params: { limit } })
  }

  async getWeeklyTrends(limit = 20) {
    return this.client.get('/api/v1/trends/weekly', { params: { limit } })
  }

  async analyzeTrends(category?: string) {
    return this.client.post('/api/v1/trends/analyze', { category })
  }

  // Affiliate
  async syncAffiliateData() {
    return this.client.post('/api/v1/affiliate/sync')
  }

  async getAffiliateProducts(platform: string, limit = 50) {
    return this.client.get(`/api/v1/affiliate/products/${platform}`, { params: { limit } })
  }

  async generateAffiliateLink(productId: string, platform: string) {
    return this.client.post('/api/v1/affiliate/generate-link', { product_id: productId, platform })
  }

  // Health
  async checkHealth() {
    return this.client.get('/health')
  }
}

export const apiClient = new APIClient()
