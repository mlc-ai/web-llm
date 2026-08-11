'use client'

import { Upload, BarChart3, FileText } from 'lucide-react'
import { useState } from 'react'
import { apiClient } from '@/utils/http'

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('search')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any>(null)

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    try {
      setLoading(true)
      const response = await apiClient.searchByImage(file)
      setResults(response.data)
    } catch (error) {
      console.error('Upload error:', error)
      alert('Failed to search by image')
    } finally {
      setLoading(false)
    }
  }

  const getTrends = async () => {
    try {
      setLoading(true)
      const response = await apiClient.getDailyTrends()
      setResults(response.data)
    } catch (error) {
      console.error('Trends error:', error)
      alert('Failed to fetch trends')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <h2 className="text-3xl font-bold text-gray-900 mb-8">Dashboard</h2>

      <div className="flex gap-4 mb-8">
        <button 
          onClick={() => setActiveTab('search')}
          className={`px-6 py-3 rounded-lg font-semibold transition flex items-center gap-2 ${
            activeTab === 'search' ? 'bg-primary text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          <Upload size={20} /> Image Search
        </button>
        <button 
          onClick={() => { setActiveTab('trends'); getTrends(); }}
          className={`px-6 py-3 rounded-lg font-semibold transition flex items-center gap-2 ${
            activeTab === 'trends' ? 'bg-primary text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          <BarChart3 size={20} /> Trends
        </button>
      </div>

      <div className="bg-white rounded-xl shadow-md p-8">
        {activeTab === 'search' && (
          <div>
            <h3 className="text-2xl font-semibold mb-6">Search Products by Image</h3>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-primary transition cursor-pointer">
              <input 
                type="file" 
                accept="image/*" 
                onChange={handleImageUpload}
                className="hidden" 
                id="image-upload"
              />
              <label htmlFor="image-upload" className="cursor-pointer">
                <Upload size={48} className="mx-auto text-gray-400 mb-4" />
                <p className="text-xl font-semibold text-gray-700 mb-2">Drop your image here</p>
                <p className="text-gray-500">or click to select from computer</p>
              </label>
            </div>
          </div>
        )}

        {activeTab === 'trends' && (
          <div>
            <h3 className="text-2xl font-semibold mb-6">Daily Trends</h3>
            {loading ? (
              <p className="text-gray-500">Loading trends...</p>
            ) : results ? (
              <div>
                <p className="text-gray-600">Trends will be displayed here</p>
              </div>
            ) : (
              <p className="text-gray-500">Click on Trends tab to load latest trends</p>
            )}
          </div>
        )}

        {results && (
          <div className="mt-8 p-6 bg-gray-50 rounded-lg">
            <h4 className="font-semibold text-gray-900 mb-4">Results</h4>
            <pre className="text-sm text-gray-700 overflow-auto max-h-96">
              {JSON.stringify(results, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}
