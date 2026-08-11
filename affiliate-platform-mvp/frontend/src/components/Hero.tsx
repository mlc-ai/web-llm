'use client'

import { ArrowRight, Image, TrendingUp, Zap } from 'lucide-react'

interface HeroProps {
  setIsLoggedIn: (value: boolean) => void
}

export default function Hero({ setIsLoggedIn }: HeroProps) {
  return (
    <div className="relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-28">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6 leading-tight">
              Empower Your Affiliate Business with <span className="text-primary">AI Analytics</span>
            </h2>
            <p className="text-xl text-gray-600 mb-8">
              Upload product images, analyze reviews, track trends, and optimize your affiliate strategy with powerful AI insights.
            </p>
            <div className="flex gap-4">
              <button 
                onClick={() => setIsLoggedIn(true)}
                className="px-8 py-3 bg-primary text-white rounded-lg hover:bg-blue-700 transition flex items-center gap-2"
              >
                Get Started <ArrowRight size={20} />
              </button>
              <button className="px-8 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition">
                Watch Demo
              </button>
            </div>
          </div>

          <div className="relative">
            <div className="bg-gradient-to-br from-primary to-blue-700 rounded-2xl p-8 text-white shadow-2xl">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white/10 rounded-lg p-4">
                  <TrendingUp size={32} className="mb-2" />
                  <p className="font-semibold">Trend Analysis</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4">
                  <Image size={32} className="mb-2" />
                  <p className="font-semibold">Image Search</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4">
                  <Zap size={32} className="mb-2" />
                  <p className="font-semibold">AI Analysis</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4">
                  <TrendingUp size={32} className="mb-2" />
                  <p className="font-semibold">Performance</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
