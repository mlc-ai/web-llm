'use client'

import { Image, BarChart3, Target, Zap, Code, TrendingUp } from 'lucide-react'

const features = [
  {
    icon: Image,
    title: 'Smart Image Search',
    description: 'Upload any product image to find similar items from your affiliate platforms'
  },
  {
    icon: BarChart3,
    title: 'Review Analysis',
    description: 'AI-powered sentiment analysis of customer reviews to understand product quality'
  },
  {
    icon: TrendingUp,
    title: 'Trend Analytics',
    description: 'Track daily, weekly, and monthly trends to identify hot products early'
  },
  {
    icon: Target,
    title: 'Product Comparison',
    description: 'Deep compare features, prices, and reviews across multiple products'
  },
  {
    icon: Code,
    title: 'Design Insights',
    description: 'Learn from successful product designs and marketing strategies'
  },
  {
    icon: Zap,
    title: 'Affiliate Integration',
    description: 'Easy sync and link generation for Shopee, Lazada, TikTok Shop and more'
  }
]

export default function Features() {
  return (
    <section id="features" className="py-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">Powerful Features</h2>
          <p className="text-xl text-gray-600">Everything you need to succeed in affiliate marketing</p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, idx) => {
            const Icon = feature.icon
            return (
              <div key={idx} className="bg-white rounded-xl p-8 shadow-md hover:shadow-lg transition transform hover:scale-105">
                <div className="bg-primary/10 w-16 h-16 rounded-lg flex items-center justify-center mb-4">
                  <Icon size={32} className="text-primary" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
