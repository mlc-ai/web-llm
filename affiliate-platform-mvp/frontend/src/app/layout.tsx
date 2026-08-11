import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Affiliate Product Analysis Platform',
  description: 'Analyze products, trends, and optimize your affiliate marketing strategy',
  keywords: 'affiliate, product analysis, trend analysis, marketing',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
