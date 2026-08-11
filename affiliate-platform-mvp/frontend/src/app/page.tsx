'use client'

import { useState } from 'react'
import Navbar from '@/components/Navbar'
import Hero from '@/components/Hero'
import Features from '@/components/Features'
import Dashboard from '@/components/Dashboard'
import Footer from '@/components/Footer'

export default function Home() {
  const [isLoggedIn, setIsLoggedIn] = useState(false)

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      <Navbar isLoggedIn={isLoggedIn} setIsLoggedIn={setIsLoggedIn} />
      
      {!isLoggedIn ? (
        <>
          <Hero setIsLoggedIn={setIsLoggedIn} />
          <Features />
        </>
      ) : (
        <Dashboard />
      )}
      
      <Footer />
    </main>
  )
}
