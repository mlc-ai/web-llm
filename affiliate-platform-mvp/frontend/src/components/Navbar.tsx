'use client'

import { Menu, X } from 'lucide-react'
import { useState } from 'react'

interface NavbarProps {
  isLoggedIn: boolean
  setIsLoggedIn: (value: boolean) => void
}

export default function Navbar({ isLoggedIn, setIsLoggedIn }: NavbarProps) {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <nav className="bg-white shadow-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <h1 className="text-2xl font-bold text-primary">📊 AffiliateAI</h1>
          </div>

          <div className="hidden md:flex items-center space-x-8">
            <a href="#features" className="text-gray-600 hover:text-primary transition">Features</a>
            <a href="#how-it-works" className="text-gray-600 hover:text-primary transition">How it works</a>
            {isLoggedIn && <a href="#dashboard" className="text-gray-600 hover:text-primary transition">Dashboard</a>}
          </div>

          <div className="hidden md:flex items-center space-x-4">
            {!isLoggedIn ? (
              <>
                <button onClick={() => setIsLoggedIn(true)} className="px-4 py-2 text-primary border border-primary rounded-lg hover:bg-primary hover:text-white transition">
                  Login
                </button>
                <button onClick={() => setIsLoggedIn(true)} className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-blue-700 transition">
                  Sign Up
                </button>
              </>
            ) : (
              <button onClick={() => setIsLoggedIn(false)} className="px-4 py-2 text-gray-600 hover:text-primary transition">
                Logout
              </button>
            )}
          </div>

          <button className="md:hidden" onClick={() => setMenuOpen(!menuOpen)}>
            {menuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {menuOpen && (
          <div className="md:hidden pb-4 space-y-2">
            <a href="#features" className="block text-gray-600 hover:text-primary py-2">Features</a>
            <a href="#how-it-works" className="block text-gray-600 hover:text-primary py-2">How it works</a>
          </div>
        )}
      </div>
    </nav>
  )
}
