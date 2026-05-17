import { useState, useEffect } from 'react'
import Auth from './pages/Auth.jsx'
import Dashboard from './pages/Dashboard.jsx'

export default function App() {
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('sb_user')) } catch { return null }
  })

  const handleLogin = (userData) => {
    localStorage.setItem('sb_user', JSON.stringify(userData))
    setUser(userData)
  }

  const handleLogout = () => {
    localStorage.removeItem('sb_user')
    setUser(null)
  }

  if (!user) return <Auth onLogin={handleLogin} />
  return <Dashboard user={user} onLogout={handleLogout} />
}
