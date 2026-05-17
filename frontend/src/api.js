// src/api.js — SkyBuffer API client (uses Vite proxy: /api → localhost:8000)
const BASE = import.meta.env.VITE_API_URL || ''

async function req(path, options = {}) {
  try {
    const res = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data) || 'Request failed')
    return data
  } catch (err) {
    if (err instanceof SyntaxError) throw new Error('Server returned invalid response')
    throw err
  }
}

export const api = {
  login:              (email, password) => req('/api/auth/login',           { method: 'POST', body: JSON.stringify({ email, password }) }),
  signup:             (email, password) => req('/api/auth/signup',          { method: 'POST', body: JSON.stringify({ email, password }) }),
  forgotPassword:     (email)           => req('/api/auth/forgot-password', { method: 'POST', body: JSON.stringify({ email }) }),
  getAirports:        ()                => req('/api/airports'),
  searchAirports:     (q)              => req(`/api/airports/search?q=${encodeURIComponent(q)}`),
  searchFlights:      (origin, dest, date) => req(`/api/flights/search?origin=${origin}&dest=${dest}&date_str=${date}`),
  getModelInfo:       ()                => req('/api/model-info'),
  getModelComparison: ()                => req('/api/model-comparison'),
  predict:            (payload)         => req('/api/predict', { method: 'POST', body: JSON.stringify(payload) }),
  getHistory:         (search = '', limit = 30) => req(`/api/history?search=${encodeURIComponent(search)}&limit=${limit}`),
}

