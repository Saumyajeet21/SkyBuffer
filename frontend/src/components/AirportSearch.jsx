import { useState, useEffect, useRef } from 'react'
import { api } from '../api.js'
import './AirportSearch.css'

const FLAG = { India:'🇮🇳', UAE:'🇦🇪', UK:'🇬🇧', Singapore:'🇸🇬', Japan:'🇯🇵',
               Qatar:'🇶🇦', France:'🇫🇷', Germany:'🇩🇪', USA:'🇺🇸',
               Australia:'🇦🇺', Malaysia:'🇲🇾', Thailand:'🇹🇭', 'Hong Kong':'🇭🇰' }

export default function AirportSearch({ label, value, onChange, placeholder = 'Type city or airport…' }) {
  const [query,   setQuery]   = useState('')
  const [results, setResults] = useState([])
  const [open,    setOpen]    = useState(false)
  const [loading, setLoading] = useState(false)
  const ref = useRef(null)

  // Show value label when selected
  const display = value
    ? `${value.iata} — ${value.city}, ${value.country}`
    : query

  useEffect(() => {
    if (!open) return
    const t = setTimeout(async () => {
      setLoading(true)
      try {
        const d = await api.searchAirports(query)
        setResults(d.airports || [])
      } catch { setResults([]) }
      finally { setLoading(false) }
    }, 200)
    return () => clearTimeout(t)
  }, [query, open])

  // Seed results on open
  useEffect(() => {
    if (open && results.length === 0 && !query) {
      api.searchAirports('').then(d => setResults(d.airports || [])).catch(() => {})
    }
  }, [open])

  // Close on outside click
  useEffect(() => {
    const handler = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const select = (ap) => {
    onChange(ap)
    setQuery('')
    setOpen(false)
  }

  const clear = () => { onChange(null); setQuery(''); setOpen(true) }

  return (
    <div className="ap-wrap" ref={ref}>
      <label className="ap-label">{label}</label>
      <div className={`ap-input-box ${open ? 'focused' : ''} ${value ? 'has-value' : ''}`}
           onClick={() => { setOpen(true) }}>
        {value ? (
          <>
            <span className="ap-flag">{FLAG[value.country] || '🌐'}</span>
            <span className="ap-selected">{value.iata} — {value.city}</span>
            <button className="ap-clear" onClick={e => { e.stopPropagation(); clear() }}>✕</button>
          </>
        ) : (
          <input
            className="ap-input"
            placeholder={placeholder}
            value={query}
            onChange={e => { setQuery(e.target.value); setOpen(true) }}
            onFocus={() => setOpen(true)}
            autoComplete="off"
          />
        )}
      </div>

      {open && (
        <div className="ap-dropdown">
          {loading && <div className="ap-hint">Searching…</div>}
          {!loading && results.length === 0 && query.length > 1 && (
            <div className="ap-hint">No airports found</div>
          )}
          {results.map(ap => (
            <div key={ap.iata} className="ap-option" onClick={() => select(ap)}>
              <span className="ap-opt-flag">{FLAG[ap.country] || '🌐'}</span>
              <div>
                <span className="ap-opt-code">{ap.iata}</span>
                <span className="ap-opt-name">{ap.city}, {ap.country}</span>
                <span className="ap-opt-full">{ap.name}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
