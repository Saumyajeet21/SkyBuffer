import { useState, useEffect, useCallback } from 'react'
import { api } from '../api.js'

const STATUS_COLORS = { 'On Time':'var(--green)', 'Delayed':'var(--red)', 'Alternate':'var(--blue)' }

const MAX_RETRIES = 4          // retry up to 4 times on fresh start
const RETRY_DELAYS = [2000, 3000, 5000, 8000]  // ms between retries

export default function History() {
  const [rows,    setRows]    = useState([])
  const [search,  setSearch]  = useState('')
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [retrying, setRetrying] = useState(false)

  // Core fetch — returns true if data was found
  const fetchHistory = useCallback(async (q = '') => {
    try {
      const d = await api.getHistory(q)
      const preds = d.predictions || []
      setRows(preds)
      setError(null)
      return preds.length > 0
    } catch (e) {
      setError('Backend not reachable — retrying…')
      return false
    }
  }, [])

  // On mount: try immediately, then auto-retry with backoff if empty/failed
  useEffect(() => {
    let cancelled = false
    const run = async () => {
      setLoading(true)
      const found = await fetchHistory('')
      setLoading(false)

      if (!found && !cancelled) {
        // Backend might still be starting up — retry with backoff
        for (let i = 0; i < MAX_RETRIES; i++) {
          if (cancelled) break
          setRetrying(true)
          await new Promise(r => setTimeout(r, RETRY_DELAYS[i]))
          if (cancelled) break
          const ok = await fetchHistory('')
          if (ok) { setRetrying(false); break }
          if (i === MAX_RETRIES - 1) {
            setRetrying(false)
            setError(null) // give up silently — show "no predictions" instead
          }
        }
      }
    }
    run()
    return () => { cancelled = true }
  }, [fetchHistory])

  // Manual search (debounced) — no retry needed
  const handleSearch = (e) => {
    const v = e.target.value
    setSearch(v)
    clearTimeout(window._ht)
    window._ht = setTimeout(() => {
      setLoading(true)
      fetchHistory(v).finally(() => setLoading(false))
    }, 400)
  }

  const handleRefresh = () => {
    setLoading(true)
    fetchHistory(search).finally(() => setLoading(false))
  }

  return (
    <div className="fade-in">
      <div className="card" style={{marginBottom:'1rem'}}>
        <div style={{display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'1rem'}}>
          <h2 style={{fontWeight:700, margin:0}}>📋 Prediction History</h2>
          <button
            onClick={handleRefresh}
            style={{
              background:'var(--accent,#6c63ff)', color:'#fff', border:'none',
              borderRadius:'8px', padding:'6px 16px', cursor:'pointer',
              fontSize:'.85rem', fontWeight:600, opacity: loading ? 0.6 : 1
            }}
            disabled={loading}
          >
            {loading ? '⏳ Loading…' : '🔄 Refresh'}
          </button>
        </div>
        <input className="input" placeholder="🔍 Search by airport, airline, or status…"
          value={search} onChange={handleSearch} />
      </div>

      <div className="card" style={{padding:0}}>
        {loading ? (
          <div style={{padding:'3rem', textAlign:'center'}}><div className="spinner" /></div>
        ) : retrying ? (
          <div style={{padding:'3rem', textAlign:'center', color:'var(--sub)'}}>
            <div className="spinner" style={{marginBottom:'1rem'}} />
            <div>Backend starting up — retrying automatically…</div>
          </div>
        ) : rows.length === 0 ? (
          <div style={{padding:'3rem', textAlign:'center', color:'var(--sub)'}}>
            {error
              ? <span style={{color:'var(--red)'}}>⚠️ {error}</span>
              : 'No predictions yet. Make your first prediction!'}
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th><th>Route</th><th>Airline</th>
                  <th>Date</th><th>Hour</th>
                  <th>Delay (min)</th><th>Status</th>
                  <th>Visibility</th><th>Wind</th><th>Congestion</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>
                    <td style={{color:'var(--sub)', fontSize:'.82rem', whiteSpace:'nowrap'}}>
                      {new Date(r.created_at).toLocaleString('en', {day:'numeric', month:'short', hour:'2-digit', minute:'2-digit'})}
                    </td>
                    <td><strong>{r.origin}</strong> → <strong>{r.destination}</strong></td>
                    <td>{r.airline}</td>
                    <td style={{whiteSpace:'nowrap'}}>{r.departure_date}</td>
                    <td>{String(r.departure_hour).padStart(2,'0')}:00</td>
                    <td>
                      <strong style={{color:'var(--blue)'}}>
                        {r.predicted_delay_min ?? r.delay_minutes ?? '—'}
                      </strong>
                    </td>
                    <td>
                      <span style={{color: STATUS_COLORS[r.status]||'var(--sub)', fontWeight:600, fontSize:'.82rem'}}>
                        {r.status}
                      </span>
                    </td>
                    <td>{r.visibility} mi</td>
                    <td>{r.wind_speed} mph</td>
                    <td>{r.congestion_index}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
