import { useState, useEffect } from 'react'
import { api } from '../api.js'

const STATUS_COLORS = { 'On Time':'var(--green)', 'Delayed':'var(--red)', 'Alternate':'var(--blue)' }

export default function History() {
  const [rows,    setRows]    = useState([])
  const [search,  setSearch]  = useState('')
  const [loading, setLoading] = useState(false)

  const load = async (q='') => {
    setLoading(true)
    try { const d = await api.getHistory(q); setRows(d.predictions || []) }
    catch(e) { console.error(e) }
    finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  const handleSearch = (e) => {
    const v = e.target.value; setSearch(v)
    clearTimeout(window._ht)
    window._ht = setTimeout(() => load(v), 400)
  }

  return (
    <div className="fade-in">
      <div className="card" style={{marginBottom:'1rem'}}>
        <h2 style={{fontWeight:700,marginBottom:'1rem'}}>📋 Prediction History</h2>
        <input className="input" placeholder="🔍 Search by airport, airline, or status…"
          value={search} onChange={handleSearch} />
      </div>

      <div className="card" style={{padding:0}}>
        {loading ? (
          <div style={{padding:'3rem',textAlign:'center'}}><div className="spinner" /></div>
        ) : rows.length === 0 ? (
          <div style={{padding:'3rem',textAlign:'center',color:'var(--sub)'}}>
            No predictions yet. Make your first prediction!
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
                    <td style={{color:'var(--sub)',fontSize:'.82rem',whiteSpace:'nowrap'}}>
                      {new Date(r.created_at).toLocaleString('en',{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'})}
                    </td>
                    <td><strong>{r.origin}</strong> → <strong>{r.destination}</strong></td>
                    <td>{r.airline}</td>
                    <td style={{whiteSpace:'nowrap'}}>{r.departure_date}</td>
                    <td>{String(r.departure_hour).padStart(2,'0')}:00</td>
                    <td><strong style={{color:'var(--blue)'}}>{r.predicted_delay_min}</strong></td>
                    <td>
                      <span style={{
                        color: STATUS_COLORS[r.status]||'var(--sub)',
                        fontWeight:600, fontSize:'.82rem'
                      }}>{r.status}</span>
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
