import { useState, useEffect } from 'react'
import { api } from '../api.js'
import ResultCard from '../components/ResultCard.jsx'
import History from '../components/History.jsx'
import AirportSearch from '../components/AirportSearch.jsx'
import FlightSelector from '../components/FlightSelector.jsx'
import './Dashboard.css'

export default function Dashboard({ user, onLogout }) {
  const [activeTab, setActiveTab] = useState('predict')
  // Step state: 'search' → 'flights' → 'result'
  const [step, setStep] = useState('search')

  // Step 1 — route + date
  const [origin, setOrigin] = useState(null)
  const [dest,   setDest]   = useState(null)
  const [depDate, setDepDate] = useState(new Date().toISOString().split('T')[0])

  // Step 2 — flights list
  const [flights,  setFlights]  = useState(null)
  const [fsLoading, setFsLoading] = useState(false)
  const [fsError,   setFsError]   = useState(null)

  // Step 3 — selected flight + prediction
  const [selFlight, setSelFlight] = useState(null)
  const [result,    setResult]    = useState(null)
  const [predLoad,  setPredLoad]  = useState(false)
  const [predErr,   setPredErr]   = useState(null)

  const searchFlights = async () => {
    if (!origin || !dest) return
    setFsError(null); setFsLoading(true); setFlights(null); setStep('flights')
    try {
      const d = await api.searchFlights(origin.iata, dest.iata, depDate)
      setFlights(d.flights || [])
    } catch(e) { setFsError(e.message) }
    finally { setFsLoading(false) }
  }

  const selectFlight = async (flight) => {
    setSelFlight(flight); setPredErr(null); setResult(null); setPredLoad(true); setStep('result')
    try {
      const data = await api.predict({
        origin: origin.iata, dest: dest.iata,
        airline: flight.airline,
        departure_date: depDate,
        departure_hour: flight.departure_hour,
      })
      setResult(data)
    } catch(e) { setPredErr(e.message) }
    finally { setPredLoad(false) }
  }

  const reset = () => {
    setStep('search'); setFlights(null); setSelFlight(null); setResult(null)
  }

  return (
    <div className="dash-layout">
      {/* Sidebar */}
      <aside className="sidebar glass">
        <div className="sidebar-logo">
          <span style={{fontSize:'2rem'}}>✈️</span>
          <h2 className="gradient-text" style={{fontWeight:800,fontSize:'1.4rem'}}>SkyBuffer</h2>
        </div>

        <nav className="sidebar-nav">
          {[['predict','🔍','Predict'],['history','📋','History']].map(([k,ico,lbl]) => (
            <button key={k} className={`nav-item ${activeTab===k?'active':''}`}
              onClick={() => setActiveTab(k)}>
              <span>{ico}</span> {lbl}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="user-pill">
            <span className="user-avatar">👤</span>
            <span className="user-email">{user.email}</span>
          </div>
          <button className="btn btn-ghost btn-full" onClick={onLogout} style={{marginTop:'.75rem'}}>
            🚪 Logout
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="dash-main">
        <header className="dash-header">
          <div>
            <h1 style={{fontWeight:800,fontSize:'1.8rem'}}>
              <span className="gradient-text">SkyBuffer</span>
            </h1>
            <p style={{color:'var(--sub)',fontSize:'.9rem'}}>AI-Powered Flight Delay Prediction</p>
          </div>
          <span style={{color:'var(--sub)',fontSize:'.85rem'}}>✅ Models Loaded</span>
        </header>

        {/* Mobile tabs */}
        <div className="tabs mobile-tabs">
          <button className={`tab ${activeTab==='predict'?'active':''}`}  onClick={()=>setActiveTab('predict')}>🔍 Predict</button>
          <button className={`tab ${activeTab==='history'?'active':''}`}  onClick={()=>setActiveTab('history')}>📋 History</button>
        </div>

        {/* ── PREDICT TAB ── */}
        {activeTab === 'predict' && (
          <div className="fade-in">

            {/* Step indicator */}
            <div className="step-bar">
              {[['1','Route & Date', 'search'],['2','Choose Flight','flights'],['3','Prediction','result']].map(([n,lbl,s],i) => (
                <div key={n} className={`step-item ${step===s?'active':''} ${(['search','flights','result'].indexOf(step)) > i ? 'done':''}`}>
                  <div className="step-dot">{(['search','flights','result'].indexOf(step)) > i ? '✓' : n}</div>
                  <span>{lbl}</span>
                  {i < 2 && <div className="step-line"/>}
                </div>
              ))}
            </div>

            {/* STEP 1 — Route search */}
            {step === 'search' && (
              <div className="card fade-in">
                <h2 style={{marginBottom:'1.5rem',fontWeight:700}}>🔍 Search Flights</h2>
                <div className="grid-2" style={{marginBottom:'1rem'}}>
                  <AirportSearch label="🛫 Origin Airport" value={origin} onChange={setOrigin} placeholder="e.g. Delhi, DEL" />
                  <AirportSearch label="🛬 Destination Airport" value={dest}   onChange={setDest}   placeholder="e.g. Mumbai, BOM" />
                </div>
                <div className="input-group" style={{maxWidth:300,marginBottom:'1.5rem'}}>
                  <label>📅 Travel Date</label>
                  <input className="input" type="date" value={depDate}
                    min={new Date().toISOString().split('T')[0]}
                    onChange={e=>setDepDate(e.target.value)} />
                </div>
                {fsError && <div className="alert alert-error">{fsError}</div>}
                <button className="btn btn-primary" style={{padding:'.85rem 2.5rem',fontSize:'1rem'}}
                  onClick={searchFlights} disabled={!origin || !dest || fsLoading}>
                  ✈️ Search Available Flights
                </button>
                {(!origin || !dest) && (
                  <p style={{color:'var(--sub)',fontSize:'.82rem',marginTop:'.75rem'}}>Select both origin and destination airports to continue.</p>
                )}
              </div>
            )}

            {/* STEP 2 — Flight list */}
            {step === 'flights' && (
              <div className="card fade-in">
                <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'1rem'}}>
                  <div>
                    <p style={{color:'var(--sub)',fontSize:'.85rem'}}>📅 {depDate}</p>
                  </div>
                  <button className="btn btn-ghost" onClick={reset}>← Change Route</button>
                </div>
                <FlightSelector
                  flights={flights} loading={fsLoading}
                  origin={origin?.iata} dest={dest?.iata}
                  onSelect={selectFlight}
                />
              </div>
            )}

            {/* STEP 3 — Prediction result */}
            {step === 'result' && (
              <div className="fade-in">
                {/* Selected flight info */}
                {selFlight && (
                  <div className="card" style={{marginBottom:'1rem',borderColor:'var(--blue)'}}>
                    <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',flexWrap:'wrap',gap:'1rem'}}>
                      <div>
                        <h3 style={{fontWeight:800,fontSize:'1.1rem'}}>
                          ✈️ {selFlight.flight_no} — {selFlight.airline_name}
                        </h3>
                        <p style={{color:'var(--sub)',fontSize:'.88rem',marginTop:'.25rem'}}>
                          {origin?.city} ({origin?.iata}) → {dest?.city} ({dest?.iata})
                          &nbsp;·&nbsp; Dep: {selFlight.departure} &nbsp;·&nbsp; {depDate}
                        </p>
                      </div>
                      <div style={{display:'flex',gap:'.75rem'}}>
                        <button className="btn btn-ghost" onClick={() => setStep('flights')}>← Back to Flights</button>
                        <button className="btn btn-ghost" onClick={reset}>🔄 New Search</button>
                      </div>
                    </div>
                  </div>
                )}

                {predLoad && (
                  <div className="card" style={{textAlign:'center',padding:'3rem'}}>
                    <div className="spinner" style={{margin:'0 auto'}} />
                    <p style={{color:'var(--sub)',marginTop:'1rem'}}>Fetching live weather & predicting delay…</p>
                  </div>
                )}
                {predErr && <div className="alert alert-error">{predErr}</div>}
                {result && (
                  <ResultCard result={result} form={{
                    origin:origin?.iata, dest:dest?.iata,
                    airline:selFlight?.airline,
                    departure_date:depDate, departure_hour:selFlight?.departure_hour
                  }} airports={{[origin?.iata]:origin,[dest?.iata]:dest}} flags={{}} />
                )}
              </div>
            )}
          </div>
        )}

        {activeTab === 'history' && <History />}
      </main>
    </div>
  )
}
