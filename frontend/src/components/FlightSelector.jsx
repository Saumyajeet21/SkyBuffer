import './FlightSelector.css'

const AIRLINE_COLORS = {
  AI:'#e8174b', '6E':'#0c4e9e', SG:'#e8341c',
  UK:'#6b0080', IX:'#e8174b', QP:'#f57c00', G8:'#0057a8',
  EK:'#c8102e', QR:'#5c0632', SQ:'#0072ce', BA:'#003087',
  LH:'#05164d', AF:'#002395',
}

const AIRCRAFT_TYPE = {
  'B787':'Boeing 787 Dreamliner', 'B737':'Boeing 737',
  'A320':'Airbus A320', 'A321':'Airbus A321',
  'A350':'Airbus A350', 'A380':'Airbus A380', 'B777':'Boeing 777',
}

export default function FlightSelector({ flights, onSelect, origin, dest, loading }) {
  if (loading) return (
    <div className="fs-loading">
      <div className="spinner" />
      <p>Searching flights…</p>
    </div>
  )

  if (!flights) return null

  if (flights.length === 0) return (
    <div className="fs-empty">
      <span style={{fontSize:'2.5rem'}}>😕</span>
      <p>No scheduled flights found for <strong>{origin} → {dest}</strong></p>
      <p style={{fontSize:'.85rem',color:'var(--sub)'}}>Try a different route or check the airport codes.</p>
    </div>
  )

  return (
    <div className="fs-wrap fade-in">
      <div className="fs-header">
        <h3>✈️ Available Flights — <span className="gradient-text">{origin} → {dest}</span></h3>
        <span className="fs-count">{flights.length} flights found</span>
      </div>

      <div className="fs-grid">
        {flights.map((f, i) => (
          <div key={i} className="fs-card" onClick={() => onSelect(f)}>
            {/* Airline strip */}
            <div className="fs-airline-strip" style={{background: AIRLINE_COLORS[f.airline] || '#4F8EF7'}}>
              <span className="fs-fn">{f.flight_no}</span>
              <span className="fs-an">{f.airline_name}</span>
            </div>

            {/* Times */}
            <div className="fs-times">
              <div className="fs-time-block">
                <span className="fs-time">{f.departure}</span>
                <span className="fs-city">{origin}</span>
              </div>
              <div className="fs-arrow">
                <span className="fs-duration">✈</span>
              </div>
              <div className="fs-time-block">
                <span className="fs-time">{f.arrival}</span>
                <span className="fs-city">{dest}</span>
              </div>
            </div>

            {/* Aircraft */}
            <div className="fs-footer">
              <span className="fs-aircraft">🛩 {AIRCRAFT_TYPE[f.aircraft] || f.aircraft}</span>
              <button className="fs-select-btn">Predict Delay →</button>
            </div>
          </div>
        ))}
      </div>

      <p className="fs-note">📋 Schedule based on DGCA timetable. Click any flight to predict delay.</p>
    </div>
  )
}
