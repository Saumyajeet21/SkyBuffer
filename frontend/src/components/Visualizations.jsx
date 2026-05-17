import { useState, useEffect, useCallback } from 'react'
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line,
} from 'recharts'
import './Visualizations.css'

const BACKEND = 'http://localhost:8000'
const COLORS  = { 'On Time': '#00b894', 'Delayed': '#e17055' }
const MODEL_COLORS = ['#4F8EF7', '#a29bfe', '#00cec9', '#00b894']

/* ── helpers ── */
const pct = v => `${(v * 100).toFixed(1)}%`

function StatCard({ label, value, sub, color }) {
  return (
    <div className="stat-card card">
      <p className="stat-label">{label}</p>
      <p className="stat-value" style={{ color }}>{value}</p>
      {sub && <p className="stat-sub">{sub}</p>}
    </div>
  )
}

/* ── LIVE SECTION ── */
function LiveCharts() {
  const [history,  setHistory]  = useState([])
  const [metrics,  setMetrics]  = useState(null)
  const [loading,  setLoading]  = useState(true)

  const refresh = useCallback(() => {
    Promise.all([
      fetch(`${BACKEND}/api/history?limit=200`).then(r => r.json()),
      fetch(`${BACKEND}/api/model-comparison`).then(r => r.json()),
    ]).then(([hist, info]) => {
      setHistory(hist.predictions || [])
      setMetrics(info)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])


  useEffect(() => { refresh(); const t = setInterval(refresh, 5000); return () => clearInterval(t) }, [refresh])

  if (loading) return <div className="viz-loading"><div className="spinner"/><p>Loading live data…</p></div>

  /* ── derive chart data ── */
  const delayed   = history.filter(h => h.status === 'Delayed').length
  const ontime    = history.filter(h => h.status === 'On Time').length
  const total     = history.length

  const pieData   = [
    { name: 'On Time', value: ontime },
    { name: 'Delayed', value: delayed },
  ].filter(d => d.value > 0)

  // Predictions by airline
  const byAirline = {}
  history.forEach(h => {
    const al = h.airline || 'Unknown'
    byAirline[al] = byAirline[al] || { On_Time: 0, Delayed: 0 }
    if (h.status === 'Delayed') byAirline[al].Delayed++
    else byAirline[al].On_Time++
  })
  const airlineData = Object.entries(byAirline)
    .map(([airline, v]) => ({ airline, ...v, total: v.On_Time + v.Delayed }))
    .sort((a, b) => b.total - a.total).slice(0, 10)

  // Delay minutes trend (last 20 predictions)
  const trendData = [...history].reverse().slice(-20).map((h, i) => ({
    i: i + 1,
    delay: parseFloat(h.predicted_delay_min || 0).toFixed(1),
    status: h.status,
  }))

  // Model radar data from model-comparison
  const radarData = metrics?.classification
    ? Object.entries(metrics.classification).map(([name, v]) => ({
        model: name,
        Accuracy:  parseFloat((v.Accuracy  * 100).toFixed(1)),
        Precision: parseFloat((v.Precision * 100).toFixed(1)),
        Recall:    parseFloat((v.Recall    * 100).toFixed(1)),
        F1:        parseFloat((v.F1        * 100).toFixed(1)),
      }))
    : []


  return (
    <div className="live-section">
      {/* refresh badge */}
      <div className="live-header">
        <h3 className="section-title">📡 Live Prediction Statistics</h3>
        <span className="live-badge">Auto-refreshes every 5s</span>
      </div>

      {total === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '2.5rem', color: 'var(--sub)' }}>
          No predictions yet. Make a prediction from the <strong>Predict</strong> tab to see live charts.
        </div>
      ) : (
        <>
          {/* Summary cards */}
          <div className="stat-grid">
            <StatCard label="Total Predictions" value={total}               color="#4F8EF7" />
            <StatCard label="On Time"            value={ontime}             color="#00b894"
                      sub={total ? `${((ontime/total)*100).toFixed(0)}%` : ''} />
            <StatCard label="Delayed"            value={delayed}            color="#e17055"
                      sub={total ? `${((delayed/total)*100).toFixed(0)}%` : ''} />
            <StatCard label="Avg Delay"
                      value={`${(history.reduce((s,h) => s + parseFloat(h.predicted_delay_min||0), 0) / total).toFixed(1)} min`}
                      color="#fdcb6e" />
          </div>

          {/* Row 1: Pie + Airline Bar */}
          <div className="charts-row">
            <div className="chart-box card">
              <h4 className="chart-title">Prediction Distribution</h4>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" outerRadius={90}
                       dataKey="value" label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`}
                       labelLine={false}>
                    {pieData.map(d => <Cell key={d.name} fill={COLORS[d.name]} />)}
                  </Pie>
                  <Tooltip formatter={(v) => [v, 'Flights']} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {airlineData.length > 0 && (
              <div className="chart-box card">
                <h4 className="chart-title">Predictions by Airline</h4>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={airlineData} margin={{ left: -10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.06)" />
                    <XAxis dataKey="airline" tick={{ fontSize: 11, fill: '#aaa' }} angle={-30} textAnchor="end" />
                    <YAxis tick={{ fontSize: 11, fill: '#aaa' }} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="On_Time" fill="#00b894" name="On Time" radius={[3,3,0,0]} />
                    <Bar dataKey="Delayed"  fill="#e17055" name="Delayed"  radius={[3,3,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Row 2: Delay trend */}
          {trendData.length > 1 && (
            <div className="chart-box card">
              <h4 className="chart-title">Predicted Delay Trend (last {trendData.length} predictions)</h4>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={trendData} margin={{ left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.06)" />
                  <XAxis dataKey="i" tick={{ fontSize: 11, fill: '#aaa' }} label={{ value: 'Prediction #', position: 'insideBottom', offset: -5, fill: '#aaa', fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11, fill: '#aaa' }} unit=" min" />
                  <Tooltip formatter={(v) => [`${v} min`, 'Est. Delay']} />
                  <Line type="monotone" dataKey="delay" stroke="#4F8EF7" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {/* Model metrics radar — always shown from model-info */}
      {radarData.length > 0 && (
        <div className="charts-row">
          {/* Radar */}
          <div className="chart-box card">
            <h4 className="chart-title">Model Metrics Radar (F1 · Acc · Prec · Recall)</h4>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={[
                { metric: 'Accuracy',  ...Object.fromEntries(radarData.map(m => [m.model.replace('\n',' '), m.Accuracy])) },
                { metric: 'Precision', ...Object.fromEntries(radarData.map(m => [m.model.replace('\n',' '), m.Precision])) },
                { metric: 'Recall',    ...Object.fromEntries(radarData.map(m => [m.model.replace('\n',' '), m.Recall])) },
                { metric: 'F1-Score',  ...Object.fromEntries(radarData.map(m => [m.model.replace('\n',' '), m.F1])) },
              ]}>
                <PolarGrid stroke="rgba(255,255,255,.1)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#ccc', fontSize: 12 }} />
                <PolarRadiusAxis domain={[0, 100]} tick={{ fill: '#888', fontSize: 10 }} />
                {radarData.map((m, i) => (
                  <Radar key={m.model} name={m.model.replace('\n', ' ')}
                    dataKey={m.model.replace('\n', ' ')}
                    stroke={MODEL_COLORS[i]} fill={MODEL_COLORS[i]} fillOpacity={0.15} strokeWidth={2} />
                ))}
                <Legend />
                <Tooltip formatter={(v) => [`${v}%`]} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* F1 + Precision bar comparison */}
          <div className="chart-box card">
            <h4 className="chart-title">F1 Score &amp; Precision Comparison</h4>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={radarData.map(m => ({ name: m.model.replace('\n',' '), F1: m.F1, Precision: m.Precision, Recall: m.Recall }))}
                margin={{ left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.06)" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#aaa' }} />
                <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11, fill: '#aaa' }} />
                <Tooltip formatter={(v) => [`${v}%`]} />
                <Legend />
                <Bar dataKey="F1"        fill="#00b894" name="F1-Score"  radius={[3,3,0,0]} />
                <Bar dataKey="Precision" fill="#4F8EF7" name="Precision" radius={[3,3,0,0]} />
                <Bar dataKey="Recall"    fill="#a29bfe" name="Recall"    radius={[3,3,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}

/* ── STATIC TRAINING PLOTS ── */
function TrainingPlots() {
  const [plots,   setPlots]   = useState([])
  const [loading, setLoading] = useState(true)
  const [active,  setActive]  = useState(null)

  useEffect(() => {
    fetch(`${BACKEND}/api/plots`)
      .then(r => r.json())
      .then(d => { setPlots(d.plots || []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return <div className="viz-loading"><div className="spinner"/></div>

  return (
    <div className="training-section">
      <div className="live-header">
        <h3 className="section-title">🎓 Training Analysis (Seaborn)</h3>
        <span className="static-badge">Generated after each training run</span>
      </div>

      <div className="viz-grid">
        {plots.map(plot => (
          <div key={plot.id} className="viz-card card" onClick={() => setActive(plot)}>
            <h3 className="viz-card-title">{plot.title}</h3>
            <p className="viz-card-desc">{plot.desc}</p>
            <div className="viz-img-wrap">
              <img src={`${BACKEND}${plot.url}`} alt={plot.title} className="viz-img" loading="lazy" />
              <div className="viz-expand-overlay"><span>Click to expand</span></div>
            </div>
          </div>
        ))}
      </div>

      {active && (
        <div className="viz-lightbox" onClick={() => setActive(null)}>
          <div className="viz-lightbox-inner" onClick={e => e.stopPropagation()}>
            <button className="viz-close" onClick={() => setActive(null)}>✕</button>
            <h3 className="viz-lb-title">{active.title}</h3>
            <p className="viz-lb-desc">{active.desc}</p>
            <img src={`${BACKEND}${active.url}`} alt={active.title} className="viz-lb-img" />
          </div>
        </div>
      )}
    </div>
  )
}

/* ── ROOT ── */
export default function Visualizations() {
  const [tab, setTab] = useState('live')
  return (
    <div className="viz-wrap fade-in">
      <div className="card viz-header">
        <h2 className="viz-title">Model Visualizations</h2>
        <p className="viz-sub">
          Live charts update automatically after every prediction · Training charts regenerate after retraining
        </p>
        <div className="viz-tabs">
          <button className={`viz-tab ${tab==='live'    ?'active':''}`} onClick={() => setTab('live')}>
            📡 Live Statistics
          </button>
          <button className={`viz-tab ${tab==='training'?'active':''}`} onClick={() => setTab('training')}>
            🎓 Training Analysis
          </button>
        </div>
      </div>

      {tab === 'live'     && <LiveCharts />}
      {tab === 'training' && <TrainingPlots />}
    </div>
  )
}
