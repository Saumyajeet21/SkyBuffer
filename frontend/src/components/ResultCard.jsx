import './ResultCard.css'


const CLF_COLORS = {
  'Logistic Regression': '#4F8EF7',
  'Decision Tree': '#a29bfe',
  'Random Forest': '#00cec9',
  'XGBoost': '#00b894',
}
const TYPE_BADGE = {
  'Base': { label: 'Base Learner', cls: 'badge-base' },
  'Ensemble': { label: 'Ensemble', cls: 'badge-ens' },
}

/* ── Inline progress bar ── */
function Bar({ val, color }) {
  const pct = Math.max(0, Math.min(100, Math.round((val || 0) * 100)))
  return (
    <div className="rc-bar-bg">
      <div className="rc-bar-fill" style={{ width: `${pct}%`, background: color }} />
      <span className="rc-bar-val">{pct}%</span>
    </div>
  )
}

/* ── Confusion Matrix 2×2 ── */
function ConfMatrix({ cm, color }) {
  if (!cm) return <span className="rc-na">Data not available</span>
  const { TN = 0, FP = 0, FN = 0, TP = 0 } = cm
  const total = TN + FP + FN + TP || 1
  const pct = v => `${((v / total) * 100).toFixed(1)}%`
  return (
    <div className="rc-cm">
      <div className="rc-cm-labels-top">
        <span />
        <span className="rc-cm-axis">Predicted On Time</span>
        <span className="rc-cm-axis">Predicted Delayed</span>
      </div>
      <div className="rc-cm-row">
        <span className="rc-cm-axis rc-cm-side">Actual On Time</span>
        <div className="rc-cm-cell rc-cm-tn" style={{ borderColor: color }}>
          <span className="rc-cm-lbl">TN</span>
          <span className="rc-cm-num">{TN.toLocaleString()}</span>
          <span className="rc-cm-pct">{pct(TN)}</span>
        </div>
        <div className="rc-cm-cell rc-cm-fp">
          <span className="rc-cm-lbl">FP</span>
          <span className="rc-cm-num">{FP.toLocaleString()}</span>
          <span className="rc-cm-pct">{pct(FP)}</span>
        </div>
      </div>
      <div className="rc-cm-row">
        <span className="rc-cm-axis rc-cm-side">Actual Delayed</span>
        <div className="rc-cm-cell rc-cm-fn">
          <span className="rc-cm-lbl">FN</span>
          <span className="rc-cm-num">{FN.toLocaleString()}</span>
          <span className="rc-cm-pct">{pct(FN)}</span>
        </div>
        <div className="rc-cm-cell rc-cm-tp" style={{ borderColor: color }}>
          <span className="rc-cm-lbl">TP</span>
          <span className="rc-cm-num">{TP.toLocaleString()}</span>
          <span className="rc-cm-pct">{pct(TP)}</span>
        </div>
      </div>
    </div>
  )
}

/* ── Feature importance bars ── */
function FeatImportance({ data, color }) {
  if (!data || !data.length) return null
  const top = data.slice(0, 8)
  const max = top[0]?.importance || 1
  return (
    <div className="rc-fi">
      {top.map((d, i) => (
        <div key={i} className="rc-fi-row">
          <span className="rc-fi-name">{d.feature}</span>
          <div className="rc-bar-bg" style={{ flex: 1 }}>
            <div className="rc-bar-fill"
              style={{ width: `${(d.importance / max) * 100}%`, background: color }} />
          </div>
          <span className="rc-fi-val">{(d.importance * 100).toFixed(2)}%</span>
        </div>
      ))}
    </div>
  )
}

export default function ResultCard({ result }) {
  const {
    prob, status, consensus, delay_minutes,
    weather, congestion, alternates,
    model_comparison, recommended_model, recommendation_reason,
    confusion_matrices, feature_importance, dataset_info
  } = result

  const clfs = model_comparison || {}
  const cms = confusion_matrices || {}
  const fi = feature_importance || {}
  const clfNames = Object.keys(clfs)
  const isDelayed = status === 'Delayed'
  const color = CLF_COLORS[recommended_model] || '#00b894'
  const bestModel = clfs[recommended_model] || {}
  const delayRate = ((dataset_info?.delayed_rate || 0) * 100).toFixed(1)

  return (
    <div className="rc-wrap fade-in">

      {/* ══ HERO ══ */}
      <div className={`rc-hero card ${isDelayed ? 'rc-delayed' : 'rc-ontime'}`}>
        <div className="rc-hero-icon">{isDelayed ? '⚠️' : '✅'}</div>
        <div className="rc-hero-content">
          <h2 className="rc-hero-status">{status}</h2>
          <div className="rc-hero-stats">
            <div className="rc-stat-pill" style={{ borderColor: isDelayed ? '#e17055' : '#00b894' }}>
              <span className="rc-stat-label">Est. Delay</span>
              <span className="rc-stat-value" style={{ color: isDelayed ? '#e17055' : '#00b894' }}>
                {delay_minutes > 0 ? `+${delay_minutes} min` : 'On Time'}
              </span>
            </div>
            <div className="rc-stat-pill" style={{ borderColor: isDelayed ? '#e17055' : '#00b894' }}>
              <span className="rc-stat-label">Delay Risk</span>
              <span className="rc-stat-value"
                style={{ color: isDelayed ? '#e17055' : '#00b894' }}>{prob}%</span>
            </div>
            <div className="rc-stat-pill" style={{ borderColor: color }}>
              <span className="rc-stat-label">Best Model</span>
              <span className="rc-stat-value" style={{ color }}>{recommended_model}</span>
            </div>
            <div className="rc-stat-pill">
              <span className="rc-stat-label">Accuracy</span>
              <span className="rc-stat-value">
                {bestModel.Accuracy ? `${(bestModel.Accuracy * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
            <div className="rc-stat-pill">
              <span className="rc-stat-label">F1-Score</span>
              <span className="rc-stat-value" style={{ color }}>
                {bestModel.F1 ? `${(bestModel.F1 * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
            <div className="rc-stat-pill">
              <span className="rc-stat-label">Precision</span>
              <span className="rc-stat-value">
                {bestModel.Precision ? `${(bestModel.Precision * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
            <div className="rc-stat-pill">
              <span className="rc-stat-label">Recall</span>
              <span className="rc-stat-value">
                {bestModel.Recall ? `${(bestModel.Recall * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
          </div>
          <p className="rc-hero-consensus">
            {clfNames.length}-model consensus: <strong>{consensus}</strong>
            &nbsp;({clfNames.filter(n => clfs[n]?.is_delayed).length}/{clfNames.length} models predict delayed)
          </p>
        </div>
      </div>

      {/* ══ SECTION 1: ALL MODEL PREDICTIONS ══ */}
      <div className="card rc-section">
        <h3 className="rc-section-title">Classification Models — Prediction for This Flight</h3>
        <p className="rc-section-sub">
          Task 3: 2 base learners + Task 5: 2 ensemble methods · Trained on{' '}
          {dataset_info?.n_samples?.toLocaleString()} samples · {delayRate}% delayed
        </p>
        <div className="rc-model-grid">
          {clfNames.map(name => {
            const m = clfs[name]
            const badge = TYPE_BADGE[m.Type] || TYPE_BADGE['Base']
            const mc = CLF_COLORS[name]
            return (
              <div key={name}
                className={`rc-model-card ${name === recommended_model ? 'rc-model-best' : ''}`}
                style={{ borderColor: mc }}>
                <div className="rc-model-header" style={{ background: mc + '22' }}>
                  <span className="rc-model-name">{name}</span>
                  <span className={`rc-badge ${badge.cls}`}>{badge.label}</span>
                  {name === recommended_model &&
                    <span className="rc-badge rc-badge-rec">Best</span>}
                </div>
                <div className="rc-model-pred">
                  <span className={m.is_delayed ? 'pred-delayed' : 'pred-ontime'}>
                    {m.is_delayed ? 'Delayed' : 'On Time'}
                  </span>
                  <span className="rc-pred-prob">{m.probability}% risk</span>
                </div>
                <div className="rc-model-metrics">
                  <div className="rc-m-row"><span>Accuracy</span>
                    <Bar val={m.Accuracy} color={mc} /></div>
                  <div className="rc-m-row"><span>Precision</span>
                    <Bar val={m.Precision} color={mc} /></div>
                  <div className="rc-m-row"><span>Recall</span>
                    <Bar val={m.Recall} color={mc} /></div>
                  <div className="rc-m-row"><span>F1-Score</span>
                    <Bar val={m.F1} color={mc} /></div>
                  <div className="rc-m-row rc-time-row">
                    <span>Train Time</span>
                    <span className="rc-time-val">{m.Training_Time}s</span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* ══ SECTION 2: CONSOLIDATED COMPARISON TABLE ══ */}
      <div className="card rc-section">
        <h3 className="rc-section-title">Consolidated Metrics Comparison — Base Learners vs Ensemble</h3>
        <p className="rc-section-sub">
          Task 6: Full comparison · Accuracy, Precision, Recall, F1-Score, Training Time
        </p>
        <div className="rc-table-wrap">
          <table className="rc-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Type</th>
                <th>Prediction</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Train Time</th>
              </tr>
            </thead>
            <tbody>
              {clfNames.map(name => {
                const m = clfs[name]
                const isRec = name === recommended_model
                return (
                  <tr key={name} className={isRec ? 'rc-best-row' : ''}>
                    <td>
                      <span className="rc-td-model" style={{ color: CLF_COLORS[name] }}>
                        {isRec ? '★ ' : ''}{name}
                      </span>
                    </td>
                    <td>
                      <span className={`rc-badge ${TYPE_BADGE[m.Type]?.cls}`}>{m.Type}</span>
                    </td>
                    <td>
                      <span className={m.is_delayed ? 'pred-delayed' : 'pred-ontime'}>
                        {m.prediction}
                      </span>
                    </td>
                    <td>{m.Accuracy !== 'N/A' ? `${(m.Accuracy * 100).toFixed(1)}%` : '—'}</td>
                    <td>{m.Precision !== 'N/A' ? `${(m.Precision * 100).toFixed(1)}%` : '—'}</td>
                    <td>{m.Recall !== 'N/A' ? `${(m.Recall * 100).toFixed(1)}%` : '—'}</td>
                    <td>
                      <strong style={{ color: CLF_COLORS[name] }}>
                        {m.F1 !== 'N/A' ? `${(m.F1 * 100).toFixed(1)}%` : '—'}
                      </strong>
                    </td>
                    <td style={{ color: 'var(--sub)' }}>{m.Training_Time}s</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ══ SECTION 3: CONFUSION MATRICES ══ */}
      {Object.keys(cms).length > 0 && (
        <div className="card rc-section">
          <h3 className="rc-section-title">Confusion Matrices — All Classification Models</h3>
          <p className="rc-section-sub">
            TN = Correctly predicted On Time &nbsp;|&nbsp;
            TP = Correctly predicted Delayed &nbsp;|&nbsp;
            FP = Predicted Delayed but was On Time &nbsp;|&nbsp;
            FN = Predicted On Time but was Delayed
          </p>
          <div className="rc-cm-grid">
            {clfNames.map(name => (
              <div key={name} className="rc-cm-wrap">
                <p className="rc-cm-title" style={{ color: CLF_COLORS[name] }}>
                  {name}
                  <span className={`rc-badge ${TYPE_BADGE[clfs[name]?.Type]?.cls}`}
                    style={{ marginLeft: '.5rem' }}>
                    {clfs[name]?.Type}
                  </span>
                </p>
                <ConfMatrix cm={cms[name]} color={CLF_COLORS[name]} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ══ SECTION 4: FEATURE IMPORTANCE ══ */}
      {Object.keys(fi).length > 0 && (
        <div className="card rc-section">
          <h3 className="rc-section-title">Feature Importance — Ensemble Models (Task 5)</h3>
          <p className="rc-section-sub">
            Shows which input features have the strongest influence on delay prediction.
            Available for tree-based ensemble models only.
          </p>
          <div className="rc-fi-grid">
            {Object.entries(fi).map(([name, imp]) => (
              <div key={name}>
                <p className="rc-fi-model-name" style={{ color: CLF_COLORS[name] }}>{name}</p>
                <FeatImportance data={imp} color={CLF_COLORS[name]} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ══ SECTION 5: RECOMMENDATION ══ */}
      <div className="card rc-section rc-rec-card" style={{ borderColor: color }}>
        <h3 className="rc-section-title">Task 6 — Final Model Recommendation</h3>
        <div className="rc-rec-body">
          <div className="rc-rec-model" style={{ background: color + '18', borderColor: color }}>
            <div className="rc-rec-badge" style={{ background: color }}>★</div>
            <div>
              <p className="rc-rec-name" style={{ color }}>{recommended_model}</p>
              <p className="rc-rec-type">
                {TYPE_BADGE[bestModel?.Type]?.label} — Highest F1-Score
              </p>
            </div>
          </div>
          <p className="rc-rec-reason">{recommendation_reason}</p>
          <div className="rc-rec-grid">
            <div className="rc-rec-item">
              <span className="rc-rec-label">Generalization</span>
              <span>
                Ensemble methods (Random Forest, XGBoost) outperform base learners
                by aggregating multiple trees, reducing overfitting.
              </span>
            </div>
            <div className="rc-rec-item">
              <span className="rc-rec-label">Overfitting vs Underfitting</span>
              <span>
                Decision Tree overfits without ensembling. Logistic Regression
                underfits non-linear patterns. XGBoost balances both.
              </span>
            </div>
            <div className="rc-rec-item">
              <span className="rc-rec-label">Why F1, not Accuracy</span>
              <span>
                Dataset is {delayRate}% delayed — imbalanced. F1 correctly balances
                Precision and Recall for the minority (Delayed) class.
              </span>
            </div>
            <div className="rc-rec-item">
              <span className="rc-rec-label">Computational Complexity</span>
              <span>
                XGBoost trains in 2.9s vs Random Forest 11s — faster ensemble
                with equal or better accuracy.
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ══ WEATHER & CONDITIONS ══ */}
      <div className="card rc-section">
        <h3 className="rc-section-title">Live Conditions at Departure</h3>
        <div className="rc-conditions">
          <div className="rc-cond-item">
            <span>Visibility</span>
            <strong>{weather?.visibility ?? '—'} mi</strong>
          </div>
          <div className="rc-cond-item">
            <span>Wind Speed</span>
            <strong>{weather?.windSpeed ?? '—'} mph</strong>
          </div>
          <div className="rc-cond-item">
            <span>Precipitation</span>
            <strong>{weather?.precip ?? '—'} mm</strong>
          </div>
          <div className="rc-cond-item">
            <span>Airport Traffic</span>
            <strong>{congestion?.count ?? '—'} flights/hr</strong>
          </div>
          <div className="rc-cond-item">
            <span>Weather Source</span>
            <strong>{weather?.source ?? '—'}</strong>
          </div>
          <div className="rc-cond-item">
            <span>Traffic Source</span>
            <strong>{congestion?.source ?? '—'}</strong>
          </div>
        </div>

        {alternates?.length > 0 && (
          <div className="rc-alt">
            <p className="rc-alt-label">Lower-risk departure times for this route:</p>
            <div className="rc-alt-pills">
              {alternates.map((a, i) => (
                <span key={i} className="rc-alt-pill">
                  {String(a.hour).padStart(2, '0')}:00 &nbsp;—&nbsp; {a.prob}% delay risk
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

    </div>
  )
}
