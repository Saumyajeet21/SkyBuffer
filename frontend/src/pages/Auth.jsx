import { useState } from 'react'
import { api } from '../api.js'
import './Auth.css'

export default function Auth({ onLogin }) {
  const [tab,     setTab]     = useState('login')
  const [form,    setForm]    = useState({ email:'', password:'', confirm:'' })
  const [showPw,  setShowPw]  = useState(false)
  const [msg,     setMsg]     = useState(null)
  const [loading, setLoading] = useState(false)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const switchTab = (t) => { setTab(t); setMsg(null); setForm({ email:'', password:'', confirm:'' }) }

  const submit = async (e) => {
    e.preventDefault(); setMsg(null); setLoading(true)
    try {
      if (tab === 'login') {
        const data = await api.login(form.email, form.password)
        onLogin(data.user)
      } else if (tab === 'signup') {
        if (form.password !== form.confirm)
          return setMsg({ type:'error', text:'Passwords do not match' })
        if (form.password.length < 6)
          return setMsg({ type:'error', text:'Password must be at least 6 characters' })
        await api.signup(form.email, form.password)
        setMsg({ type:'success', text:'Account created! Check your email to verify, then sign in.' })
      } else {
        await api.forgotPassword(form.email)
        setMsg({ type:'success', text:'Password reset link sent to your email!' })
      }
    } catch(err) {
      setMsg({ type:'error', text: err.message })
    } finally { setLoading(false) }
  }

  return (
    <div className="auth-root">
      {/* Animated background */}
      <div className="auth-bg-orb orb-1" />
      <div className="auth-bg-orb orb-2" />
      <div className="auth-bg-orb orb-3" />

      <div className="auth-card fade-up">

        {/* Icon + Brand */}
        <div className="auth-brand">
          <div className="auth-icon">✈️</div>
          <h1 className="auth-title">Sky<span>Buffer</span></h1>
          <p className="auth-sub">
            {tab === 'login'  ? 'Sign in to your dashboard'  :
             tab === 'signup' ? 'Create your free account'   :
                               'Reset your password'}
          </p>
        </div>

        {/* Tab toggle — only for login/signup */}
        {tab !== 'forgot' && (
          <div className="auth-toggle">
            <button className={`auth-toggle-btn ${tab==='login'  ? 'active' : ''}`} onClick={() => switchTab('login')}>Sign In</button>
            <button className={`auth-toggle-btn ${tab==='signup' ? 'active' : ''}`} onClick={() => switchTab('signup')}>Sign Up</button>
          </div>
        )}

        {/* Alert */}
        {msg && <div className={`auth-alert ${msg.type}`}>{msg.text}</div>}

        {/* Form */}
        <form onSubmit={submit} className="auth-form">

          {/* Email */}
          <div className="auth-field">
            <label>✉ EMAIL ADDRESS</label>
            <input
              type="email" placeholder="you@example.com" autoComplete="email"
              value={form.email} onChange={e => set('email', e.target.value)} required
            />
          </div>

          {/* Password */}
          {tab !== 'forgot' && (
            <div className="auth-field">
              <label>🔒 PASSWORD</label>
              <div className="auth-pw-wrap">
                <input
                  type={showPw ? 'text' : 'password'} placeholder="••••••••"
                  autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
                  value={form.password} onChange={e => set('password', e.target.value)} required
                />
                <button type="button" className="auth-eye" onClick={() => setShowPw(s => !s)}>
                  {showPw ? '🙈' : '👁'}
                </button>
              </div>
            </div>
          )}

          {/* Confirm password */}
          {tab === 'signup' && (
            <div className="auth-field">
              <label>🔒 CONFIRM PASSWORD</label>
              <div className="auth-pw-wrap">
                <input
                  type={showPw ? 'text' : 'password'} placeholder="••••••••"
                  value={form.confirm} onChange={e => set('confirm', e.target.value)} required
                />
              </div>
            </div>
          )}

          {/* Submit */}
          <button className="auth-submit" type="submit" disabled={loading}>
            {loading
              ? <span className="spinner" style={{ width:20, height:20, borderWidth:2 }} />
              : tab === 'login'  ? 'Sign In →'
              : tab === 'signup' ? 'Create Account →'
              :                    'Send Reset Link →'}
          </button>
        </form>

        {/* Footer links */}
        <div className="auth-links">
          {tab === 'login' && (
            <button className="auth-link" onClick={() => switchTab('forgot')}>Forgot password?</button>
          )}
          {tab === 'forgot' && (
            <button className="auth-link" onClick={() => switchTab('login')}>← Back to Sign In</button>
          )}
          {tab === 'signup' && (
            <p className="auth-link-text">
              Already have an account?{' '}
              <button className="auth-link" onClick={() => switchTab('login')}>Sign In</button>
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
