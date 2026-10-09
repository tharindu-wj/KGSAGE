// Data arrives one of two ways: injected by export.py --single
// (window.__DASHBOARD_DATA__), or fetched from the static root -- which is
// outputs/dashboard/, where the exporter writes -- in dev and in a build.
//
// Routes live in the URL hash, so a case can be linked and reloaded:
//   #/                     the overview
//   #/deck/<key>/<n>       case n (from 1) of one deck
//   #/case/<id>            one case, inside the full docket

import { Component, useCallback, useEffect, useMemo, useState } from 'react'
import Overview from './components/Overview.jsx'
import CaseCard from './components/CaseCard.jsx'
import { buildDecks } from './decks.js'
import { useTheme } from './theme.js'

function readHash() {
  const parts = window.location.hash.replace(/^#\/?/, '').split('/').map(decodeURIComponent)
  if (parts[0] === 'deck' && parts[1]) {
    return { deck: parts[1], index: Math.max(0, (parseInt(parts[2], 10) || 1) - 1) }
  }
  if (parts[0] === 'case' && parts[1]) return { deck: 'all', caseId: parts[1], index: 0 }
  return { deck: null, index: 0 }
}

async function fetchJson(path) {
  const response = await fetch(path)
  if (!response.ok) throw new Error(`could not fetch ${path} (${response.status})`)
  return response.json()
}

class Boundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }
  static getDerivedStateFromError(error) {
    return { error }
  }
  render() {
    if (this.state.error) {
      return (
        <div className="shell">
          <p className="loading" role="alert">
            The dashboard hit an error: {String(this.state.error.message || this.state.error)}
          </p>
        </div>
      )
    }
    return this.props.children
  }
}

const THEME_LABEL = { auto: 'theme: auto', light: 'theme: light', dark: 'theme: dark' }

function Dashboard() {
  const [manifest, setManifest] = useState(null)
  const [runFile, setRunFile] = useState(null)
  const [data, setData] = useState(() => window.__DASHBOARD_DATA__ || null)
  const [error, setError] = useState(null)
  const [route, setRoute] = useState(readHash)
  const [theme, cycleTheme] = useTheme()

  useEffect(() => {
    if (window.__DASHBOARD_DATA__) return
    fetchJson('./data/manifest.json').then((runs) => {
      if (!runs.length) throw new Error('no exported runs in data/manifest.json')
      setManifest(runs)
      setRunFile(runs[0].file)
    }).catch((e) => setError(String(e.message || e)))
  }, [])
  useEffect(() => {
    if (!runFile) return
    setData(null)
    fetchJson(`./data/${runFile}`).then(setData).catch((e) => setError(String(e.message || e)))
  }, [runFile])
  useEffect(() => {
    const onHash = () => setRoute(readHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const decks = useMemo(() => (data ? buildDecks(data) : []), [data])
  const deck = decks.find((d) => d.key === route.deck) || null
  let index = route.index || 0
  if (deck && route.caseId) {
    const found = deck.cases.findIndex((c) => c.id === route.caseId)
    index = found >= 0 ? found : 0
  }
  if (deck) index = Math.min(index, deck.cases.length - 1)

  const go = useCallback((deckKey, i = 0) => {
    window.location.hash = deckKey ? `#/deck/${encodeURIComponent(deckKey)}/${i + 1}` : '#/'
  }, [])

  // opening a deck or returning home starts at the top; paging does not jump
  useEffect(() => { window.scrollTo(0, 0) }, [route.deck])

  useEffect(() => {
    function onKey(event) {
      if (!deck) return
      const tag = event.target && event.target.tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
      if (event.key === 'ArrowRight' && index < deck.cases.length - 1) go(deck.key, index + 1)
      if (event.key === 'ArrowLeft' && index > 0) go(deck.key, index - 1)
      if (event.key === 'Escape') go(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [deck, index, go])

  const toolbar = (
    <div className="toolbar">
      {manifest && manifest.length > 1 && (
        <label className="run-picker">
          run{' '}
          <select value={runFile || ''} onChange={(e) => { setRunFile(e.target.value); go(null) }}>
            {manifest.map((entry) => (
              <option key={entry.file} value={entry.file}>
                {entry.run} ({entry.explained}/{entry.cases} explained)
              </option>
            ))}
          </select>
        </label>
      )}
      <button className="ghost" onClick={cycleTheme}>{THEME_LABEL[theme]}</button>
    </div>
  )

  if (error) {
    return (
      <div className="shell">
        {toolbar}
        <p className="loading" role="alert">
          Could not load the dashboard data: {error}. Run dashboard/export.py first.
        </p>
      </div>
    )
  }
  if (!data) return <div className="shell">{toolbar}<p className="loading">loading{'…'}</p></div>

  return (
    <div className="shell">
      {toolbar}
      {!deck && <Overview data={data} decks={decks} onOpenDeck={(key) => go(key, 0)} />}
      {deck && (
        <CaseCard
          data={data}
          deck={deck}
          index={index}
          onPrev={() => go(deck.key, Math.max(index - 1, 0))}
          onNext={() => go(deck.key, Math.min(index + 1, deck.cases.length - 1))}
          onHome={() => go(null)}
        />
      )}
    </div>
  )
}

export default function App() {
  return (
    <Boundary>
      <Dashboard />
    </Boundary>
  )
}
