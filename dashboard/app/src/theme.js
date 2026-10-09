// Light, dark, or whatever the OS prefers. The choice is a per-viewer
// convenience, so it lives in localStorage -- wrapped, because storage can
// be blocked (private windows, file:// pages in some browsers) and the page
// must render either way. ?theme=dark in the URL forces one for a session,
// which is how the headless screenshots check both themes.

import { useCallback, useEffect, useState } from 'react'

const KEY = 'kgsage-dashboard-theme'
export const THEMES = ['auto', 'light', 'dark']

function initial() {
  try {
    const forced = new URLSearchParams(window.location.search).get('theme')
    if (THEMES.includes(forced)) return forced
  } catch {
    // no URL to read; fall through
  }
  try {
    const saved = window.localStorage.getItem(KEY)
    if (THEMES.includes(saved)) return saved
  } catch {
    // storage blocked; the default stands
  }
  return 'auto'
}

export function useTheme() {
  const [mode, setMode] = useState(initial)
  useEffect(() => {
    const root = document.documentElement
    if (mode === 'auto') delete root.dataset.theme
    else root.dataset.theme = mode
    try {
      window.localStorage.setItem(KEY, mode)
    } catch {
      // storage blocked; the choice lasts for this page only
    }
    // the graph reads its colours from CSS and must hear about the change
    window.dispatchEvent(new Event('themechange'))
  }, [mode])
  const cycle = useCallback(
    () => setMode((m) => THEMES[(THEMES.indexOf(m) + 1) % THEMES.length]),
    [],
  )
  return [mode, cycle]
}
