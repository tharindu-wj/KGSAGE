// The graph's colours, read from the CSS custom properties in styles.css.
// CSS stays the one place a colour is defined; the SVG receives concrete
// hex values as attributes, which is what makes an exported SVG or PNG
// self-contained (a file opened outside this page has no stylesheet).

import { useEffect, useState } from 'react'

const NAMES = [
  'card', 'ink', 'muted', 'line',
  'g-anchor', 'g-true', 'g-replacement', 'g-shared',
  'g-hop1', 'g-hop2', 'g-edge', 'g-bridge', 'g-halo',
]

export function readPalette() {
  const style = getComputedStyle(document.documentElement)
  const palette = {}
  for (const name of NAMES) palette[name] = style.getPropertyValue(`--${name}`).trim()
  return palette
}

export function usePalette() {
  const [palette, setPalette] = useState(readPalette)
  useEffect(() => {
    const update = () => setPalette(readPalette())
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    media.addEventListener('change', update)
    window.addEventListener('themechange', update)
    return () => {
      media.removeEventListener('change', update)
      window.removeEventListener('themechange', update)
    }
  }, [])
  return palette
}
