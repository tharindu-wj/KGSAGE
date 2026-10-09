import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteSingleFile } from 'vite-plugin-singlefile'
import { fileURLToPath } from 'node:url'

// The exporter writes into the dashboard component's own output folder and
// nowhere else (CONTRACTS.md), so outputs/dashboard/ IS this app's static
// root -- not a public/ folder inside the app. Serving it here is what lets
// App.jsx fetch ./data/manifest.json unchanged, in dev and in the built site.
const publicDir = fileURLToPath(
  new URL('../../outputs/dashboard', import.meta.url)
)

// SINGLE=1 npm run build  ->  one self-contained index.html (export.py's
// --single then injects the run data, so no fetch is needed at all).
export default defineConfig({
  base: './',
  publicDir,
  // The app and the data it shows are siblings under the repository root,
  // so the dev server has to be allowed to read above the app folder.
  server: { fs: { allow: ['../..'] } },
  plugins: [react(), ...(process.env.SINGLE ? [viteSingleFile()] : [])],
})
