import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  // Relative asset URLs support deployment under a reverse-proxy subpath.
  base: './',
  plugins: [react()],
  // lucide-react's single CJS bundle is self-contained. Prefer it over the
  // per-icon ESM tree so production builds are robust on shared filesystems
  // where interrupted package extraction can leave individual icon files
  // missing while the bundle itself is intact.
  resolve: {
    alias: {
      'lucide-react': fileURLToPath(
        new URL('./node_modules/lucide-react/dist/cjs/lucide-react.js', import.meta.url),
      ),
    },
  },
  test: {
    environment: 'jsdom',
  },
})
