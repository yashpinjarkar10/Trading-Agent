import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// three@0.165+ ships `three.webgpu.js` with top-level await, which requires an
// esbuild target newer than the default `es2020`. Bumping pre-bundle + build
// targets to `esnext` resolves the "Top-level await is not available" error.
// All modern browsers we care about (Chrome 89+, Edge 89+, FF 89+, Safari 15+)
// support TLA in modules.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
  optimizeDeps: {
    esbuildOptions: {
      target: 'esnext',
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    target: 'esnext',
  },
})
