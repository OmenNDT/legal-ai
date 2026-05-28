import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/legal-ai/',
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:9010', // Unified Legal AI backend
        changeOrigin: true,
        secure: false,
      }
    }
  }
})