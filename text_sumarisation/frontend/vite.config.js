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
        target: 'http://localhost:9020', // Flask backend
        changeOrigin: true,
        secure: false,
      }
    }
  }
})