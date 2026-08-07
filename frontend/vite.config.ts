import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base:'/app/',
  server: {
    host: true,
    port: 5173,
    allowedHosts: ['transcriber.cloud.cci.charite.de'],
  },
})
