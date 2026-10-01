import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'
import { pragmaApi } from './server/pragmaApi.ts'

// Локальный сервер: только 127.0.0.1. /api читает телеметрию
// прогонов из ../data (или PRAGMA_DATA_DIR).
export default defineConfig({
  plugins: [react(), pragmaApi()],
  server: { host: '127.0.0.1', port: 5173 },
  preview: { host: '127.0.0.1', port: 4173 },
  build: { chunkSizeWarningLimit: 2000 },
  test: {
    include: ['src/**/*.test.ts', 'server/**/*.test.ts'],
    environment: 'node',
  },
})
