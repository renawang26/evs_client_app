import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  test: {
    environment: 'jsdom',
    globals: true,
  },
  server: {
    proxy: {
      '/auth': 'http://localhost:8000',
      '/files': 'http://localhost:8000',
      '/tasks': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
