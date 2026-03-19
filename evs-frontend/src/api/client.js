import axios from 'axios'

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  headers: { 'Content-Type': 'application/json' },
})

// Inject JWT token on every request
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('evs_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Redirect to /login on 401
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('evs_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)
