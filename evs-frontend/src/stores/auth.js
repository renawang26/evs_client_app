import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { apiClient } from '../api/client.js'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('evs_token') || null)
  const isAuthenticated = computed(() => !!token.value)

  async function login(email, password) {
    const { data } = await apiClient.post('/auth/login', { email, password })
    token.value = data.access_token
    localStorage.setItem('evs_token', data.access_token)
    return data
  }

  function logout() {
    token.value = null
    localStorage.removeItem('evs_token')
  }

  return { token, isAuthenticated, login, logout }
})
