import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'

// Mock apiClient
vi.mock('../api/client.js', () => ({
  apiClient: {
    post: vi.fn(),
  },
}))

import { useAuthStore } from '../stores/auth.js'
import { apiClient } from '../api/client.js'

describe('useAuthStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
    vi.clearAllMocks()
  })

  it('isAuthenticated is false when no token', () => {
    const store = useAuthStore()
    expect(store.isAuthenticated).toBe(false)
  })

  it('login sets token and isAuthenticated', async () => {
    apiClient.post.mockResolvedValue({ data: { access_token: 'tok123' } })
    const store = useAuthStore()
    await store.login('user@test.com', 'pass')
    expect(store.token).toBe('tok123')
    expect(store.isAuthenticated).toBe(true)
    expect(localStorage.getItem('evs_token')).toBe('tok123')
  })

  it('logout clears token', async () => {
    apiClient.post.mockResolvedValue({ data: { access_token: 'tok123' } })
    const store = useAuthStore()
    await store.login('user@test.com', 'pass')
    store.logout()
    expect(store.token).toBeNull()
    expect(store.isAuthenticated).toBe(false)
    expect(localStorage.getItem('evs_token')).toBeNull()
  })
})
