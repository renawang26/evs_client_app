import { describe, it, expect, beforeEach, vi } from 'vitest'

// Mock axios
vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => ({
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() },
      },
      get: vi.fn(),
      post: vi.fn(),
    })),
  },
}))

describe('apiClient', () => {
  it('creates an axios instance with base URL', async () => {
    const { apiClient } = await import('../api/client.js')
    expect(apiClient).toBeDefined()
  })
})
