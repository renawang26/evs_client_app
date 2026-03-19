import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'

vi.mock('../api/client.js', () => ({
  apiClient: {
    post: vi.fn(),
    get: vi.fn(),
  },
}))

// fetchEventSource only used at runtime — mock it
vi.mock('@microsoft/fetch-event-source', () => ({
  fetchEventSource: vi.fn(),
}))

import { useTasksStore } from '../stores/tasks.js'
import { apiClient } from '../api/client.js'

describe('useTasksStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('tasks is empty initially', () => {
    const store = useTasksStore()
    expect(store.tasks).toEqual({})
  })

  it('submitASR creates a task entry', async () => {
    apiClient.post.mockResolvedValue({
      data: { id: 'uuid-1', type: 'asr', status: 'pending', progress: 0 },
    })
    const store = useTasksStore()
    const taskId = await store.submitASR('test.wav', 'en', 'crisperwhisper', 'default')
    expect(taskId).toBe('uuid-1')
    expect(store.tasks['uuid-1'].status).toBe('pending')
  })

  it('getTask fetches and updates task', async () => {
    apiClient.get.mockResolvedValue({
      data: { id: 'uuid-2', status: 'running', progress: 50 },
    })
    const store = useTasksStore()
    await store.getTask('uuid-2')
    expect(store.tasks['uuid-2'].progress).toBe(50)
  })
})
