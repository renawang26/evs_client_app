import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'

vi.mock('../api/client.js', () => ({
  apiClient: {
    get: vi.fn(),
    post: vi.fn(),
  },
}))

import { useFilesStore } from '../stores/files.js'
import { apiClient } from '../api/client.js'

describe('useFilesStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('files is empty initially', () => {
    const store = useFilesStore()
    expect(store.files).toEqual([])
    expect(store.loading).toBe(false)
  })

  it('fetchFiles populates store', async () => {
    apiClient.get.mockResolvedValue({
      data: [{ id: 1, file_name: 'test.wav', lang: 'en' }],
    })
    const store = useFilesStore()
    await store.fetchFiles()
    expect(store.files).toHaveLength(1)
    expect(store.files[0].file_name).toBe('test.wav')
    expect(store.loading).toBe(false)
  })

  it('uploadFile prepends to files list', async () => {
    apiClient.post.mockResolvedValue({
      data: { id: 2, file_name: 'new.wav', lang: 'zh' },
    })
    const store = useFilesStore()
    const result = await store.uploadFile(new File([''], 'new.wav'), 'zh')
    expect(result.file_name).toBe('new.wav')
    expect(store.files[0].file_name).toBe('new.wav')
  })
})
