import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import AudioUploader from '../components/AudioUploader.vue'

vi.mock('../stores/files.js', () => ({
  useFilesStore: () => ({
    uploadFile: vi.fn().mockResolvedValue({ id: 1, file_name: 'test.wav' }),
    loading: false,
  }),
}))

describe('AudioUploader', () => {
  it('renders upload area', () => {
    const wrapper = mount(AudioUploader, {
      global: { plugins: [createPinia(), ElementPlus] },
    })
    expect(wrapper.find('.upload-area').exists()).toBe(true)
  })

  it('emits file-uploaded after successful upload', async () => {
    const wrapper = mount(AudioUploader, {
      global: { plugins: [createPinia(), ElementPlus] },
    })
    const file = new File(['audio'], 'test.wav', { type: 'audio/wav' })
    await wrapper.vm.handleFileSelect(file, 'en')
    expect(wrapper.emitted('file-uploaded')).toBeTruthy()
  })
})
