import { defineStore } from 'pinia'
import { ref } from 'vue'
import { apiClient } from '../api/client.js'

export const useFilesStore = defineStore('files', () => {
  const files = ref([])
  const loading = ref(false)

  async function fetchFiles() {
    loading.value = true
    try {
      const { data } = await apiClient.get('/files')
      files.value = data
    } finally {
      loading.value = false
    }
  }

  async function uploadFile(file, lang) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('lang', lang)
    const { data } = await apiClient.post('/files/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    files.value.unshift(data)
    return data
  }

  return { files, loading, fetchFiles, uploadFile }
})
