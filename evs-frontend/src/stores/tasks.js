import { defineStore } from 'pinia'
import { ref } from 'vue'
import { apiClient } from '../api/client.js'
import { fetchEventSource } from '@microsoft/fetch-event-source'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

export const useTasksStore = defineStore('tasks', () => {
  const tasks = ref({})

  async function submitASR(fileName, lang, provider, model) {
    const { data } = await apiClient.post('/tasks/asr', {
      file_name: fileName,
      lang,
      provider,
      model,
    })
    tasks.value[data.id] = data
    return data.id
  }

  async function getTask(taskId) {
    const { data } = await apiClient.get(`/tasks/${taskId}`)
    tasks.value[taskId] = data
    return data
  }

  function streamTask(taskId, onUpdate) {
    const token = localStorage.getItem('evs_token')
    const abortController = new AbortController()

    fetchEventSource(`${API_BASE}/tasks/${taskId}/stream`, {
      headers: { Authorization: `Bearer ${token}` },
      signal: abortController.signal,
      onmessage(event) {
        const update = JSON.parse(event.data)
        tasks.value[taskId] = { ...tasks.value[taskId], ...update }
        if (onUpdate) onUpdate(update)
        if (['done', 'failed'].includes(update.status)) {
          abortController.abort()
        }
      },
      onerror(err) {
        console.error('SSE error:', err)
        abortController.abort()
      },
    })

    return abortController
  }

  return { tasks, submitASR, getTask, streamTask }
})
