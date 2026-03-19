<template>
  <div class="transcribe-view">
    <el-page-header title="Transcribe Audio" @back="$router.push('/files')" />

    <!-- Step 1: Upload -->
    <el-card style="margin-top: 16px">
      <template #header><span>1. Upload Audio File</span></template>
      <AudioUploader @file-uploaded="onFileUploaded" />
    </el-card>

    <!-- Step 2: Configure & Submit -->
    <el-card v-if="uploadedFile" style="margin-top: 16px">
      <template #header><span>2. Configure & Start Transcription</span></template>

      <el-form :model="config" label-width="120px">
        <el-form-item label="File">
          <el-input :value="uploadedFile.file_name" readonly />
        </el-form-item>
        <el-form-item label="Language">
          <el-radio-group v-model="config.lang">
            <el-radio-button value="en">English</el-radio-button>
            <el-radio-button value="zh">Chinese</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="ASR Engine">
          <el-select v-model="config.provider">
            <el-option value="crisperwhisper" label="CrisperWhisper (EN)" />
            <el-option value="funasr" label="FunASR (ZH)" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="submitting" @click="submitTask">
            Start Transcription
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- Step 3: Progress -->
    <el-card v-if="activeTaskId" style="margin-top: 16px">
      <template #header><span>3. Transcription Progress</span></template>
      <TaskProgressBar
        :task-id="activeTaskId"
        :task="tasksStore.tasks[activeTaskId] || { status: 'pending', progress: 0 }"
      />
      <el-alert
        v-if="tasksStore.tasks[activeTaskId]?.status === 'done'"
        title="Transcription complete! Navigate to Files to view results."
        type="success"
        show-icon
        style="margin-top: 12px"
      />
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import AudioUploader from '../components/AudioUploader.vue'
import TaskProgressBar from '../components/TaskProgressBar.vue'
import { useTasksStore } from '../stores/tasks.js'

const tasksStore = useTasksStore()
const uploadedFile = ref(null)
const submitting = ref(false)
const activeTaskId = ref(null)
const config = ref({ lang: 'en', provider: 'crisperwhisper', model: 'default' })

function onFileUploaded(file) {
  uploadedFile.value = file
  // Auto-select provider based on language
  config.value.lang = file.lang
  config.value.provider = file.lang === 'zh' ? 'funasr' : 'crisperwhisper'
}

async function submitTask() {
  if (!uploadedFile.value) return
  submitting.value = true
  try {
    const taskId = await tasksStore.submitASR(
      uploadedFile.value.file_name,
      config.value.lang,
      config.value.provider,
      config.value.model
    )
    activeTaskId.value = taskId
    // Start SSE stream
    tasksStore.streamTask(taskId)
  } finally {
    submitting.value = false
  }
}
</script>

<style scoped>
.transcribe-view { padding: 24px; }
</style>
