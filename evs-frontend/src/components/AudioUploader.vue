<template>
  <div class="upload-area">
    <el-upload
      drag
      action=""
      :auto-upload="false"
      accept="audio/*"
      :on-change="onFileChange"
    >
      <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
      <div class="el-upload__text">
        Drop audio file here or <em>click to browse</em>
      </div>
      <template #tip>
        <div class="el-upload__tip">Supported: .wav, .mp3, .m4a, .flac</div>
      </template>
    </el-upload>

    <div v-if="pendingFile" class="file-config">
      <el-divider />
      <p><strong>{{ pendingFile.name }}</strong> ({{ formatSize(pendingFile.size) }})</p>
      <el-radio-group v-model="selectedLang" style="margin: 12px 0">
        <el-radio-button value="en">English</el-radio-button>
        <el-radio-button value="zh">Chinese</el-radio-button>
      </el-radio-group>
      <el-button
        type="primary"
        :loading="uploading"
        @click="doUpload"
        style="margin-left: 12px"
      >
        Upload
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { UploadFilled } from '@element-plus/icons-vue'
import { useFilesStore } from '../stores/files.js'

const emit = defineEmits(['file-uploaded'])
const filesStore = useFilesStore()
const pendingFile = ref(null)
const selectedLang = ref('en')
const uploading = ref(false)

function onFileChange(uploadFile) {
  pendingFile.value = uploadFile.raw
}

async function handleFileSelect(file, lang) {
  uploading.value = true
  try {
    const result = await filesStore.uploadFile(file, lang)
    emit('file-uploaded', result)
    pendingFile.value = null
  } finally {
    uploading.value = false
  }
}

async function doUpload() {
  if (!pendingFile.value) return
  await handleFileSelect(pendingFile.value, selectedLang.value)
}

function formatSize(bytes) {
  return bytes > 1024 * 1024
    ? `${(bytes / 1024 / 1024).toFixed(1)} MB`
    : `${(bytes / 1024).toFixed(0)} KB`
}

// expose for testing
defineExpose({ handleFileSelect })
</script>

<style scoped>
.upload-area { width: 100%; }
.file-config { margin-top: 12px; }
</style>
