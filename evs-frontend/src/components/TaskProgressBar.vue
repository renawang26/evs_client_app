<template>
  <div v-if="task" class="task-progress">
    <div class="task-header">
      <el-tag :type="statusType">{{ task.status }}</el-tag>
      <span class="task-id">{{ taskId.slice(0, 8) }}...</span>
    </div>

    <el-progress
      :percentage="task.progress || 0"
      :status="progressStatus"
      :stroke-width="12"
    />

    <p v-if="task.error" class="task-error">
      <el-icon><Warning /></el-icon> {{ task.error }}
    </p>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { Warning } from '@element-plus/icons-vue'

const props = defineProps({
  taskId: { type: String, required: true },
  task: { type: Object, required: true },
})

const statusType = computed(() => ({
  pending: 'info',
  running: 'warning',
  done: 'success',
  failed: 'danger',
}[props.task.status] || 'info'))

const progressStatus = computed(() => ({
  done: 'success',
  failed: 'exception',
}[props.task.status] || undefined))
</script>

<style scoped>
.task-progress { padding: 12px 0; }
.task-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.task-id { color: #909399; font-size: 12px; font-family: monospace; }
.task-error { color: #f56c6c; font-size: 13px; margin-top: 8px; display: flex; align-items: center; gap: 4px; }
</style>
