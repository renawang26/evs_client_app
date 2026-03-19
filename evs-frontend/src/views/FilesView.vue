<template>
  <div class="files-view">
    <el-page-header title="Files" />

    <el-table :data="filesStore.files" v-loading="filesStore.loading" style="margin-top: 16px">
      <el-table-column prop="file_name" label="File Name" />
      <el-table-column prop="lang" label="Language" width="100">
        <template #default="{ row }">
          <el-tag>{{ row.lang.toUpperCase() }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="asr_provider" label="Provider" />
      <el-table-column prop="total_segments" label="Segments" width="100" />
      <el-table-column prop="created_at" label="Created" width="180">
        <template #default="{ row }">
          {{ row.created_at ? new Date(row.created_at).toLocaleString() : '—' }}
        </template>
      </el-table-column>
      <el-table-column label="Actions" width="120">
        <template #default="{ row }">
          <el-button size="small" @click="$router.push(`/transcribe?file=${row.file_name}`)">
            Transcribe
          </el-button>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useFilesStore } from '../stores/files.js'

const filesStore = useFilesStore()
onMounted(() => filesStore.fetchFiles())
</script>

<style scoped>
.files-view { padding: 24px; }
</style>
